#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 11a: CfC training on log-mel spectrograms (transposed and hybrid architectures).

Trains a Closed-form Continuous-time network (CfC-mmRNN, torch_cfc.py) on 4-class ICBHI
respiratory cycles. "transposed" treats the 128 mel bands as timesteps with 251 features
each; "hybrid" uses a 3-block Conv2D frontend followed by a CfC over the time axis.
Regularization (backbone dropout 0.4, weight decay 1e-4, LR 2e-4, batch 64, Dropout2d 0.3
in the conv frontend) is set to curb the fast overfitting seen with lighter settings.

In the hybrid model, frequency is always pooled /2 per block, while the temporal stride per
block is configurable (--temporal_strides): [2,2,2] -> 31 timesteps, [1,2,2] -> 62,
[1,1,2] -> 125, [1,1,1] -> 251. With 31 timesteps each CfC step spans about 256 ms, so a
wheeze contour (80-250 ms) occupies at most one step. Frequency pooling is unchanged, so the
parameter count stays the same across variants; longer sequences increase latency because
the CfC recurrence is sequential. The ICBHI score logged here is the macro one-vs-rest
average, used only for monitoring; the official score is computed by scripts 17 and 18.

Inputs:  data/features/mel_spectrograms/, data/splits/<split>/ (via script 08),
         scripts/torch_cfc.py, scripts/08_dataset_dataloader.py
Outputs: models/checkpoints/best_<run_name>.pth, <run_name>_log.txt (base directory),
         results/metrics/<run_name>_{training_log.csv,config.json,test_summary.json},
         results/figures/<run_name>_training_curves.png, results/tensorboard/
Usage:
    python 11a_cfc_mel_v7.py --architecture {transposed,hybrid} [--seed 42]
                             [--split strategy1_official] [--temporal_strides 1,1,2]
"""

import os
import sys
import json
import time
import copy
import argparse
import logging
import warnings
import importlib.util
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _detect_base_dir():
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    if (script_dir / "data").exists() or (script_dir / "models").exists():
        return script_dir
    parent = script_dir.parent
    if (parent / "data").exists() or (parent / "models").exists():
        return parent
    return script_dir


BASE_DIR = _detect_base_dir()
SCRIPTS_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()

FEATURES_DIR = BASE_DIR / "data" / "features"
MEL_DIR = FEATURES_DIR / "mel_spectrograms"
SPLITS_DIR = BASE_DIR / "data" / "splits"
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
FIGURES_DIR = BASE_DIR / "results" / "figures"
METRICS_DIR = BASE_DIR / "results" / "metrics"
TENSORBOARD_DIR = BASE_DIR / "results" / "tensorboard"

for d in [CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, TENSORBOARD_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def import_torch_cfc():
    """Import torch_cfc.py from the scripts folder or the base directory."""
    candidate_paths = [
        SCRIPTS_DIR / "torch_cfc.py",
        BASE_DIR / "torch_cfc.py",
        BASE_DIR / "scripts" / "torch_cfc.py",
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("torch_cfc", str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module, path
    raise FileNotFoundError(
        "torch_cfc.py not found. Copy it to the scripts/ folder."
    )


def import_script08():
    """Import the script 08 dataloader module, or return None if it is not found."""
    candidate_paths = [
        SCRIPTS_DIR / "08_dataset_dataloader.py",
        BASE_DIR / "08_dataset_dataloader.py",
        BASE_DIR / "scripts" / "08_dataset_dataloader.py",
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location("script08", str(path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


# Hyperparameters follow the CfC-mmRNN (mixed) configuration of Hasani et al. (2022)
# for PhysioNet, with stronger regularization for this dataset.

CONFIG_TRANSPOSED = {
    "architecture": "transposed",
    "model_name": "CfC_Transposed_mmRNN_v7reg",

    # CfC hyperparameters (mixed variant = CfC-mmRNN)
    "backbone_activation": "silu",
    "backbone_units": 128,
    "backbone_layers": 2,
    "backbone_dr": 0.4,
    "init": 0.6,                  # Xavier gain; training is sensitive to it
    "use_mixed": True,            # CfC-mmRNN (LSTM wrapper)
    "no_gate": False,
    "minimal": False,
    "hidden_size": 64,

    # Optimization
    "optimizer": "adamw",
    "base_lr": 2e-4,
    "decay_lr": 0.97,
    "weight_decay": 1e-4,
    "clipnorm": 1.0,

    # Training
    "batch_size": 64,
    "epochs": 80,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 1e-4,
    "random_seed": 42,
    "num_workers": 0,
    "use_weighted_sampling": True,

    # Data
    "split_strategy": "strategy2_custom",
    "class_mode": "4class",
    "n_classes": 4,
    "class_names": ["Normal", "Crackle", "Wheeze", "Both"],

    # CfC geometry
    "cfc_in_features": 251,       # 251 time frames as features per mel band
    "seq_length": 128,            # 128 mel bands as timesteps
}

CONFIG_HYBRID = {
    "architecture": "hybrid",
    "model_name": "CfC_HybridDeep_mmRNN_v7reg",

    # CfC hyperparameters identical to the transposed model for a fair comparison
    "backbone_activation": "silu",
    "backbone_units": 128,
    "backbone_layers": 2,
    "backbone_dr": 0.4,
    "init": 0.6,
    "use_mixed": True,
    "no_gate": False,
    "minimal": False,
    "hidden_size": 64,

    # Optimization
    "optimizer": "adamw",
    "base_lr": 2e-4,
    "decay_lr": 0.97,
    "weight_decay": 1e-4,
    "clipnorm": 1.0,

    # Training
    "batch_size": 64,
    "epochs": 80,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 1e-4,
    "random_seed": 42,
    "num_workers": 0,
    "use_weighted_sampling": True,

    # Data
    "split_strategy": "strategy2_custom",
    "class_mode": "4class",
    "n_classes": 4,
    "class_names": ["Normal", "Crackle", "Wheeze", "Both"],

    # Conv2D frontend (3 blocks)
    "conv_channels": [16, 32, 64],
    "conv_kernel": 3,
    "conv_pool": 2,               # frequency pooling per block (128 -> 16)
    "conv_dropout": 0.3,          # Dropout2d between conv blocks

    # Temporal stride per conv block (overridable with --temporal_strides):
    #   [2,2,2] -> seq 31 | [1,2,2] -> seq 62 | [1,1,2] -> seq 125 | [1,1,1] -> seq 251
    "temporal_strides": [2, 2, 2],

    # Reference values for [2,2,2]: (1, 128, 251) -> (64, 16, 31) -> (31, 1024).
    # The model overwrites both from a symbolic forward pass in __init__.
    "cfc_in_features": 1024,
    "seq_length": 31,
}

CONFIGS = {
    "transposed": CONFIG_TRANSPOSED,
    "hybrid": CONFIG_HYBRID,
}


class CfCMelTransposedV6(nn.Module):
    """
    CfC over mel bands: input (B, 1, 128, 251) -> (B, 128, 251) -> logits (B, n_classes).

    Each mel band is a timestep and its 251 time frames are the features, so the CfC
    models dependencies between adjacent frequency bands.
    """

    def __init__(self, n_classes, config):
        super().__init__()
        torch_cfc, _ = import_torch_cfc()

        self.config = config
        self.cfc = torch_cfc.Cfc(
            in_features=config["cfc_in_features"],
            hidden_size=config["hidden_size"],
            out_feature=n_classes,
            hparams=config,
            return_sequences=False,
            use_mixed=config.get("use_mixed", True),
            use_ltc=False,
        )
        self.model_name = config["model_name"]

    def forward(self, x):
        x = x.squeeze(1)

        # Uniform timespans (constant sampling rate)
        B, T = x.size(0), x.size(1)
        ts = torch.ones(B, T, 1, device=x.device)

        logits = self.cfc(x, timespans=ts)
        return logits


class CfCMelHybridV6(nn.Module):
    """
    Conv2D frontend (3 blocks) followed by a CfC over the time axis.

    With the default temporal strides [2,2,2]:
      (B, 1, 128, 251) -> (B, 16, 64, 125) -> (B, 32, 32, 62) -> (B, 64, 16, 31)
      -> reshape (B, 31, 1024) -> CfC -> (B, n_classes)
    The convolutions extract local spectro-temporal patterns (crackles, wheezes); the CfC
    models their evolution over the respiratory cycle. The sequence length depends on
    temporal_strides and is inferred at construction time.
    """

    def __init__(self, n_classes, config):
        super().__init__()
        torch_cfc, _ = import_torch_cfc()

        self.config = config

        # MaxPool decouples frequency (fixed conv_pool) from time
        # (config["temporal_strides"][i] per block).
        conv_dr = config.get("conv_dropout", 0.0)
        f_pool = config["conv_pool"]
        t_strides = config.get("temporal_strides", [f_pool] * len(config["conv_channels"]))
        assert len(t_strides) == len(config["conv_channels"]), \
            "temporal_strides must have one stride per conv block"
        conv_blocks = []
        in_ch = 1
        for i, out_ch in enumerate(config["conv_channels"]):
            conv_blocks += [
                nn.Conv2d(in_ch, out_ch, config["conv_kernel"], padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(f_pool, t_strides[i]),
                             stride=(f_pool, t_strides[i])),
            ]
            # No Dropout2d after the last block, right before the CfC.
            if conv_dr > 0 and i < len(config["conv_channels"]) - 1:
                conv_blocks.append(nn.Dropout2d(p=conv_dr))
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_blocks)

        # Infer cfc_in_features (C*H) and seq_length (W) from a symbolic forward pass so
        # the CfC input always matches the frontend, and write them back to the config so
        # the log and the saved JSON reflect the actual geometry.
        with torch.no_grad():
            _, _C, _H, _W = self.conv(torch.zeros(1, 1, 128, 251)).shape
        config["cfc_in_features"] = _C * _H
        config["seq_length"] = _W

        self.cfc = torch_cfc.Cfc(
            in_features=config["cfc_in_features"],
            hidden_size=config["hidden_size"],
            out_feature=n_classes,
            hparams=config,
            return_sequences=False,
            use_mixed=config.get("use_mixed", True),
            use_ltc=False,
        )
        self.model_name = config["model_name"]

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape

        # W (time) becomes the sequence axis; channels x frequency become the features.
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H)
        x = x.view(B, W, C * H)                 # (B, W, C*H)

        ts = torch.ones(B, W, 1, device=x.device)
        logits = self.cfc(x, timespans=ts)
        return logits


class EarlyStopping:
    """Stop when val_loss has not improved by min_delta for `patience` epochs."""

    def __init__(self, patience=15, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"  EarlyStopping: {self.counter}/{self.patience} "
                    f"(best val_loss: {self.best_loss:.6f} at epoch {self.best_epoch+1})"
                )
            if self.counter >= self.patience:
                self.early_stop = True


def compute_icbhi_score(y_true, y_pred, n_classes=4):
    """Macro ICBHI score: (mean one-vs-rest Se + mean one-vs-rest Sp) / 2 (monitoring only)."""
    sensitivities = []
    specificities = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        tn = np.sum((y_true != c) & (y_pred != c))
        se = tp / (tp + fn + 1e-9)
        sp = tn / (tn + fp + 1e-9)
        sensitivities.append(se)
        specificities.append(sp)
    se_avg = np.mean(sensitivities)
    sp_avg = np.mean(specificities)
    return (se_avg + sp_avg) / 2.0, se_avg, sp_avg


def per_class_metrics(y_true, y_pred, n_classes=4, class_names=None):
    """Per-class one-vs-rest counts (TP, FN, FP, TN), sensitivity and specificity."""
    results = []
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        se = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        results.append({
            "class": class_names[c],
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "sensitivity": se, "specificity": sp,
        })
    return results


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, config):
    """One training epoch with gradient clipping; returns loss, accuracy (%), preds, labels."""
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"  Train Ep{epoch+1}", leave=False, ncols=100)
    for specs, labels in pbar:
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["clipnorm"])
        optimizer.step()

        running_loss += loss.item() * specs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})

    return running_loss / total, 100. * correct / total, np.array(all_preds), np.array(all_labels)


def validate(model, loader, criterion, device, epoch, split_name="Val"):
    """Evaluate without gradients; returns loss, accuracy (%), preds, labels."""
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"  {split_name} Ep{epoch+1}", leave=False, ncols=100)
        for specs, labels in pbar:
            specs = specs.to(device)
            labels = labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * specs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})

    return running_loss / total, 100. * correct / total, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, config, device, run_name):
    """
    Train with exponential LR decay and early stopping on val_loss.

    The checkpoint with the lowest val_loss is saved to best_<run_name>.pth and its
    weights are restored before returning (model, history, best_epoch).
    """
    criterion = nn.CrossEntropyLoss()

    opt_name = config.get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(),
                                 lr=config["base_lr"],
                                 weight_decay=config["weight_decay"])
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(),
                                lr=config["base_lr"],
                                weight_decay=config["weight_decay"])
    elif opt_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(),
                                   lr=config["base_lr"],
                                   weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Unrecognized optimizer: {opt_name}")

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: config["decay_lr"] ** epoch)

    early_stopper = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
    )

    if HAS_TENSORBOARD:
        tb_dir = TENSORBOARD_DIR / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"  TensorBoard logs: {tb_dir}")
    else:
        writer = None

    best_val_loss = float("inf")
    best_val_icbhi = 0
    best_model_state = None
    best_epoch = 0
    history = []
    checkpoint_path = CHECKPOINTS_DIR / f"best_{run_name}.pth"

    logger.info(f"\n{'='*70}")
    logger.info(f"STARTING TRAINING: {run_name}")
    logger.info(f"{'='*70}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Max epochs: {config['epochs']}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Base LR: {config['base_lr']} (decay {config['decay_lr']}/epoch)")
    logger.info(f"  Early stopping patience: {config['early_stopping_patience']}")
    logger.info(f"{'='*70}\n")

    t_start = time.time()

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        train_loss, train_acc, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch, "Val"
        )
        val_icbhi, val_se, val_sp = compute_icbhi_score(val_labels, val_preds)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1:3d}/{config['epochs']:3d} ({epoch_time:5.1f}s) | "
            f"Train: loss={train_loss:.4f} acc={train_acc:5.1f}% | "
            f"Val: loss={val_loss:.4f} acc={val_acc:5.1f}% | "
            f"ICBHI={val_icbhi:.4f} (Se={val_se:.3f} Sp={val_sp:.3f}) | "
            f"lr={current_lr:.2e}"
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)
            writer.add_scalar("icbhi/val", val_icbhi, epoch)
            writer.add_scalar("sensitivity/val", val_se, epoch)
            writer.add_scalar("specificity/val", val_sp, epoch)
            writer.add_scalar("lr", current_lr, epoch)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_icbhi": val_icbhi, "val_se": val_se, "val_sp": val_sp,
            "lr": current_lr, "epoch_time_s": epoch_time,
        })

        # Model selection by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_icbhi = val_icbhi
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_icbhi": val_icbhi,
                "val_se": val_se,
                "val_sp": val_sp,
                "config": config,
                "model_name": model.model_name,
            }, checkpoint_path)
            logger.info(f"  New best model saved (val_loss={val_loss:.6f}, ICBHI={val_icbhi:.4f})")

        early_stopper(val_loss, epoch)
        if early_stopper.early_stop:
            logger.info(f"\n  Early stopping at epoch {epoch+1} (best: epoch {best_epoch+1})")
            break

    total_time = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETED ({total_time/60:.1f} min)")
    logger.info(f"  Best epoch: {best_epoch+1}")
    logger.info(f"  Best val_loss: {best_val_loss:.6f}")
    logger.info(f"  Best val_ICBHI: {best_val_icbhi:.4f}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"{'='*70}")

    if writer is not None:
        writer.close()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history, best_epoch


def plot_training_curves(history, run_name, save_dir):
    """Plot loss, accuracy and validation ICBHI/Se/Sp per epoch; return the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train", color="steelblue")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val", color="firebrick")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"], label="Train", color="steelblue")
    axes[1].plot(df["epoch"], df["val_acc"], label="Val", color="firebrick")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)"); axes[1].set_title("Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["epoch"], df["val_icbhi"], label="ICBHI Score", color="darkgreen", linewidth=2)
    axes[2].plot(df["epoch"], df["val_se"], label="Sensitivity", color="orange", alpha=0.7)
    axes[2].plot(df["epoch"], df["val_sp"], label="Specificity", color="purple", alpha=0.7)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Score")
    axes[2].set_title("ICBHI Score (Val)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3); axes[2].set_ylim([0, 1])

    plt.suptitle(f"{run_name} - Training curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = Path(save_dir) / f"{run_name}_training_curves.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="CfC training on mel-spectrograms (transposed or hybrid architecture)"
    )
    parser.add_argument(
        "--architecture", type=str, required=True,
        choices=["transposed", "hybrid"],
        help="'transposed': mel bands as timesteps. 'hybrid': Conv2D + CfC."
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--base_lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--temporal_strides", type=str, default=None,
        help="(hybrid only) comma-separated temporal strides per conv block, "
             "e.g. '1,1,2'. Default = config [2,2,2] (seq=31). "
             "[1,2,2]=seq62, [1,1,2]=seq125, [1,1,1]=seq251."
    )
    parser.add_argument(
        "--split", type=str, default=None,
        choices=["strategy1_official", "strategy2_custom"],
        help="Split strategy. Default = config (strategy2_custom, zero leakage). "
             "Use strategy1_official for results COMPARABLE with the literature "
             "(ICBHI 60/40 split; minor leakage of 2 patients: 156, 218)."
    )
    return parser.parse_args()


def setup_logging(run_name):
    """Log to stdout and to <run_name>_log.txt in the base directory."""
    global logger
    log_file = BASE_DIR / f"{run_name}_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    return logger, log_file


def set_seed(seed):
    """Seed NumPy and PyTorch and make cuDNN deterministic."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    # Base configuration plus CLI overrides
    config = copy.deepcopy(CONFIGS[args.architecture])
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.base_lr is not None:
        config["base_lr"] = args.base_lr
    if args.seed is not None:
        config["random_seed"] = args.seed
    if args.temporal_strides is not None:
        config["temporal_strides"] = [int(s) for s in args.temporal_strides.split(",")]
        if args.architecture != "hybrid":
            print("[WARNING] --temporal_strides only affects 'hybrid'; "
                  "ignored for 'transposed' (no conv frontend).")

    if args.split is not None:
        config["split_strategy"] = args.split

    # The temporal-stride signature and the official-split tag are added to the run name
    # so that variants never overwrite each other's checkpoints and logs.
    run_name = f"cfc_v7_mel_{args.architecture}"
    if args.architecture == "hybrid" and args.temporal_strides is not None:
        run_name += f"_t{args.temporal_strides.replace(',', '')}"
    if config.get("split_strategy") == "strategy1_official":
        run_name += "_official"

    setup_logging(run_name)

    logger.info("="*70)
    logger.info("Script 11a: CfC on mel-spectrograms")
    logger.info("="*70)
    logger.info(f"Run name:     {run_name}")
    logger.info(f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Model name:   {config['model_name']}")
    logger.info("="*70)

    set_seed(config["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nDevice: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    try:
        _, torch_cfc_path = import_torch_cfc()
        logger.info(f"\ntorch_cfc.py found: {torch_cfc_path}")
    except FileNotFoundError as e:
        logger.error(f"\n{e}")
        sys.exit(1)

    if not MEL_DIR.exists():
        logger.error(f"\n[ERROR] {MEL_DIR} does not exist. Run script 06 first.")
        sys.exit(1)
    splits_path = SPLITS_DIR / config["split_strategy"]
    if not splits_path.exists():
        logger.error(f"\n[ERROR] {splits_path} does not exist. Run script 07 first.")
        sys.exit(1)

    logger.info("\n[CREATING DATALOADERS]")
    script08 = import_script08()
    if script08 is None:
        logger.error("[ERROR] Script 08 not found")
        sys.exit(1)

    loaders = script08.create_dataloaders(
        strategy=config["split_strategy"],
        class_mode=config["class_mode"],
        augment_preset="none",
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        use_weighted_sampling=config["use_weighted_sampling"],
    )
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    logger.info(f"  Train: {len(train_loader.dataset)} samples -> {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_loader.dataset)} samples -> {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_loader.dataset)} samples -> {len(test_loader)} batches")

    sample_specs, sample_labels = next(iter(train_loader))
    logger.info(f"  Sample shape: {sample_specs.shape}")
    logger.info(f"  Spec range: [{sample_specs.min():.4f}, {sample_specs.max():.4f}]")

    logger.info(f"\n[BUILDING MODEL]")
    logger.info(f"  Architecture:     {args.architecture}")
    logger.info(f"  Model:            {config['model_name']}")
    logger.info(f"  use_mixed:        {config.get('use_mixed', False)} (CfC-mmRNN)")
    logger.info(f"  init (Xavier):    {config.get('init')}")
    logger.info(f"  backbone_units:   {config['backbone_units']}")
    logger.info(f"  backbone_layers:  {config['backbone_layers']}")
    logger.info(f"  backbone_dr:      {config['backbone_dr']}    (raised from 0.2)")
    logger.info(f"  hidden_size:      {config['hidden_size']}")
    logger.info(f"  seq_length:       {config['seq_length']}")
    logger.info(f"  cfc_in_features:  {config['cfc_in_features']}")
    logger.info(f"  optimizer:        {config['optimizer']}")
    logger.info(f"  base_lr:          {config['base_lr']}   (lowered from 5e-4)")
    logger.info(f"  weight_decay:     {config['weight_decay']}    (raised from 1e-5)")
    logger.info(f"  batch_size:       {config['batch_size']}        (raised from 32)")
    if args.architecture == "hybrid":
        logger.info(f"  conv_dropout:     {config.get('conv_dropout', 0)}    (Dropout2d in Conv2D)")
    logger.info(f"  decay_lr:         {config['decay_lr']}")
    logger.info(f"\n  ------------------------------------------------------------")
    logger.info(f"  Regularization relative to lighter settings (anti-overfitting):")
    logger.info(f"    - backbone_dr:  0.2  -> 0.4   (+100% dropout)")
    logger.info(f"    - base_lr:      5e-4 -> 2e-4  (2.5x lower LR)")
    logger.info(f"    - weight_decay: 1e-5 -> 1e-4  (10x more L2)")
    logger.info(f"    - batch_size:   32   -> 64    (more stable gradients)")
    if args.architecture == "hybrid":
        logger.info(f"    - conv_dropout: 0.0  -> 0.3   (Dropout2d between Conv2D blocks)")
    logger.info(f"  ------------------------------------------------------------")

    if args.architecture == "transposed":
        model = CfCMelTransposedV6(n_classes=config["n_classes"], config=config)
    elif args.architecture == "hybrid":
        model = CfCMelHybridV6(n_classes=config["n_classes"], config=config)
    else:
        raise ValueError(f"Invalid architecture: {args.architecture}")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\n  Total parameters: {total_params:,}")
    logger.info(f"  Compared with the CNN baseline (556,500 params):")
    if total_params < 556500:
        logger.info(f"    -> {556500/total_params:.1f}x smaller")
    else:
        logger.info(f"    -> {total_params/556500:.2f}x the CNN size")

    # Shape check on two samples
    logger.info(f"\n[MODEL CHECK]")
    model.eval()
    with torch.no_grad():
        test_input = sample_specs[:2].to(device)
        test_output = model(test_input)
        logger.info(f"  Input shape:  {tuple(test_input.shape)}")
        logger.info(f"  Output shape: {tuple(test_output.shape)} (expected: (2, 4))")

    model, history, best_epoch = train_model(
        model, train_loader, val_loader, config, device, run_name
    )

    history_df = pd.DataFrame(history)
    history_csv = METRICS_DIR / f"{run_name}_training_log.csv"
    history_df.to_csv(history_csv, index=False)
    logger.info(f"\nTraining history saved: {history_csv}")

    try:
        curves_path = plot_training_curves(history, run_name, FIGURES_DIR)
        logger.info(f"Curves saved: {curves_path}")
    except Exception as e:
        logger.warning(f"Could not generate curves: {e}")

    config_path = METRICS_DIR / f"{run_name}_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "config": {k: v for k, v in config.items() if not isinstance(v, np.ndarray)},
            "run_name": run_name,
            "model_name": model.model_name,
            "architecture": args.architecture,
            "total_params": total_params,
            "best_epoch": best_epoch + 1,
            "best_val_loss": min(h["val_loss"] for h in history),
            "best_val_icbhi": max(h["val_icbhi"] for h in history),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)
    logger.info(f"Config saved: {config_path}")

    # Test set evaluation with the best (lowest val_loss) weights
    logger.info(f"\n{'='*70}")
    logger.info(f"TEST SET EVALUATION")
    logger.info(f"{'='*70}")
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, nn.CrossEntropyLoss(), device, best_epoch, "Test"
    )
    test_icbhi, test_se, test_sp = compute_icbhi_score(test_labels, test_preds)
    per_class = per_class_metrics(test_labels, test_preds, n_classes=4,
                                   class_names=config["class_names"])

    logger.info(f"  Test loss:        {test_loss:.4f}")
    logger.info(f"  Test accuracy:    {test_acc:.2f}%")
    logger.info(f"  Test ICBHI Score: {test_icbhi:.4f}")
    logger.info(f"  Test Sensitivity: {test_se:.4f}")
    logger.info(f"  Test Specificity: {test_sp:.4f}")
    logger.info(f"")
    logger.info(f"  Per-class metrics:")
    for r in per_class:
        logger.info(f"    {r['class']:10s}: Se={r['sensitivity']:.3f}  Sp={r['specificity']:.3f}  "
                    f"(TP={r['TP']}, FN={r['FN']}, FP={r['FP']}, TN={r['TN']})")

    # The CNN reference value is a fixed constant from an earlier run, not recomputed here.
    logger.info(f"")
    logger.info(f"  COMPARISON:")
    logger.info(f"    CNN baseline (RespiratoryNet): ICBHI = 0.549")
    logger.info(f"    This model ({args.architecture}):       ICBHI = {test_icbhi:.4f}")
    diff = test_icbhi - 0.549
    if diff > 0:
        logger.info(f"    Difference: +{diff:.4f} ({(diff/0.549)*100:.1f}% BETTER than CNN)")
    else:
        logger.info(f"    Difference: {diff:.4f} ({(diff/0.549)*100:.1f}% worse than CNN)")

    test_summary = {
        "run_name": run_name,
        "architecture": args.architecture,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_icbhi": float(test_icbhi),
        "test_sensitivity": float(test_se),
        "test_specificity": float(test_sp),
        "per_class": per_class,
        "total_params": total_params,
        "best_epoch": best_epoch + 1,
        "vs_cnn_baseline": {
            "cnn_icbhi": 0.549,
            "our_icbhi": float(test_icbhi),
            "delta": float(diff),
            "relative_pct": float((diff / 0.549) * 100),
        },
    }
    test_summary_path = METRICS_DIR / f"{run_name}_test_summary.json"
    with open(test_summary_path, "w") as f:
        json.dump(test_summary, f, indent=2, default=str)
    logger.info(f"\nTest summary saved: {test_summary_path}")

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING ({args.architecture}) COMPLETED")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
