#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 21: Transfer learning with an ImageNet-pretrained backbone (ICBHI 4-class).

Trains a 4-class classifier (Normal / Crackle / Wheeze / Both) on ICBHI mel spectrograms
with a pretrained torchvision backbone (MobileNetV3-Small, ~1.52 M params, or
EfficientNet-B0, ~4.01 M params) instead of a compact CNN trained from scratch
(556,500 params), to test whether large-scale pretraining plus augmentation improves the
task. Regularization: SpecAugment (time/frequency masking, Park et al. 2019) from the
Script 08 augmentation registry, and batch-level mixup (Zhang et al. 2018). Checkpoint
selection by validation loss. The test set is evaluated at the end (official ICBHI score,
per-class metrics, Normal-vs-Abnormal AUC), so the script does not depend on Script 18.
With --no_pretrained the same backbone is trained from random initialization.

Inputs:  data loaders from 08_dataset_dataloader.py (split strategy1_official by default),
         focal_loss.py (official_icbhi, macro_f1)
Outputs: models/checkpoints/{run_tag}.pth, results/metrics/{run_tag}_metrics.json,
         {run_tag}_log.txt (working directory);
         with --eval_only: results/metrics/full_reeval/arrays/{run_tag}_probs.npy and
         {run_tag}_labels_preds.npz
Usage:
    python 21_transfer_learning.py --backbone mobilenet_v3_small --seed 42 \
        --split strategy1_official --augment_preset moderate --mixup_alpha 0.2
    python 21_transfer_learning.py --backbone mobilenet_v3_small --seed 42 --eval_only
"""

import os
import sys
import json
import time
import copy
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.models as tvm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("transfer_learning_log.txt", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def setup_file_logger(run_tag: str):
    """Redirect the file handler to {run_tag}_log.txt (same pattern as Script 09)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            h.close()
    fh = logging.FileHandler(f"{run_tag}_log.txt", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)


BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
METRICS_DIR = BASE_DIR / "results" / "metrics"
ARRAYS_DIR = METRICS_DIR / "full_reeval" / "arrays"   # same destination as Script 18 (ROC curves)


def _import_focal_loss():
    """Import focal_loss.py (official_icbhi, macro_f1) from the known locations."""
    from importlib.util import spec_from_file_location, module_from_spec
    here = Path(__file__).resolve().parent
    candidates = [here / "focal_loss.py", SCRIPTS_DIR / "focal_loss.py",
                  BASE_DIR / "focal_loss.py", BASE_DIR / "scripts" / "focal_loss.py"]
    for path in candidates:
        if path.exists():
            spec = spec_from_file_location("focal_loss", str(path))
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("focal_loss.py not found in the known locations")


def _import_create_dataloaders():
    """Import create_dataloaders() from Script 08 (same mechanism as Script 09)."""
    from importlib.util import spec_from_file_location, module_from_spec
    here = Path(__file__).resolve().parent
    candidates = [here / "08_dataset_dataloader.py",
                  SCRIPTS_DIR / "08_dataset_dataloader.py",
                  BASE_DIR / "08_dataset_dataloader.py"]
    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path.parent))
            spec = spec_from_file_location("dataset_module", str(path))
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.create_dataloaders
    raise FileNotFoundError("08_dataset_dataloader.py not found in the known locations")


TRAINING_CONFIG = {
    # Data / split
    "split_strategy": "strategy1_official",   # comparable with the literature
    "class_mode": "4class",
    "augment_preset": "moderate",             # SpecAugment preset from Script 08
    "use_weighted_sampling": True,            # same as the CNN baseline
    "batch_size": 32,

    # Pretrained backbone
    "backbone": "mobilenet_v3_small",         # 'mobilenet_v3_small' | 'efficientnet_b0'
    "pretrained": True,
    "imagenet_norm": True,                    # normalize the [0,1] mel with ImageNet stats
    "dropout": 0.3,
    "n_classes": 4,

    # Batch-level mixup
    "mixup_alpha": 0.2,                       # 0.0 disables mixup

    # Optimization (fine-tuning, low LR)
    "optimizer": "adamw",
    "learning_rate": 1e-4,
    "head_lr_mult": 1.0,                      # classifier LR = lr * head_lr_mult
    "weight_decay": 1e-4,
    "freeze_epochs": 0,                       # >0: backbone frozen for these epochs (head warm-up)

    # Loop
    "n_epochs": 80,
    "early_stopping_patience": 15,
    "select_metric": "val_loss",
    "random_seed": 42,
    "run_tag": None,                          # generated automatically if not given
}


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BackboneClassifier(nn.Module):
    """
    Pretrained torchvision backbone adapted to single-channel mel spectrograms.

    Input is (B, 1, n_mels, time) in [0, 1]. The channel is repeated to 3 so the
    pretrained first convolution is kept, the input is optionally normalized with
    ImageNet statistics (the distribution seen during pretraining), and the
    classification head is replaced by a new n_classes layer. head_params() and
    backbone_params() allow a differential learning rate.
    """

    def __init__(self, backbone_name="mobilenet_v3_small", n_classes=4,
                 pretrained=True, imagenet_norm=True, dropout=0.3):
        super().__init__()
        self.backbone_name = backbone_name
        self.imagenet_norm = imagenet_norm
        weights = "DEFAULT" if pretrained else None

        if backbone_name == "mobilenet_v3_small":
            net = tvm.mobilenet_v3_small(weights=weights)
            in_f = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_f, n_classes)
            if dropout is not None:
                for i, m in enumerate(net.classifier):
                    if isinstance(m, nn.Dropout):
                        net.classifier[i] = nn.Dropout(dropout)
        elif backbone_name == "efficientnet_b0":
            net = tvm.efficientnet_b0(weights=weights)
            in_f = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_f, n_classes)
            if dropout is not None:
                net.classifier[0] = nn.Dropout(dropout, inplace=True)
        else:
            raise ValueError(
                f"unknown backbone: {backbone_name!r}. "
                f"Options: 'mobilenet_v3_small', 'efficientnet_b0'"
            )

        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.model_name = f"TL_{backbone_name}{'_pretrained' if pretrained else '_scratch'}"

    def forward(self, x):
        if x.size(1) == 1:                 # (B,1,H,W) -> (B,3,H,W)
            x = x.repeat(1, 3, 1, 1)
        if self.imagenet_norm:
            x = (x - self.mean) / self.std
        return self.net(x)

    def head_params(self):
        return self.net.classifier.parameters()

    def backbone_params(self):
        head_ids = {id(p) for p in self.net.classifier.parameters()}
        return [p for p in self.net.parameters() if id(p) not in head_ids]

    def set_backbone_frozen(self, frozen: bool):
        """Freeze or unfreeze everything except the classifier."""
        for p in self.backbone_params():
            p.requires_grad = not frozen


def build_model(config, device):
    """Build the BackboneClassifier from the config."""
    model = BackboneClassifier(
        backbone_name=config["backbone"],
        n_classes=config["n_classes"],
        pretrained=config["pretrained"],
        imagenet_norm=config["imagenet_norm"],
        dropout=config["dropout"],
    ).to(device)
    return model


def mixup_batch(x, y, alpha):
    """
    Mix a batch: x_mix = lam*x + (1-lam)*x[perm].
    Returns (x_mix, y_a, y_b, lam). If alpha <= 0, no mixing (lam = 1).
    """
    if alpha is None or alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    """Mixup loss: lam*L(pred,y_a) + (1-lam)*L(pred,y_b)."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"  EarlyStopping: {self.counter}/{self.patience} "
                    f"(best val_loss: {self.best_loss:.6f} at epoch {self.best_epoch + 1})"
                )
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    mixup_alpha=0.0):
    """One training epoch with optional mixup."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        specs_in, y_a, y_b, lam = mixup_batch(specs, labels, mixup_alpha)
        logits = model(specs_in)
        if lam < 1.0:
            loss = mixup_loss(criterion, logits, y_a, y_b, lam)
        else:
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)
        # Monitoring accuracy against the original labels (approximate under mixup)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, split_name="Val"):
    """Evaluation pass: loss, accuracy, predictions, labels and softmax probabilities."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels, all_probs = [], [], []

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(specs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * specs.size(0)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return (epoch_loss, epoch_acc, np.array(all_preds),
            np.array(all_labels), np.array(all_probs))


def per_class_metrics(y_true, y_pred, n_classes=4):
    """One-vs-rest sensitivity, specificity, PPV and F1 per class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * ppv * se / (ppv + se)) if (ppv + se) > 0 else 0.0
        out[c] = {"se": se, "sp": sp, "ppv": ppv, "f1": f1}
    return out


def _roc_auc(y_bin, scores):
    """ROC AUC with sklearn if available, otherwise via the Mann-Whitney U statistic (exact equivalent)."""
    y_bin = np.asarray(y_bin)
    scores = np.asarray(scores)
    if len(np.unique(y_bin)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_bin, scores))
    except Exception:
        pos = scores[y_bin == 1]
        neg = scores[y_bin == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        order = scores.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        u = ranks[y_bin == 1].sum() - len(pos) * (len(pos) + 1) / 2.0
        return float(u / (len(pos) * len(neg)))


def binary_screening_metrics(y_true, y_pred, probs):
    """
    Binary screening Normal (0) vs Abnormal (1, 2, 3), with P(abnormal) = 1 - P(Normal).
    Returns Se, Sp, PPV, NPV, F1, accuracy and ROC AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    yt_bin = (y_true != 0).astype(int)
    yp_bin = (y_pred != 0).astype(int)
    p_abn = 1.0 - np.asarray(probs)[:, 0]

    tp = int(np.sum((yt_bin == 1) & (yp_bin == 1)))
    tn = int(np.sum((yt_bin == 0) & (yp_bin == 0)))
    fp = int(np.sum((yt_bin == 0) & (yp_bin == 1)))
    fn = int(np.sum((yt_bin == 1) & (yp_bin == 0)))
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = (2 * ppv * se / (ppv + se)) if (ppv + se) > 0 else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    auc = _roc_auc(yt_bin, p_abn)
    return {"se": se, "sp": sp, "ppv": ppv, "npv": npv, "f1": f1,
            "accuracy": acc, "auc_roc": auc}


def evaluate_test(model, test_loader, device, fl_mod, class_names):
    """Evaluate the test set (same metric logic as Scripts 17/18) and return all metrics."""
    _, _, preds, labels, probs = validate(model, test_loader, nn.CrossEntropyLoss(),
                                           device, epoch=-1, split_name="Test")
    score, se, sp = fl_mod.official_icbhi(labels, preds)
    mf1, _ = fl_mod.macro_f1(labels, preds)
    pc = per_class_metrics(labels, preds, n_classes=len(class_names))
    binm = binary_screening_metrics(labels, preds, probs)

    logger.info("\n" + "=" * 70)
    logger.info("TEST SET EVALUATION (comparable with the literature)")
    logger.info("=" * 70)
    logger.info(f"  OFFICIAL ICBHI: {score:.4f}  (Se={se:.3f}  Sp={sp:.3f})")
    logger.info(f"  Macro-F1:      {mf1:.4f}")
    logger.info("  Per class (Se / Sp / PPV / F1):")
    for c, name in enumerate(class_names):
        m = pc[c]
        logger.info(f"    {name:>8s}: Se={m['se']:.3f}  Sp={m['sp']:.3f}  "
                    f"PPV={m['ppv']:.3f}  F1={m['f1']:.3f}")
    auc_str = f"{binm['auc_roc']:.4f}" if binm["auc_roc"] is not None else "N/A"
    logger.info("  Binary screening Normal-vs-Abnormal:")
    logger.info(f"    Se={binm['se']:.3f}  Sp={binm['sp']:.3f}  "
                f"F1={binm['f1']:.3f}  AUC-ROC={auc_str}")
    if binm["auc_roc"] is not None and binm["auc_roc"] < 0.55:
        logger.warning("    [GUARDRAIL] Binary AUC < 0.55: near-chance discrimination. "
                       "A high ICBHI score here may reflect a degenerate collapse.")

    return {
        "icbhi_official": float(score),
        "se_official": float(se),
        "sp_official": float(sp),
        "macro_f1": float(mf1),
        "per_class": {class_names[c]: pc[c] for c in range(len(class_names))},
        "binary": binm,
        "n_test": int(len(labels)),
    }


def build_optimizer(model, config):
    """AdamW with an optional differential LR (head vs backbone)."""
    head_mult = config.get("head_lr_mult", 1.0)
    base_lr = config["learning_rate"]
    wd = config["weight_decay"]
    if abs(head_mult - 1.0) < 1e-9:
        params = model.parameters()
        opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=wd)
    else:
        groups = [
            {"params": list(model.backbone_params()), "lr": base_lr},
            {"params": list(model.head_params()), "lr": base_lr * head_mult},
        ]
        opt = torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd)
    return opt


def train_model(model, train_loader, val_loader, test_loader, config, device,
                fl_mod, class_names):
    """Training loop: mixup, optional backbone freezing, checkpoint selection by val_loss."""
    n_epochs = config["n_epochs"]
    mixup_alpha = config.get("mixup_alpha", 0.0)
    freeze_epochs = config.get("freeze_epochs", 0)

    criterion = nn.CrossEntropyLoss()   # plain CE; mixup handled by mixup_loss
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=config["early_stopping_patience"])

    if freeze_epochs > 0:
        model.set_backbone_frozen(True)
        logger.info(f"  Backbone FROZEN for the first {freeze_epochs} epochs "
                    f"(only the classifier is trained)")

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING START")
    logger.info("=" * 70)
    total_start = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Unfreeze the backbone once freeze_epochs have elapsed
        if freeze_epochs > 0 and epoch == freeze_epochs:
            model.set_backbone_frozen(False)
            optimizer = build_optimizer(model, config)   # rebuilt with all parameters
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
            )
            logger.info(f"  Backbone UNFROZEN at epoch {epoch + 1} (full fine-tuning)")

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            mixup_alpha=mixup_alpha
        )
        val_loss, val_acc, val_preds, val_labels, _ = validate(
            model, val_loader, criterion, device, epoch
        )
        val_off, val_se, val_sp = fl_mod.official_icbhi(val_labels, val_preds)
        val_mf1, _ = fl_mod.macro_f1(val_labels, val_preds)

        scheduler.step(val_loss)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        # Per-epoch line parsed by Script 38: keep the Epoch/Train/Val token format
        logger.info(
            f"\n  Epoch {epoch+1:3d}/{n_epochs} ({epoch_time:.1f}s) | "
            f"LR: {current_lr:.2e} | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.1f}% | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.1f}% | "
            f"OFFICIAL={val_off:.4f} (Se={val_se:.3f} Sp={val_sp:.3f}) | "
            f"macroF1={val_mf1:.4f}"
        )
        # Collapse guardrails (same logic as Script 09)
        if val_se < 0.02 and val_sp > 0.95:
            logger.warning("    [COLLAPSE->Normal] Se≈0, Sp≈1: predicts almost everything as Normal. "
                           "High ICBHI score but useless.")
        elif val_sp < 0.05:
            logger.warning("    [COLLAPSE->Abnormal] Sp≈0: predicts almost everything as abnormal.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": best_state,
                "val_loss": best_val_loss,
                "val_acc": val_acc,
                "select_metric": "val_loss",
                "config": config,
            }
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, CHECKPOINTS_DIR / f"{config['run_tag']}.pth")
            logger.info(f"  New best model saved "
                        f"(val_loss={best_val_loss:.6f}, OFFICIAL_val={val_off:.4f})")

        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            logger.info(f"\n  EARLY STOPPING at epoch {epoch+1}")
            logger.info(f"    Best model: epoch {best_epoch+1} (val_loss={best_val_loss:.6f})")
            break

    total_time = (time.time() - total_start) / 60.0
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info(f"  Total time: {total_time:.1f} min")
    logger.info(f"  Best epoch: {best_epoch+1} (val_loss={best_val_loss:.6f})")

    # Restore the best state for the test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch + 1, "best_val_loss": float(best_val_loss),
                   "history": history}


def build_run_tag(config, explicit_tag=None):
    """Run name; an explicit --tag is used as is.

    The tag includes the regime (pretrained/scratch) so that a --no_pretrained run with
    the same backbone, split and seed does not overwrite the checkpoint, JSON and log of
    the pretrained run.
    """
    if explicit_tag:
        return explicit_tag
    regime = "pretrained" if config.get("pretrained", True) else "scratch"
    split_suffix = "_official" if config["split_strategy"] == "strategy1_official" else ""
    return f"tl_{config['backbone']}_{regime}{split_suffix}_seed{config['random_seed']}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Script 21: Transfer learning (pretrained backbone) for 4-class ICBHI"
    )
    p.add_argument("--backbone", choices=["mobilenet_v3_small", "efficientnet_b0"],
                   default=None, help="Pretrained backbone (default: mobilenet_v3_small)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--split", choices=["strategy1_official", "strategy2_custom"],
                   default=None, help="Split strategy (default: strategy1_official)")
    p.add_argument("--augment_preset",
                   choices=["none", "conservative", "moderate", "aggressive"],
                   default=None, help="Script 08 augmentation preset (default: moderate)")
    p.add_argument("--mixup_alpha", type=float, default=None,
                   help="Mixup alpha (0 disables; default: 0.2)")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Do not load pretrained weights (random init; from-scratch ablation)")
    p.add_argument("--freeze_epochs", type=int, default=None,
                   help="Initial epochs with the backbone frozen (default: 0)")
    p.add_argument("--epochs", type=int, default=None, help="Maximum epochs")
    p.add_argument("--lr", type=float, default=None, help="Base learning rate")
    p.add_argument("--head_lr_mult", type=float, default=None,
                   help="LR multiplier for the new classifier (default: 1.0)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--tag", type=str, default=None, help="Explicit run/checkpoint name")
    p.add_argument("--eval_only", action="store_true",
                   help="No training: load a checkpoint, evaluate the official test set and export "
                        "per-cycle softmax probabilities (for ROC curves).")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to the .pth evaluated with --eval_only (derived from run_tag if omitted).")
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_only_export(config, args, device):
    """
    Load a trained checkpoint, evaluate the official test set and export per-cycle outputs
    without retraining: {run_tag}_probs.npy (N, 4 softmax) and {run_tag}_labels_preds.npz
    (y_true, y_pred) in results/metrics/full_reeval/arrays/, used for the ROC curves.
    """
    # Locate the checkpoint; run_tag is taken from its file name
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        run_tag = ckpt_path.stem
    else:
        run_tag = config["run_tag"]
        ckpt_path = CHECKPOINTS_DIR / f"{run_tag}.pth"
    if not ckpt_path.exists():
        logger.error(f"  Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("EVAL-ONLY MODE: export of per-cycle probabilities (for ROC)")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint: {ckpt_path}")
    logger.info(f"  Run tag:    {run_tag}")

    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    mcfg = {**config, **ck.get("config", {})}   # the config stored in the checkpoint takes precedence

    # Test loader with the same split and preprocessing as training; test is never augmented
    create_dataloaders = _import_create_dataloaders()
    data = create_dataloaders(
        strategy=mcfg["split_strategy"],
        class_mode=mcfg.get("class_mode", "4class"),
        augment_preset="none",
        batch_size=mcfg.get("batch_size", 32),
        use_weighted_sampling=False,
    )
    test_loader = data["test_loader"]
    class_names = data["class_names"]

    fl_mod = _import_focal_loss()
    model = build_model(mcfg, device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    _, _, preds, labels, probs = validate(
        model, test_loader, nn.CrossEntropyLoss(), device, epoch=-1, split_name="Test")

    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARRAYS_DIR / f"{run_tag}_probs.npy", probs)
    np.savez(ARRAYS_DIR / f"{run_tag}_labels_preds.npz", y_true=labels, y_pred=preds)
    logger.info(f"  Probabilities:  {ARRAYS_DIR / (run_tag + '_probs.npy')}  shape={probs.shape}")
    logger.info(f"  Labels/preds: {ARRAYS_DIR / (run_tag + '_labels_preds.npz')}")

    # Consistency check: must match the test metrics of the training run
    score, se, sp = fl_mod.official_icbhi(labels, preds)
    binm = binary_screening_metrics(labels, preds, probs)
    auc_str = f"{binm['auc_roc']:.4f}" if binm["auc_roc"] is not None else "N/A"
    logger.info(f"  [check] ICBHI={score:.4f} (Se={se:.3f} Sp={sp:.3f}) | binary AUC={auc_str}")
    logger.info("  These values must match the test metrics stored for this run.")


def main():
    args = parse_args()
    config = TRAINING_CONFIG.copy()

    # CLI overrides
    if args.backbone is not None:        config["backbone"] = args.backbone
    if args.seed is not None:            config["random_seed"] = args.seed
    if args.split is not None:           config["split_strategy"] = args.split
    if args.augment_preset is not None:  config["augment_preset"] = args.augment_preset
    if args.mixup_alpha is not None:     config["mixup_alpha"] = args.mixup_alpha
    if args.no_pretrained:               config["pretrained"] = False
    if args.freeze_epochs is not None:   config["freeze_epochs"] = args.freeze_epochs
    if args.epochs is not None:          config["n_epochs"] = args.epochs
    if args.lr is not None:              config["learning_rate"] = args.lr
    if args.head_lr_mult is not None:    config["head_lr_mult"] = args.head_lr_mult
    if args.batch_size is not None:      config["batch_size"] = args.batch_size

    config["run_tag"] = build_run_tag(config, args.tag)
    setup_file_logger(config["run_tag"])
    set_seed(config["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(args, "eval_only", False):
        eval_only_export(config, args, device)
        return

    logger.info("=" * 70)
    logger.info("Script 21: TRANSFER LEARNING")
    logger.info("=" * 70)
    logger.info(f"Run tag:       {config['run_tag']}")
    logger.info(f"Backbone:      {config['backbone']} (pretrained={config['pretrained']})")
    logger.info(f"Split:         {config['split_strategy']}")
    logger.info(f"Augmentation:  {config['augment_preset']} (SpecAugment from Script 08)")
    logger.info(f"mixup_alpha:   {config['mixup_alpha']}")
    logger.info(f"Weighted samp: {config['use_weighted_sampling']}")
    logger.info(f"Optimizer:     adamw lr={config['learning_rate']} "
                f"(head_mult={config['head_lr_mult']}) wd={config['weight_decay']}")
    logger.info(f"freeze_epochs: {config['freeze_epochs']}")
    logger.info(f"Selection:     {config['select_metric']}")
    logger.info(f"Seed:          {config['random_seed']}")
    logger.info(f"Device:        {device}")

    logger.info("\n" + "═" * 70)
    logger.info("STEP 1: DATALOADERS (Script 08)")
    logger.info("═" * 70)
    create_dataloaders = _import_create_dataloaders()
    data = create_dataloaders(
        strategy=config["split_strategy"],
        class_mode=config["class_mode"],
        augment_preset=config["augment_preset"],
        batch_size=config["batch_size"],
        use_weighted_sampling=config["use_weighted_sampling"],
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    class_names = data["class_names"]
    config["input_shape"] = data.get("input_shape", None)
    config["class_names"] = class_names
    logger.info(f"  Input shape: {config['input_shape']} | Classes: {class_names}")

    logger.info("\n" + "═" * 70)
    logger.info("STEP 2: MODEL (pretrained backbone)")
    logger.info("═" * 70)
    fl_mod = _import_focal_loss()
    model = build_model(config, device)
    n_tot = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model: {model.model_name}")
    logger.info(f"  Total parameters: {n_tot:,}")
    logger.info(f"  Comparison: CNN baseline=556,500 -> this model is "
                f"{n_tot/556500:.1f}x")

    logger.info("\n" + "═" * 70)
    logger.info("STEP 3: TRAINING")
    logger.info("═" * 70)
    model, train_info = train_model(
        model, train_loader, val_loader, test_loader, config, device, fl_mod, class_names
    )

    logger.info("\n" + "═" * 70)
    logger.info("STEP 4: TEST EVALUATION")
    logger.info("═" * 70)
    test_metrics = evaluate_test(model, test_loader, device, fl_mod, class_names)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "run_tag": config["run_tag"],
        "backbone": config["backbone"],
        "pretrained": config["pretrained"],
        "split": config["split_strategy"],
        "augment_preset": config["augment_preset"],
        "mixup_alpha": config["mixup_alpha"],
        "seed": config["random_seed"],
        "n_params": int(n_tot),
        "best_epoch": train_info["best_epoch"],
        "best_val_loss": train_info["best_val_loss"],
        "test": test_metrics,
    }
    json_path = METRICS_DIR / f"{config['run_tag']}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETED")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint: {CHECKPOINTS_DIR / (config['run_tag'] + '.pth')}")
    logger.info(f"  Metrics:    {json_path}")
    logger.info(f"  Official ICBHI (test): {test_metrics['icbhi_official']:.4f} "
                f"(Se={test_metrics['se_official']:.3f} Sp={test_metrics['sp_official']:.3f})")
    return out


if __name__ == "__main__":
    main()
