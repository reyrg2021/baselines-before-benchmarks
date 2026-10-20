#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 22: Transfer learning with PANNs CNN6 pretrained on AudioSet.

Fine-tunes PANNs CNN6, pretrained on AudioSet (sound) rather than ImageNet, for
4-class classification (Normal/Crackle/Wheeze/Both) of ICBHI respiratory cycles.
Unlike Script 21 (ImageNet backbone on mel spectrograms), it takes raw audio at
32 kHz and uses the PANNs log-mel front end seen during pretraining. It tests
whether audio-domain pretraining improves on the compact CNN trained from
scratch. CNN6 (~5.9M parameters with the head) is the smallest PANNs model,
about 10x the CNN baseline; CNN14 (~80M) is out of the deployment budget.
A --no_pretrained run provides the from-scratch control for the ablation.

Inputs:  data/segments/, data/splits/<split>/, scripts/panns_cnn6.py,
         scripts/raw_audio_dataset.py, scripts/focal_loss.py,
         AudioSet checkpoint Cnn6_mAP=0.343.pth (Zenodo record 3960586; needs torchlibrosa)
Outputs: models/checkpoints/<run_tag>.pth, results/metrics/<run_tag>_metrics.json,
         results/metrics/full_reeval/arrays/<run_tag>_{probs.npy,labels_preds.npz} (--eval_only),
         <run_tag>_log.txt
Usage:
    python 22_panns_transfer.py --seed 42 --split strategy1_official \
        --weights_path scripts/Cnn6_mAP=0.343.pth
    python 22_panns_transfer.py --seed 42 --no_pretrained
    python 22_panns_transfer.py --seed 42 --eval_only
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("panns_transfer_log.txt", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def setup_file_logger(run_tag: str):
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
SEGMENTS_DIR = BASE_DIR / "data" / "segments"
SPLITS_DIR = BASE_DIR / "data" / "splits"

# Default AudioSet checkpoint (can be overridden with --weights_path)
DEFAULT_PANNS_WEIGHTS = SCRIPTS_DIR / "Cnn6_mAP=0.343.pth"


def _import_module(modname, filename):
    from importlib.util import spec_from_file_location, module_from_spec
    here = Path(__file__).resolve().parent
    candidates = [here / filename, SCRIPTS_DIR / filename,
                  BASE_DIR / filename, BASE_DIR / "scripts" / filename]
    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path.parent))
            spec = spec_from_file_location(modname, str(path))
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"Could not find {filename} in known locations: {candidates}")


TRAINING_CONFIG = {
    "split_strategy": "strategy1_official",
    "use_weighted_sampling": True,
    "batch_size": 16,                 # CNN6 on raw audio: 16 fits in 6 GB of VRAM
    "classes_num": 4,

    # Waveform-level mixup (0.0 disables it)
    "mixup_alpha": 0.2,

    # Fine-tuning: low LR for the backbone, higher for the new head
    "learning_rate": 1e-4,            # pretrained backbone LR
    "head_lr_mult": 10.0,
    "weight_decay": 1e-4,
    "freeze_epochs": 2,               # warm up the head before unfreezing the backbone

    "n_epochs": 60,
    "early_stopping_patience": 12,
    "random_seed": 42,
    "run_tag": None,
    "weights_path": None,             # AudioSet .pth (None uses DEFAULT_PANNS_WEIGHTS)
}


def mixup_batch(x, y, alpha):
    """Waveform mixup: x_mix = lam*x + (1-lam)*x[perm]; alpha <= 0 disables it."""
    if alpha is None or alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


class EarlyStopping:
    def __init__(self, patience=12, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
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
            logger.info(f"  EarlyStopping: {self.counter}/{self.patience} "
                        f"(best val_loss: {self.best_loss:.6f} at epoch {self.best_epoch+1})")
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.0):
    model.train()
    running, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        x_in, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
        logits = model(x_in)
        loss = mixup_loss(criterion, logits, y_a, y_b, lam) if lam < 1.0 else criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return running / max(total, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running, correct, total = 0.0, 0, 0
    preds, labels, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        running += loss.item() * x.size(0)
        p = F.softmax(logits, dim=1)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
        probs.extend(p.cpu().numpy())
    return (running / max(total, 1), 100.0 * correct / max(total, 1),
            np.array(preds), np.array(labels), np.array(probs))


# Test metrics use the same criteria as Script 21.
def per_class_metrics(y_true, y_pred, n_classes=4):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
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
    y_bin = np.asarray(y_bin); scores = np.asarray(scores)
    if len(np.unique(y_bin)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_bin, scores))
    except Exception:
        pos = scores[y_bin == 1]; neg = scores[y_bin == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        order = scores.argsort()
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
        return float((ranks[y_bin == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def binary_screening_metrics(y_true, y_pred, probs):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    yt = (y_true != 0).astype(int); yp = (y_pred != 0).astype(int)
    p_abn = 1.0 - np.asarray(probs)[:, 0]
    tp = int(np.sum((yt == 1) & (yp == 1))); tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1))); fn = int(np.sum((yt == 1) & (yp == 0)))
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = (2 * ppv * se / (ppv + se)) if (ppv + se) > 0 else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    return {"se": se, "sp": sp, "ppv": ppv, "npv": npv, "f1": f1,
            "accuracy": acc, "auc_roc": _roc_auc(yt, p_abn)}


def evaluate_test(model, test_loader, device, fl_mod, class_names):
    _, _, preds, labels, probs = validate(model, test_loader, nn.CrossEntropyLoss(), device)
    score, se, sp = fl_mod.official_icbhi(labels, preds)
    mf1, _ = fl_mod.macro_f1(labels, preds)
    pc = per_class_metrics(labels, preds, len(class_names))
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
    logger.info("  Binary Normal-vs-Abnormal screening:")
    logger.info(f"    Se={binm['se']:.3f}  Sp={binm['sp']:.3f}  "
                f"F1={binm['f1']:.3f}  AUC-ROC={auc_str}")
    if binm["auc_roc"] is not None and binm["auc_roc"] < 0.55:
        logger.warning("    [GUARDRAIL] Binary AUC < 0.55: near-chance discrimination. "
                       "A high ICBHI may reflect degenerate collapse, not signal.")

    return {
        "icbhi_official": float(score), "se_official": float(se), "sp_official": float(sp),
        "macro_f1": float(mf1),
        "per_class": {class_names[c]: pc[c] for c in range(len(class_names))},
        "binary": binm, "n_test": int(len(labels)),
    }


def build_optimizer(model, config):
    """AdamW with differential learning rates for backbone and head."""
    base_lr = config["learning_rate"]
    head_mult = config.get("head_lr_mult", 1.0)
    wd = config["weight_decay"]
    groups = [
        {"params": [p for p in model.base_params() if p.requires_grad], "lr": base_lr},
        {"params": list(model.head_params()), "lr": base_lr * head_mult},
    ]
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd)


def train_model(model, train_loader, val_loader, config, device, fl_mod):
    n_epochs = config["n_epochs"]
    mixup_alpha = config.get("mixup_alpha", 0.0)
    freeze_epochs = config.get("freeze_epochs", 0)

    criterion = nn.CrossEntropyLoss()
    if freeze_epochs > 0:
        model.set_base_frozen(True)
        logger.info(f"  Backbone FROZEN for {freeze_epochs} epochs (head warm-up)")
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=4, min_lr=1e-6)
    early = EarlyStopping(patience=config["early_stopping_patience"])

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING START")
    logger.info("=" * 70)
    t0 = time.time()

    for epoch in range(n_epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            model.set_base_frozen(False)
            optimizer = build_optimizer(model, config)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                                   factor=0.5, patience=4, min_lr=1e-6)
            logger.info(f"  Backbone UNFROZEN at epoch {epoch+1} (full fine-tuning)")

        lr_now = optimizer.param_groups[0]["lr"]
        et = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha)
        val_loss, val_acc, vp, vl, _ = validate(model, val_loader, criterion, device)
        val_off, val_se, val_sp = fl_mod.official_icbhi(vl, vp)
        val_mf1, _ = fl_mod.macro_f1(vl, vp)
        scheduler.step(val_loss)

        # Parsed by Script 38: keep the Epoch/Train/Val tokens and formats.
        logger.info(
            f"\n  Epoch {epoch+1:3d}/{n_epochs} ({time.time()-et:.1f}s) | LR: {lr_now:.2e} | "
            f"Train: loss={tr_loss:.4f}, acc={tr_acc:.1f}% | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.1f}% | "
            f"OFFICIAL={val_off:.4f} (Se={val_se:.3f} Sp={val_sp:.3f}) | macroF1={val_mf1:.4f}")
        if val_se < 0.02 and val_sp > 0.95:
            logger.warning("    [COLLAPSE->Normal] Se~0, Sp~1: predicts almost everything as Normal.")
        elif val_sp < 0.05:
            logger.warning("    [COLLAPSE->Abnormal] Sp~0: predicts almost everything as abnormal.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model_state_dict": best_state,
                        "val_loss": best_val_loss, "select_metric": "val_loss",
                        "config": config}, CHECKPOINTS_DIR / f"{config['run_tag']}.pth")
            logger.info(f"  New best model (val_loss={best_val_loss:.6f}, OFFICIAL_val={val_off:.4f})")

        early(val_loss, epoch)
        if early.early_stop:
            logger.info(f"\n  EARLY STOPPING at epoch {epoch+1}")
            break

    logger.info("\n" + "=" * 70)
    logger.info(f"TRAINING COMPLETED | {(time.time()-t0)/60:.1f} min | "
                f"best epoch {best_epoch+1} (val_loss={best_val_loss:.6f})")
    logger.info("=" * 70)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch + 1, "best_val_loss": float(best_val_loss)}


def build_run_tag(config, explicit_tag=None):
    if explicit_tag:
        return explicit_tag
    split_suffix = "_official" if config["split_strategy"] == "strategy1_official" else ""
    return f"panns_cnn6{split_suffix}_seed{config['random_seed']}"


def parse_args():
    p = argparse.ArgumentParser(description="Script 22 - PANNs CNN6 (AudioSet) transfer for ICBHI")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split", choices=["strategy1_official", "strategy2_custom"], default=None)
    p.add_argument("--weights_path", type=str, default=None,
                   help="Path to the AudioSet checkpoint Cnn6_mAP=0.343.pth")
    p.add_argument("--mixup_alpha", type=float, default=None, help="Mixup alpha (0 disables it)")
    p.add_argument("--freeze_epochs", type=int, default=None,
                   help="Initial epochs with a frozen backbone")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="Backbone LR")
    p.add_argument("--head_lr_mult", type=float, default=None, help="Head LR multiplier")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--no_pretrained", action="store_true",
                   help="Do NOT load AudioSet weights (from-scratch control for the ablation)")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--eval_only", action="store_true",
                   help="No training: load a checkpoint, evaluate the official test set and dump "
                        "the per-cycle softmax probabilities (for ROC curves).")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to the .pth evaluated with --eval_only (derived from run_tag if omitted).")
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_only_export(config, args, device):
    """
    Load a trained checkpoint, evaluate the official test set and dump probabilities.

    No retraining. Writes to results/metrics/full_reeval/arrays/:
    {run_tag}_probs.npy (N, 4 per-cycle softmax) and {run_tag}_labels_preds.npz
    (y_true, y_pred), used to build the ROC curves.
    """
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
    logger.info("EVAL-ONLY MODE - dumping per-cycle probabilities (for ROC)")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint: {ckpt_path}")
    logger.info(f"  Run tag:    {run_tag}")

    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    mcfg = {**config, **ck.get("config", {})}

    fl_mod = _import_module("focal_loss", "focal_loss.py")
    pann = _import_module("panns_cnn6", "panns_cnn6.py")
    rad = _import_module("raw_audio_dataset", "raw_audio_dataset.py")

    # Raw-audio test loader, same split as in training
    data = rad.create_raw_audio_dataloaders(
        splits_dir=SPLITS_DIR, segments_dir=SEGMENTS_DIR,
        strategy=mcfg["split_strategy"], batch_size=mcfg.get("batch_size", 16),
        use_weighted_sampling=False)
    test_loader = data["test_loader"]
    class_names = data["class_names"]

    # Model rebuilt from the checkpoint (AudioSet weights are not reloaded)
    model = pann.Transfer_Cnn6(classes_num=mcfg.get("classes_num", 4),
                               freeze_base=False).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    _, _, preds, labels, probs = validate(model, test_loader, nn.CrossEntropyLoss(), device)

    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARRAYS_DIR / f"{run_tag}_probs.npy", probs)
    np.savez(ARRAYS_DIR / f"{run_tag}_labels_preds.npz", y_true=labels, y_pred=preds)
    logger.info(f"  Probabilities:  {ARRAYS_DIR / (run_tag + '_probs.npy')}  shape={probs.shape}")
    logger.info(f"  Labels/preds: {ARRAYS_DIR / (run_tag + '_labels_preds.npz')}")

    # Check: ICBHI/AUC must match the published figures
    score, se, sp = fl_mod.official_icbhi(labels, preds)
    binm = binary_screening_metrics(labels, preds, probs)
    auc_str = f"{binm['auc_roc']:.4f}" if binm["auc_roc"] is not None else "N/A"
    logger.info(f"  [check] ICBHI={score:.4f} (Se={se:.3f} Sp={sp:.3f}) | binary AUC={auc_str}")
    logger.info("  These numbers must match the published results table.")


def main():
    args = parse_args()
    config = TRAINING_CONFIG.copy()
    if args.seed is not None:           config["random_seed"] = args.seed
    if args.split is not None:          config["split_strategy"] = args.split
    if args.weights_path is not None:   config["weights_path"] = args.weights_path
    if args.mixup_alpha is not None:    config["mixup_alpha"] = args.mixup_alpha
    if args.freeze_epochs is not None:  config["freeze_epochs"] = args.freeze_epochs
    if args.epochs is not None:         config["n_epochs"] = args.epochs
    if args.lr is not None:             config["learning_rate"] = args.lr
    if args.head_lr_mult is not None:   config["head_lr_mult"] = args.head_lr_mult
    if args.batch_size is not None:     config["batch_size"] = args.batch_size
    config["pretrained"] = not args.no_pretrained
    if args.no_pretrained:
        # Distinct tag so the from-scratch run does not overwrite the pretrained one
        config["run_tag"] = build_run_tag(config, args.tag) + ("" if args.tag else "_scratch")
    else:
        config["run_tag"] = build_run_tag(config, args.tag)

    setup_file_logger(config["run_tag"])
    set_seed(config["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(args, "eval_only", False):
        eval_only_export(config, args, device)
        return

    logger.info("=" * 70)
    logger.info("Script 22: PANNs CNN6 (AudioSet) transfer")
    logger.info("=" * 70)
    logger.info(f"Run tag:       {config['run_tag']}")
    logger.info(f"Pretrained:    {config['pretrained']} (AudioSet via PANNs CNN6)")
    logger.info(f"Split:         {config['split_strategy']}")
    logger.info(f"mixup_alpha:   {config['mixup_alpha']}")
    logger.info(f"Weighted samp: {config['use_weighted_sampling']}")
    logger.info(f"LR backbone:   {config['learning_rate']} | head x{config['head_lr_mult']} | "
                f"wd {config['weight_decay']}")
    logger.info(f"freeze_epochs: {config['freeze_epochs']} | batch {config['batch_size']}")
    logger.info(f"Seed:          {config['random_seed']} | Device: {device}")

    fl_mod = _import_module("focal_loss", "focal_loss.py")
    pann = _import_module("panns_cnn6", "panns_cnn6.py")
    rad = _import_module("raw_audio_dataset", "raw_audio_dataset.py")

    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: RAW-AUDIO DATALOADERS")
    logger.info("=" * 70)
    data = rad.create_raw_audio_dataloaders(
        splits_dir=SPLITS_DIR, segments_dir=SEGMENTS_DIR,
        strategy=config["split_strategy"], batch_size=config["batch_size"],
        use_weighted_sampling=config["use_weighted_sampling"])
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    class_names = data["class_names"]

    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: PANNs CNN6 MODEL + transfer head")
    logger.info("=" * 70)
    model = pann.Transfer_Cnn6(classes_num=config["classes_num"], freeze_base=False).to(device)

    if config["pretrained"]:
        wpath = Path(config["weights_path"]) if config["weights_path"] else DEFAULT_PANNS_WEIGHTS
        if not wpath.exists():
            logger.error(f"  AudioSet checkpoint not found: {wpath}")
            logger.error("    Download it from Zenodo (record 3960586, Cnn6_mAP=0.343.pth) "
                         "or pass it with --weights_path. Aborting.")
            sys.exit(1)
        logger.info(f"  Loading AudioSet weights from: {wpath}")
        model.load_from_pretrain(str(wpath), logger=logger)
    else:
        logger.warning("  [ABLATION] --no_pretrained: PANNs CNN6 WITHOUT AudioSet weights (random init)")

    n_tot = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model: {model.model_name} | total params: {n_tot:,}")
    logger.info(f"  Comparison: CNN baseline=556,500 -> this model is {n_tot/556500:.1f}x")

    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: TRAINING (fine-tuning)")
    logger.info("=" * 70)
    model, info = train_model(model, train_loader, val_loader, config, device, fl_mod)

    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: TEST EVALUATION")
    logger.info("=" * 70)
    test_metrics = evaluate_test(model, test_loader, device, fl_mod, class_names)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "run_tag": config["run_tag"], "model": "panns_cnn6",
        "pretrained": config["pretrained"], "pretrain_source": "audioset",
        "split": config["split_strategy"], "mixup_alpha": config["mixup_alpha"],
        "seed": config["random_seed"], "n_params": int(n_tot),
        "best_epoch": info["best_epoch"], "best_val_loss": info["best_val_loss"],
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
                f"(Se={test_metrics['se_official']:.3f} Sp={test_metrics['sp_official']:.3f}) | "
                f"Binary AUC: {test_metrics['binary']['auc_roc']}")
    return out


if __name__ == "__main__":
    main()
