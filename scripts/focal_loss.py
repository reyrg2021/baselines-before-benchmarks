#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared module: focal loss, class weights, and checkpoint-selection metrics.

Used by the CNN (09) and CfC (11a) training scripts. Provides a multiclass focal
loss (Lin et al., 2017), class-weight vectors from class counts (inverse frequency or
effective number of samples, Cui et al., 2019), a criterion factory, and the official
ICBHI score and macro-F1 used to select checkpoints on the validation set.

Rebalancing mechanisms must not be stacked. The two clean configurations are
(A) WeightedRandomSampler + FocalLoss(gamma>0, alpha=None), where the sampler handles
class frequency and gamma handles difficulty, or (B) no sampler + FocalLoss(gamma>0,
alpha=balanced), as in Pham et al. (2022). Sampler + alpha together over-compensates.

Usage:
    python focal_loss.py        # runs the self-test
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multiclass focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    gamma >= 0 down-weights easy examples (gamma=0 reduces to cross-entropy; 2 is the
    RetinaNet default). alpha is an optional per-class weight vector of size C.
    Inputs to forward are raw logits (B, C) and integer targets (B,). The 'mean'
    reduction is sum/N (not sum(alpha_t)/N as in CrossEntropyLoss(weight=...)), so that
    gamma=0, alpha=None matches CrossEntropyLoss(reduction='mean') exactly.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            # Registered as a buffer so that .to(device) moves it with the module
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)          # (B, C)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,)
        pt = log_pt.exp()                                  # (B,)

        focal_factor = (1.0 - pt).pow(self.gamma)          # (B,)
        loss = -focal_factor * log_pt                      # (B,)

        if self.alpha is not None:
            at = self.alpha.gather(0, target)              # (B,)
            loss = at * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"unrecognized reduction: {self.reduction}")


def alpha_from_counts(counts, mode: str = "inverse", beta: float = 0.999,
                      normalize: str = "mean"):
    """
    Per-class weight vector from training class counts (order: Normal, Crackle, Wheeze, Both).

    mode 'inverse': alpha_c proportional to 1/n_c. mode 'effective': alpha_c proportional
    to (1 - beta) / (1 - beta^n_c) (Cui et al., 2019), smoother than pure inverse
    frequency and avoids over-weighting very rare classes. normalize 'mean' scales the
    weights to mean 1 so the global loss magnitude is unchanged; 'none' leaves them raw.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"All counts must be > 0, got {counts}")

    if mode == "inverse":
        w = 1.0 / counts
    elif mode == "effective":
        eff_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / eff_num
    else:
        raise ValueError(f"unrecognized mode: {mode}")

    C = len(counts)
    if normalize == "mean":
        w = w * C / w.sum()      # mean = 1
    elif normalize == "sum":
        w = w * C / w.sum()      # sum = C (same as 'mean')
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"unrecognized normalize: {normalize}")

    return w


def build_criterion(config: dict, class_counts=None, device=None, logger=None):
    """
    Return CrossEntropyLoss or FocalLoss according to config.

    Relevant keys: loss ('ce' | 'focal'), focal_gamma (default 2.0), focal_alpha
    ('none' | 'balanced' | 'effective' | list), focal_beta (for 'effective'),
    ce_class_weights. class_counts is required for 'balanced' and 'effective'.
    Without these keys, a plain nn.CrossEntropyLoss() is returned.
    """
    loss_type = str(config.get("loss", "ce")).lower()

    def _log(msg):
        if logger is not None:
            logger.info(msg)

    if loss_type == "ce":
        ce_weight = config.get("ce_class_weights", None)
        if ce_weight is not None:
            wt = torch.tensor(ce_weight, dtype=torch.float32)
            if device is not None:
                wt = wt.to(device)
            _log(f"  Loss: CrossEntropyLoss with weights {list(np.round(ce_weight, 4))}")
            return nn.CrossEntropyLoss(weight=wt)
        _log("  Loss: CrossEntropyLoss (unweighted)")
        return nn.CrossEntropyLoss()

    elif loss_type == "focal":
        gamma = float(config.get("focal_gamma", 2.0))
        alpha_cfg = config.get("focal_alpha", "none")

        alpha_vec = None
        if isinstance(alpha_cfg, (list, tuple, np.ndarray)):
            alpha_vec = np.asarray(alpha_cfg, dtype=np.float64)
            _log(f"  Loss: FocalLoss(gamma={gamma}, alpha=explicit {list(np.round(alpha_vec,4))})")
        elif str(alpha_cfg).lower() in ("balanced", "inverse"):
            if class_counts is None:
                raise ValueError("focal_alpha='balanced' requires class_counts.")
            alpha_vec = alpha_from_counts(class_counts, mode="inverse")
            _log(f"  Loss: FocalLoss(gamma={gamma}, alpha=inverse {list(np.round(alpha_vec,4))})")
        elif str(alpha_cfg).lower() == "effective":
            if class_counts is None:
                raise ValueError("focal_alpha='effective' requires class_counts.")
            beta = float(config.get("focal_beta", 0.999))
            alpha_vec = alpha_from_counts(class_counts, mode="effective", beta=beta)
            _log(f"  Loss: FocalLoss(gamma={gamma}, alpha=effective β={beta} {list(np.round(alpha_vec,4))})")
        else:
            _log(f"  Loss: FocalLoss(gamma={gamma}, alpha=None)")

        crit = FocalLoss(gamma=gamma, alpha=alpha_vec, reduction="mean")
        if device is not None:
            crit = crit.to(device)
        return crit

    else:
        raise ValueError(f"unrecognized config['loss']: {loss_type!r} (use 'ce' or 'focal')")


def official_icbhi(y_true, y_pred, n_classes: int = 4):
    """
    Official 4-class ICBHI score (exact subclass match); returns (score, se, sp).

    Sp = recall of Normal (class 0); Se = abnormal cycles assigned to their own class /
    all abnormal cycles; score = (Se + Sp) / 2.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    normal_total = int(np.sum(y_true == 0))
    sp = float(np.sum((y_true == 0) & (y_pred == 0)) / normal_total) if normal_total > 0 else 0.0

    abn_mask = (y_true != 0)
    abn_total = int(np.sum(abn_mask))
    correct_abn = int(np.sum(abn_mask & (y_true == y_pred)))
    se = float(correct_abn / abn_total) if abn_total > 0 else 0.0

    return (se + sp) / 2.0, se, sp


def macro_f1(y_true, y_pred, n_classes: int = 4):
    """
    Macro-F1 (mean one-vs-rest F1); returns (macro_f1, per_class_f1).

    Robust to collapse: an all-Normal predictor gets F1=0 on the three abnormal classes,
    whereas the official ICBHI score rewards that collapse with 0.5.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(float(f1))
    return float(np.mean(f1s)), f1s


def select_score(metric: str, val_loss: float, y_true=None, y_pred=None):
    """
    Return (value_to_maximize, metric_name) for checkpoint selection.

    metric: 'val_loss' (returns -val_loss), 'macro_f1' (recommended, collapse-robust),
    'val_official' (official ICBHI; rewards all-Normal collapse with 0.5, so not to be
    used alone), or 'val_macro' (one-vs-rest macro ICBHI). Everything is mapped to
    "higher is better" so the training loop can always test `score > best`.
    """
    metric = metric.lower()
    if metric == "val_loss":
        return -float(val_loss), "val_loss"
    elif metric == "macro_f1":
        mf1, _ = macro_f1(y_true, y_pred)
        return mf1, "val_macro_f1"
    elif metric == "val_official":
        score, _, _ = official_icbhi(y_true, y_pred)
        return score, "val_official_icbhi"
    elif metric == "val_macro":
        ses, sps = [], []
        yt, yp = np.asarray(y_true), np.asarray(y_pred)
        for c in range(4):
            tp = np.sum((yt == c) & (yp == c)); fn = np.sum((yt == c) & (yp != c))
            fp = np.sum((yt != c) & (yp == c)); tn = np.sum((yt != c) & (yp != c))
            ses.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            sps.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        return (float(np.mean(ses)) + float(np.mean(sps))) / 2.0, "val_macro_icbhi"
    else:
        raise ValueError(f"unrecognized selection metric: {metric!r}")


if __name__ == "__main__":
    print("=" * 70)
    print("SELF-TEST focal_loss.py")
    print("=" * 70)
    torch.manual_seed(0)
    np.random.seed(0)

    B, C = 64, 4
    logits = torch.randn(B, C)
    target = torch.randint(0, C, (B,))

    # Test 1: gamma=0, alpha=None must equal CrossEntropyLoss
    fl0 = FocalLoss(gamma=0.0, alpha=None, reduction="mean")
    ce = nn.CrossEntropyLoss(reduction="mean")
    a, b = fl0(logits, target).item(), ce(logits, target).item()
    ok1 = abs(a - b) < 1e-6
    print(f"\n[1] gamma=0, alpha=None == CrossEntropy")
    print(f"    Focal={a:.6f}  CE={b:.6f}  diff={abs(a-b):.2e}  [{'OK' if ok1 else 'FAIL'}]")

    # Test 2: gamma>0 down-weights an easy example more than a hard one
    easy = torch.tensor([[10.0, 0.0, 0.0, 0.0]]); easy_t = torch.tensor([0])
    hard = torch.tensor([[0.3, 0.2, 0.25, 0.25]]); hard_t = torch.tensor([0])
    ce_none = nn.CrossEntropyLoss(reduction="none")
    fl_none = FocalLoss(gamma=2.0, alpha=None, reduction="none")
    ce_easy = ce_none(easy, easy_t).item(); fl_easy = fl_none(easy, easy_t).item()
    ce_hard = ce_none(hard, hard_t).item(); fl_hard = fl_none(hard, hard_t).item()
    ratio_easy = fl_easy / ce_easy
    ratio_hard = fl_hard / ce_hard
    ok2 = ratio_easy < ratio_hard and ratio_easy < 0.01
    print(f"\n[2] gamma=2 down-weights the easy example MORE than the hard one")
    print(f"    easy: CE={ce_easy:.5f} Focal={fl_easy:.5f} ratio={ratio_easy:.5f}")
    print(f"    hard: CE={ce_hard:.5f} Focal={fl_hard:.5f} ratio={ratio_hard:.3f}")
    print(f"    [{'OK' if ok2 else 'FAIL'}]")

    # Test 3: alpha up-weights rare classes (approximate strategy2 train counts)
    counts = [2700, 1450, 630, 360]
    a_inv = alpha_from_counts(counts, mode="inverse")
    a_eff = alpha_from_counts(counts, mode="effective", beta=0.999)
    ok3 = (a_inv[3] > a_inv[2] > a_inv[1] > a_inv[0]) and abs(a_inv.mean() - 1.0) < 1e-9
    print(f"\n[3] inverse alpha up-weights rare classes, mean=1")
    print(f"    inverse:   {np.round(a_inv, 3).tolist()}  (Both>Wheeze>Crackle>Normal, mean={a_inv.mean():.3f})")
    print(f"    effective: {np.round(a_eff, 3).tolist()}  (smoother)")
    print(f"    [{'OK' if ok3 else 'FAIL'}]")

    # Test 4: build_criterion returns the expected types
    c_ce = build_criterion({"loss": "ce"})
    c_focal = build_criterion({"loss": "focal", "focal_gamma": 2.0, "focal_alpha": "balanced"},
                              class_counts=counts)
    ok4 = isinstance(c_ce, nn.CrossEntropyLoss) and isinstance(c_focal, FocalLoss)
    print(f"\n[4] build_criterion: ce→CrossEntropyLoss, focal→FocalLoss")
    print(f"    types: {type(c_ce).__name__}, {type(c_focal).__name__}  [{'OK' if ok4 else 'FAIL'}]")

    # Test 5: official_icbhi on a hand-computed case
    # 2 Normal (1 correct), 3 abnormal (Crackle correct, Wheeze and Both wrong)
    yt = np.array([0, 0, 1, 2, 3]); yp = np.array([0, 1, 1, 0, 1])
    score, se, sp = official_icbhi(yt, yp)
    # Sp = 1/2 = 0.5; Se = 1/3 = 0.3333; score = 0.4167
    ok5 = abs(sp - 0.5) < 1e-9 and abs(se - 1/3) < 1e-9 and abs(score - (0.5 + 1/3)/2) < 1e-9
    print(f"\n[5] official_icbhi vs manual computation")
    print(f"    Sp={sp:.4f}(exp 0.5) Se={se:.4f}(exp 0.333) score={score:.4f}(exp 0.4167)  [{'OK' if ok5 else 'FAIL'}]")

    # Test 6: gradients are finite (trainable)
    lg = torch.randn(B, C, requires_grad=True)
    fl = FocalLoss(gamma=2.0, alpha=torch.tensor([1.0, 1.5, 2.0, 3.0]))
    out = fl(lg, target)
    out.backward()
    ok6 = lg.grad is not None and torch.isfinite(lg.grad).all().item()
    print(f"\n[6] finite gradient (trainable)")
    print(f"    loss={out.item():.4f}  finite grad={ok6}  [{'OK' if ok6 else 'FAIL'}]")

    # Test 7: macro_f1 penalizes all-Normal collapse, the official score does not
    yt7 = np.repeat([0, 1, 2, 3], 10)
    yp_perfect = yt7.copy()
    yp_collapse = np.zeros_like(yt7)
    mf1_perfect, _ = macro_f1(yt7, yp_perfect)
    mf1_collapse, _ = macro_f1(yt7, yp_collapse)
    off_collapse, _, _ = official_icbhi(yt7, yp_collapse)
    ok7 = abs(mf1_perfect - 1.0) < 1e-9 and mf1_collapse < 0.2 and abs(off_collapse - 0.5) < 1e-9
    print(f"\n[7] macro_f1 penalizes collapse to Normal (the official score does NOT)")
    print(f"    perfect: macro_f1={mf1_perfect:.3f}")
    print(f"    all-Normal collapse: macro_f1={mf1_collapse:.3f} (low) vs official={off_collapse:.3f} (misleading)")
    print(f"    [{'OK' if ok7 else 'FAIL'}]")

    all_ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7
    print("\n" + "=" * 70)
    print(f"RESULT: {'ALL OK' if all_ok else 'FAILURES FOUND'}")
    print("=" * 70)
