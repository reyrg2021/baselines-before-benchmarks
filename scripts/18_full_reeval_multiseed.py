#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 18: Full multi-seed re-evaluation of the trained checkpoints.

Re-runs test-set inference for every {model}_seed*.pth checkpoint and saves the
raw per-cycle artefacts (4x4 confusion matrix, softmax probabilities, argmax
predictions) needed for metrics that cannot be derived from per-class counts:
binary ROC AUC, per-class precision/F1, the seed-averaged confusion matrix and
sample-wise paired McNemar tests (CNN vs CfC-Hybrid). It also recomputes the
official ICBHI score from the fresh confusion matrix as a cross-check of
script 17 (which derives it from stored counts); a mismatch would indicate a
checkpoint-reloading bug or a non-deterministic DataLoader order.

Binary screening collapses Normal -> 0 and {Crackle, Wheeze, Both} -> 1, with
P(abnormal) = 1 - P(Normal) from the 4-class softmax. A Crackle predicted as
Wheeze is still abnormal in binary terms, which cannot be reconstructed from
one-vs-rest metrics, hence the raw probabilities are stored. All metrics are
reported as mean ± std over seeds, never as the best seed.

Inputs:  models/checkpoints/{model}[_official]_seed*.pth, test split loaded via
         08_dataset_dataloader.py; model classes from 09_cnn_training.py and
         11a_cfc_mel_v7.py
Outputs: results/metrics/full_reeval/ (full_reeval_results.json, full_reeval_runs.csv,
         full_reeval_perclass.csv, full_reeval_binary.csv, full_reeval_summary.md,
         arrays/{model}_seed{S}_{confusion,probs}.npy, arrays/{model}_seed{S}_labels_preds.npz),
         full_reeval_log.txt
Usage:
    python 18_full_reeval_multiseed.py [--split strategy1_official] [--models cnn hybrid]
                                       [--seeds 7 42 99] [--checkpoints_dir PATH] [--no_gpu]
"""

import os
import sys
import json
import argparse
import importlib.util
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Paths

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent

if (SCRIPTS_DIR.parent / "models").exists():
    BASE_DIR = SCRIPTS_DIR.parent
elif (SCRIPTS_DIR / "models").exists():
    BASE_DIR = SCRIPTS_DIR
else:
    BASE_DIR = SCRIPTS_DIR

CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
METRICS_DIR = BASE_DIR / "results" / "metrics"
REEVAL_DIR = METRICS_DIR / "full_reeval"
ARRAYS_DIR = REEVAL_DIR / "arrays"
REEVAL_DIR.mkdir(parents=True, exist_ok=True)
ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]

# Supported models. "kind" selects the constructor used when reloading:
#   - cnn -> RespiratoryNet (script 09)
#   - cfc -> CfCMelHybridV6 / CfCMelTransposedV6 (script 11a), from the saved config
MODEL_SPECS = {
    "cnn":        {"display": "CNN-RespiratoryNet", "kind": "cnn",
                   "pattern": "cnn_seed*.pth"},
    "hybrid":     {"display": "CfC-Hybrid",         "kind": "cfc",
                   "pattern": "hybrid_seed*.pth"},
    "transposed": {"display": "CfC-Transposed",     "kind": "cfc",
                   "pattern": "transposed_seed*.pth"},
    # Hybrid variants with less aggressive temporal pooling. The saved config
    # (architecture="hybrid" + temporal_strides) determines seq_length. The
    # "hybrid_t" prefix does not collide with "hybrid_seed*.pth".
    "hybrid_t122": {"display": "CfC-Hybrid seq62",  "kind": "cfc",
                    "pattern": "hybrid_t122_seed*.pth"},
    "hybrid_t112": {"display": "CfC-Hybrid seq125", "kind": "cfc",
                    "pattern": "hybrid_t112_seed*.pth"},
    "hybrid_t111": {"display": "CfC-Hybrid seq251", "kind": "cfc",
                    "pattern": "hybrid_t111_seed*.pth"},
}


# Logging

logger = logging.getLogger(__name__)


def setup_logging():
    """Log to stdout and to full_reeval_log.txt in BASE_DIR."""
    log_file = BASE_DIR / "full_reeval_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return log_file


# Project module imports

def import_module_from_path(name: str, path: Path):
    """Import a project script as a module."""
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_script09():
    """Script 09: RespiratoryNet class."""
    return import_module_from_path("script09", SCRIPTS_DIR / "09_cnn_training.py")


def import_script11a():
    """Script 11a: CfCMelHybridV6 / CfCMelTransposedV6 classes."""
    return import_module_from_path("script11a", SCRIPTS_DIR / "11a_cfc_mel_v7.py")


def import_script08():
    """Script 08: create_dataloaders."""
    return import_module_from_path("script08", SCRIPTS_DIR / "08_dataset_dataloader.py")


# Checkpoint loading

def load_checkpoint_model(checkpoint_path: Path, kind: str,
                          script09, script11a, device: torch.device):
    """
    Rebuild the model from a checkpoint.

    - kind == "cnn": RespiratoryNet with the saved config (same logic as
      scripts 10/16).
    - kind == "cfc": reads config["architecture"] ('hybrid'/'transposed') and
      builds the matching class from script 11a.

    Returns (model.eval(), checkpoint_dict, config).
    """
    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = ck.get("config", {})

    if kind == "cnn":
        model = script09.RespiratoryNet(
            n_classes=config.get("n_classes", 4),
            input_channels=config.get("input_channels", 1),
            base_filters=config.get("base_filters", 16),
            n_conv_blocks=config.get("n_conv_blocks", 4),
            fc_hidden_dim=config.get("fc_hidden_dim", 128),
            dropout_rate=config.get("dropout_rate", 0.5),
        ).to(device)

    elif kind == "cfc":
        arch = config.get("architecture", None)
        n_classes = config.get("n_classes", 4)
        if arch == "hybrid":
            model = script11a.CfCMelHybridV6(n_classes=n_classes, config=config).to(device)
        elif arch == "transposed":
            model = script11a.CfCMelTransposedV6(n_classes=n_classes, config=config).to(device)
        else:
            raise ValueError(
                f"config['architecture'] unknown or missing: {arch!r}. "
                f"Checkpoint: {checkpoint_path.name}"
            )
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck, config


def find_checkpoints(ckpt_dir: Path, pattern: str):
    """
    Find checkpoints matching the pattern and parse the seed from the file name.
    Returns a list of (seed, path) sorted by seed.
    """
    out = []
    for p in sorted(ckpt_dir.glob(pattern)):
        try:
            seed = int(p.stem.split("seed")[-1])
            out.append((seed, p))
        except ValueError:
            logger.warning(f"  Could not parse seed from {p.name}")
    return out


# Inference

def evaluate_with_probs(model: nn.Module, test_loader, device: torch.device):
    """
    Inference over the whole test set.

    Returns:
      y_true : (N,)    true labels
      y_pred : (N,)    argmax of the logits
      probs  : (N, C)  softmax of the logits (for AUC)
    """
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            logits = model(specs)
            p = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(p.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    probs = np.concatenate(all_probs, axis=0)
    return y_true, y_pred, probs


# 4-class metrics

def confusion_matrix_4x4(y_true, y_pred, n_classes=4):
    """Confusion matrix, rows = true, columns = predicted."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_full_metrics(y_true, y_pred, n_classes=4):
    """
    Per-class Se, Sp, precision (PPV) and F1, with TP/FN/FP/TN. Precision
    exposes the Crackle trade-off (high Se at the cost of low precision).
    """
    results = []
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # sensitivity / recall
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # specificity
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # precision / PPV
        f1 = (2 * ppv * se / (ppv + se)) if (ppv + se) > 0 else 0.0
        results.append({
            "class": CLASS_NAMES[c],
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "sensitivity": float(se),
            "specificity": float(sp),
            "precision": float(ppv),
            "f1": float(f1),
        })
    return results


def icbhi_macro_from_perclass(per_class):
    """
    Macro ICBHI score = (mean Se + mean Sp) / 2 over one-vs-rest of the 4 classes,
    as reported by scripts 09/11a/14/16.

    This is not the official ICBHI 2017 metric. It inflates the score because,
    for classes the model rarely predicts (e.g. Wheeze), false positives are ~0
    and one-vs-rest specificity tends to 1.0. Kept only to reproduce the earlier
    number and to measure the inflation.
    """
    se = float(np.mean([pc["sensitivity"] for pc in per_class]))
    sp = float(np.mean([pc["specificity"] for pc in per_class]))
    return (se + sp) / 2.0, se, sp


def icbhi_official_from_confusion(cm):
    """
    Official ICBHI 2017 score (4-class, exact-subclass) from the confusion matrix,
    the metric comparable with the literature:

        Sp_official = Normal recall
                    = CM[0,0] / sum(CM[0,:])
        Se_official = abnormal cycles classified in their own class / all abnormal
                    = (CM[1,1] + CM[2,2] + CM[3,3]) / sum(CM[1:,:])
        Score       = (Se + Sp) / 2

    Computed from fresh inference; it must match script 17 (computed from stored
    counts).
    """
    cm = np.asarray(cm, dtype=float)
    normal_total = cm[0, :].sum()
    abn_total = cm[1:, :].sum()
    sp = cm[0, 0] / normal_total if normal_total > 0 else 0.0
    correct_abn = cm[1, 1] + cm[2, 2] + cm[3, 3]
    se = correct_abn / abn_total if abn_total > 0 else 0.0
    return (se + sp) / 2.0, float(se), float(sp)


# Binary screening: Normal vs Abnormal

def binary_screening_metrics(y_true, y_pred, probs):
    """
    Binary collapse: Normal (class 0) = negative, {1,2,3} = positive (abnormal,
    i.e. refer). P(abnormal) = 1 - P(Normal) from the 4-class softmax gives the AUC.

    Returns a dict with Se, Sp, PPV, NPV, F1, accuracy, ROC AUC and the 2x2 counts.
    """
    yt_bin = (y_true != 0).astype(int)
    yp_bin = (y_pred != 0).astype(int)
    p_abnormal = 1.0 - probs[:, 0]  # continuous score for the ROC

    tp = int(np.sum((yt_bin == 1) & (yp_bin == 1)))
    fn = int(np.sum((yt_bin == 1) & (yp_bin == 0)))
    fp = int(np.sum((yt_bin == 0) & (yp_bin == 1)))
    tn = int(np.sum((yt_bin == 0) & (yp_bin == 0)))

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = (2 * ppv * se / (ppv + se)) if (ppv + se) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)

    auc = _roc_auc(yt_bin, p_abnormal)

    return {
        "sensitivity": float(se),
        "specificity": float(sp),
        "precision": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "accuracy": float(acc),
        "auc_roc": float(auc) if auc is not None else None,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


def _roc_auc(y_bin, scores):
    """
    ROC AUC via sklearn, falling back to the Mann-Whitney U statistic (exactly
    equivalent). Returns None if only one class is present.
    """
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_bin)) < 2:
            return None
        return float(roc_auc_score(y_bin, scores))
    except ImportError:
        pass
    # Fallback: AUC = P(score_pos > score_neg) via ranks
    pos = scores[y_bin == 1]
    neg = scores[y_bin == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _assign_tie_ranks(scores, ranks)
    sum_ranks_pos = ranks[y_bin == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _assign_tie_ranks(scores, ranks):
    """Assign the average rank to tied values, in place."""
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


# Multi-seed aggregation (mean ± std, sample std as in scripts 14/16)

def _mean_std(values):
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": arr.tolist(),
    }


def aggregate_model(per_seed: list) -> dict:
    """Aggregate scalar, per-class and binary metrics over seeds."""
    off_scores = [r["icbhi_official"] for r in per_seed]
    macro_scores = [r["icbhi_macro"] for r in per_seed]
    agg = {
        "n_seeds": len(per_seed),
        "seeds": [r["seed"] for r in per_seed],
        # Official ICBHI 2017 score, comparable with the literature
        "icbhi_official": _mean_std(off_scores),
        "se_official": _mean_std([r["se_official"] for r in per_seed]),
        "sp_official": _mean_std([r["sp_official"] for r in per_seed]),
        # Macro score, kept to measure the inflation
        "icbhi_macro": _mean_std(macro_scores),
        "inflation_macro_minus_official": float(np.mean(macro_scores) - np.mean(off_scores)),
        "sensitivity": _mean_std([r["sensitivity"] for r in per_seed]),
        "specificity": _mean_std([r["specificity"] for r in per_seed]),
        "accuracy": _mean_std([r["accuracy"] for r in per_seed]),
        "params": per_seed[0]["params"],
    }

    per_class_agg = []
    for i, cls in enumerate(CLASS_NAMES):
        per_class_agg.append({
            "class": cls,
            "sensitivity": _mean_std([r["per_class"][i]["sensitivity"] for r in per_seed]),
            "specificity": _mean_std([r["per_class"][i]["specificity"] for r in per_seed]),
            "precision": _mean_std([r["per_class"][i]["precision"] for r in per_seed]),
            "f1": _mean_std([r["per_class"][i]["f1"] for r in per_seed]),
        })
    agg["per_class"] = per_class_agg

    bin_keys = ["sensitivity", "specificity", "precision", "npv", "f1", "accuracy"]
    binary_agg = {k: _mean_std([r["binary"][k] for r in per_seed]) for k in bin_keys}
    aucs = [r["binary"]["auc_roc"] for r in per_seed if r["binary"]["auc_roc"] is not None]
    binary_agg["auc_roc"] = _mean_std(aucs) if aucs else None
    agg["binary"] = binary_agg

    # Seed-averaged 4x4 confusion matrix
    cms = np.stack([r["confusion"] for r in per_seed], axis=0).astype(float)
    agg["confusion_mean"] = cms.mean(axis=0).tolist()
    agg["confusion_std"] = cms.std(axis=0, ddof=1).tolist() if len(per_seed) > 1 else \
        np.zeros_like(cms[0]).tolist()

    return agg


# Paired McNemar test (CNN vs CfC-Hybrid), per seed

def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar test on the sample-wise correct/incorrect outcomes of two models.

    b = #(A correct, B incorrect), c = #(A incorrect, B correct);
    H0: both models have the same error rate. Uses Edwards' continuity
    correction and also reports the exact binomial test, which is more reliable
    when there are few discordant pairs (b + c < 25).

    Returns a dict with b, c, chi2, chi2 p-value and exact p-value when available.
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    out = {"b_AonlyCorrect": b, "c_BonlyCorrect": c, "n_discordant": b + c}

    if (b + c) == 0:
        out["note"] = "No discordant pairs; models identical on this test set"
        out["chi2"] = 0.0
        out["p_value_chi2"] = 1.0
        return out

    # Chi2 with Edwards' continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    out["chi2"] = float(chi2)

    try:
        from scipy import stats
        out["p_value_chi2"] = float(stats.chi2.sf(chi2, df=1))
        try:
            res = stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided")
            out["p_value_exact"] = float(res.pvalue)
        except AttributeError:
            # scipy < 1.7 provides binom_test instead
            out["p_value_exact"] = float(stats.binom_test(min(b, c), b + c, 0.5))
    except ImportError:
        out["p_value_chi2"] = None
        out["note_scipy"] = "scipy not installed; p-values not computed"

    return out


def mcnemar_pooled_and_perseed(runs_a: dict, runs_b: dict, y_true_by_seed: dict):
    """
    Per-seed paired McNemar between two models (A = CNN, B = Hybrid), plus a
    pooled test summing the discordant pairs over seeds.

    runs_a, runs_b: dict seed -> y_pred (np.array)
    y_true_by_seed: dict seed -> y_true (np.array), identical across models.
    """
    common = sorted(set(runs_a.keys()) & set(runs_b.keys()))
    per_seed = {}
    pooled_b, pooled_c = 0, 0
    for s in common:
        yt = y_true_by_seed[s]
        res = mcnemar_test(yt, runs_a[s], runs_b[s])
        per_seed[s] = res
        pooled_b += res["b_AonlyCorrect"]
        pooled_c += res["c_BonlyCorrect"]

    pooled = {"b_AonlyCorrect": pooled_b, "c_BonlyCorrect": pooled_c,
              "n_discordant": pooled_b + pooled_c}
    if (pooled_b + pooled_c) > 0:
        chi2 = (abs(pooled_b - pooled_c) - 1) ** 2 / (pooled_b + pooled_c)
        pooled["chi2"] = float(chi2)
        try:
            from scipy import stats
            pooled["p_value_chi2"] = float(stats.chi2.sf(chi2, df=1))
        except ImportError:
            pooled["p_value_chi2"] = None
    else:
        pooled["chi2"] = 0.0
        pooled["p_value_chi2"] = 1.0

    return {"per_seed": per_seed, "pooled": pooled, "seeds": common}


# Persistence

def save_arrays(model_key: str, seed: int, y_true, y_pred, probs, cm):
    """Save confusion matrix, probabilities and labels/preds per seed (.npy/.npz)."""
    np.save(ARRAYS_DIR / f"{model_key}_seed{seed}_confusion.npy", cm)
    np.save(ARRAYS_DIR / f"{model_key}_seed{seed}_probs.npy", probs)
    np.savez(ARRAYS_DIR / f"{model_key}_seed{seed}_labels_preds.npz",
             y_true=y_true, y_pred=y_pred)


def write_outputs(per_model_seed: dict, aggregated: dict,
                  mcnemar_results: dict, sysinfo: dict):
    """Write the JSON, CSV and Markdown outputs."""

    json_path = REEVAL_DIR / "full_reeval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # confusion is an np.array; convert to list
        serializable = {}
        for mk, runs in per_model_seed.items():
            serializable[mk] = []
            for r in runs:
                rr = dict(r)
                rr["confusion"] = r["confusion"].tolist()
                serializable[mk].append(rr)
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "system": sysinfo,
            "per_model_seed": serializable,
            "aggregated": aggregated,
            "mcnemar_cnn_vs_hybrid": mcnemar_results,
        }, f, indent=2, default=str)
    logger.info(f"  JSON:          {json_path}")

    # One row per (model, seed)
    csv_runs = REEVAL_DIR / "full_reeval_runs.csv"
    with open(csv_runs, "w", encoding="utf-8") as f:
        f.write("model,seed,icbhi_official,se_official,sp_official,icbhi_macro,"
                "accuracy,bin_se,bin_sp,bin_ppv,bin_npv,bin_f1,bin_acc,bin_auc\n")
        for mk, runs in per_model_seed.items():
            for r in runs:
                b = r["binary"]
                auc = f"{b['auc_roc']:.4f}" if b["auc_roc"] is not None else "NA"
                f.write(f"{mk},{r['seed']},{r['icbhi_official']:.4f},"
                        f"{r['se_official']:.4f},{r['sp_official']:.4f},"
                        f"{r['icbhi_macro']:.4f},"
                        f"{r['accuracy']:.4f},"
                        f"{b['sensitivity']:.4f},{b['specificity']:.4f},"
                        f"{b['precision']:.4f},{b['npv']:.4f},"
                        f"{b['f1']:.4f},{b['accuracy']:.4f},{auc}\n")
    logger.info(f"  CSV runs:      {csv_runs}")

    # Per-class (aggregated)
    csv_pc = REEVAL_DIR / "full_reeval_perclass.csv"
    with open(csv_pc, "w", encoding="utf-8") as f:
        f.write("model,class,se_mean,se_std,sp_mean,sp_std,"
                "prec_mean,prec_std,f1_mean,f1_std\n")
        for mk, agg in aggregated.items():
            for pc in agg["per_class"]:
                f.write(f"{mk},{pc['class']},"
                        f"{pc['sensitivity']['mean']:.4f},{pc['sensitivity']['std']:.4f},"
                        f"{pc['specificity']['mean']:.4f},{pc['specificity']['std']:.4f},"
                        f"{pc['precision']['mean']:.4f},{pc['precision']['std']:.4f},"
                        f"{pc['f1']['mean']:.4f},{pc['f1']['std']:.4f}\n")
    logger.info(f"  CSV per-class: {csv_pc}")

    # Binary (aggregated)
    csv_bin = REEVAL_DIR / "full_reeval_binary.csv"
    with open(csv_bin, "w", encoding="utf-8") as f:
        f.write("model,metric,mean,std,min,max\n")
        for mk, agg in aggregated.items():
            b = agg["binary"]
            for metric in ["sensitivity", "specificity", "precision", "npv", "f1", "accuracy"]:
                m = b[metric]
                f.write(f"{mk},{metric},{m['mean']:.4f},{m['std']:.4f},"
                        f"{m['min']:.4f},{m['max']:.4f}\n")
            if b["auc_roc"] is not None:
                m = b["auc_roc"]
                f.write(f"{mk},auc_roc,{m['mean']:.4f},{m['std']:.4f},"
                        f"{m['min']:.4f},{m['max']:.4f}\n")
    logger.info(f"  CSV binary:    {csv_bin}")

    md_path = REEVAL_DIR / "full_reeval_summary.md"
    _write_markdown(md_path, aggregated, mcnemar_results, sysinfo)
    logger.info(f"  MD summary:    {md_path}")

    return md_path


def _write_markdown(md_path, aggregated, mcnemar_results, sysinfo):
    """Write the Markdown summary tables."""
    lines = []
    lines.append("# Full Re-Evaluation Multi-Seed: results\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_  ")
    lines.append(f"_Device: {sysinfo.get('device')} · PyTorch {sysinfo.get('torch_version')}_\n")

    # Table 0: official ICBHI score vs macro, and the inflation
    lines.append("## 0. ICBHI Score: official vs macro\n")
    lines.append("The **official** score (ICBHI 2017) is the one comparable with the literature "
                 "(RespireNet, Pham 2022, etc.). The **macro** score is the earlier number of "
                 "scripts 09/11a/14/16, shown only to measure the inflation. "
                 "The official score here is computed from fresh inference and must match "
                 "script 17 (computed from the stored counts).\n")
    lines.append("| Model | Official ICBHI | Official Se | Official Sp | Macro ICBHI | Inflation (macro−official) |")
    lines.append("|---|---|---|---|---|---|")
    for mk, agg in aggregated.items():
        disp = MODEL_SPECS[mk]["display"]
        off = agg["icbhi_official"]; mac = agg["icbhi_macro"]
        lines.append(
            f"| {disp} | "
            f"**{off['mean']:.3f} ± {off['std']:.3f}** | "
            f"{agg['se_official']['mean']:.3f} ± {agg['se_official']['std']:.3f} | "
            f"{agg['sp_official']['mean']:.3f} ± {agg['sp_official']['std']:.3f} | "
            f"{mac['mean']:.3f} ± {mac['std']:.3f} | "
            f"{agg['inflation_macro_minus_official']:+.3f} |"
        )
    lines.append("")
    lines.append("> **Note:** comparisons with the literature must use the official column, "
                 "not the macro one. The inflation quantifies how much the one-vs-rest "
                 "average over minority classes raised the score.\n")

    # Table 1: per-class metrics with precision/F1
    lines.append("## 1. Per-class metrics (4-class): mean ± std\n")
    lines.append("| Model | Class | Sensitivity | Specificity | Precision (PPV) | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for mk, agg in aggregated.items():
        disp = MODEL_SPECS[mk]["display"]
        for pc in agg["per_class"]:
            lines.append(
                f"| {disp} | {pc['class']} | "
                f"{pc['sensitivity']['mean']:.3f} ± {pc['sensitivity']['std']:.3f} | "
                f"{pc['specificity']['mean']:.3f} ± {pc['specificity']['std']:.3f} | "
                f"{pc['precision']['mean']:.3f} ± {pc['precision']['std']:.3f} | "
                f"{pc['f1']['mean']:.3f} ± {pc['f1']['std']:.3f} |"
            )
    lines.append("")

    lines.append("> **Note:** high sensitivity with low precision for a class (e.g. Crackle) "
                 "means the model over-predicts that class. This is acceptable in triage "
                 "screening, where over-referral is preferable, but it must be reported.\n")

    # Table 2: binary screening
    lines.append("## 2. Binary screening Normal-vs-Abnormal: mean ± std\n")
    lines.append("| Model | Se | Sp | PPV | NPV | F1 | Accuracy | AUC-ROC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for mk, agg in aggregated.items():
        disp = MODEL_SPECS[mk]["display"]
        b = agg["binary"]
        auc = f"{b['auc_roc']['mean']:.3f} ± {b['auc_roc']['std']:.3f}" if b["auc_roc"] else "NA"
        lines.append(
            f"| {disp} | "
            f"{b['sensitivity']['mean']:.3f} ± {b['sensitivity']['std']:.3f} | "
            f"{b['specificity']['mean']:.3f} ± {b['specificity']['std']:.3f} | "
            f"{b['precision']['mean']:.3f} ± {b['precision']['std']:.3f} | "
            f"{b['npv']['mean']:.3f} ± {b['npv']['std']:.3f} | "
            f"{b['f1']['mean']:.3f} ± {b['f1']['std']:.3f} | "
            f"{b['accuracy']['mean']:.3f} ± {b['accuracy']['std']:.3f} | "
            f"{auc} |"
        )
    lines.append("")
    lines.append("> Binary screening is the actual triage decision (normal vs refer). "
                 "Binary sensitivity weighs most: false negatives (missed patients who "
                 "should be referred) must be minimised.\n")

    # Table 3: McNemar CNN vs Hybrid
    lines.append("## 3. Paired McNemar: CNN vs CfC-Hybrid\n")
    lines.append("Compares sample-wise correct/incorrect outcomes (more powerful than "
                 "comparing mean ICBHI scores with n=5 seeds).\n")
    pooled = mcnemar_results.get("pooled", {})
    lines.append("**Pooled (discordant pairs summed over all seeds):**\n")
    lines.append(f"- b (only CNN correct): {pooled.get('b_AonlyCorrect')}")
    lines.append(f"- c (only Hybrid correct): {pooled.get('c_BonlyCorrect')}")
    lines.append(f"- χ² (Edwards): {pooled.get('chi2')}")
    p = pooled.get("p_value_chi2")
    if p is not None:
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        lines.append(f"- p-value: {p:.4f} ({sig})")
    lines.append("")
    lines.append("**Per seed:**\n")
    lines.append("| Seed | b (CNN) | c (Hybrid) | χ² | p-value |")
    lines.append("|---|---|---|---|---|")
    for s, res in mcnemar_results.get("per_seed", {}).items():
        pv = res.get("p_value_chi2")
        pv_str = f"{pv:.4f}" if pv is not None else "NA"
        lines.append(f"| {s} | {res['b_AonlyCorrect']} | {res['c_BonlyCorrect']} | "
                     f"{res.get('chi2', 0):.3f} | {pv_str} |")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


# Main

def main():
    global CHECKPOINTS_DIR

    parser = argparse.ArgumentParser(
        description="Full multi-seed re-evaluation of the trained checkpoints."
    )
    parser.add_argument("--no_gpu", action="store_true",
                        help="Force evaluation on CPU.")
    parser.add_argument("--checkpoints_dir", type=str, default=None,
                        help="Override the checkpoints directory.")
    parser.add_argument("--models", nargs="+", default=["cnn", "hybrid", "transposed"],
                        choices=list(MODEL_SPECS.keys()),
                        help="Subset of models to evaluate. Default: cnn hybrid "
                             "transposed. Reduced-pooling variants: hybrid_t122 (seq62), "
                             "hybrid_t112 (seq125), hybrid_t111 (seq251).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Subset of seeds (default: auto-detected from disk).")
    parser.add_argument("--split", type=str, default="strategy2_custom",
                        choices=["strategy1_official", "strategy2_custom"],
                        help="Split strategy to evaluate. Default strategy2_custom. "
                             "strategy1_official looks for *_official_seed*.pth checkpoints "
                             "and evaluates on the test set of the ICBHI 60/40 split.")
    args = parser.parse_args()

    # Split tag: "" for custom, "_official" for official. It is inserted into the
    # checkpoint patterns (cnn_seed*.pth -> cnn_official_seed*.pth).
    split_tag = "_official" if args.split == "strategy1_official" else ""

    log_file = setup_logging()

    if args.checkpoints_dir is not None:
        CHECKPOINTS_DIR = Path(args.checkpoints_dir)

    logger.info("=" * 70)
    logger.info("Script 18: Full Re-Evaluation Multi-Seed")
    logger.info("=" * 70)
    logger.info(f"  BASE_DIR:      {BASE_DIR}")
    logger.info(f"  Checkpoints:   {CHECKPOINTS_DIR}")
    logger.info(f"  Outputs:       {REEVAL_DIR}")
    logger.info(f"  Split:         {args.split}" +
                (f"  (patterns *_official_seed*.pth)" if split_tag else ""))
    logger.info(f"  Arrays:        {ARRAYS_DIR}")
    logger.info(f"  Log file:      {log_file}")
    logger.info(f"  Models:        {args.models}")

    device = torch.device("cpu") if args.no_gpu else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device:        {device}")
    if device.type == "cuda":
        logger.info(f"  GPU:           {torch.cuda.get_device_name(0)}")

    sysinfo = {
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        sysinfo["gpu"] = torch.cuda.get_device_name(0)

    logger.info("\n[Importing project modules]")
    script09 = script11a = script08 = None
    try:
        # Import modules according to the kind of each requested model, not its
        # exact name, so that hybrid_t* variants also load script 11a.
        need_cnn = any(MODEL_SPECS[m]["kind"] == "cnn" for m in args.models)
        need_cfc = any(MODEL_SPECS[m]["kind"] == "cfc" for m in args.models)
        if need_cnn:
            script09 = import_script09()
            logger.info("  Script 09 (RespiratoryNet) imported")
        if need_cfc:
            script11a = import_script11a()
            logger.info("  Script 11a (CfC Hybrid/Transposed) imported")
        script08 = import_script08()
        logger.info("  Script 08 (DataLoaders) imported")
    except Exception as e:
        logger.error(f"  Error importing modules: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("\n[Creating test DataLoader]")
    loaders = script08.create_dataloaders(
        strategy=args.split,
        class_mode="4class",
        augment_preset="none",
        batch_size=32,
        num_workers=0,
        use_weighted_sampling=False,
    )
    test_loader = loaders["test_loader"]
    n_test = len(test_loader.dataset)
    logger.info(f"  Test set: {n_test} samples, {len(test_loader)} batches")

    # Evaluate every model x seed
    per_model_seed = {}     # model_key -> [run dicts]
    preds_by_model_seed = {}  # model_key -> {seed: y_pred}
    ytrue_by_seed = {}        # seed -> y_true (must match across models)

    for mk in args.models:
        spec = MODEL_SPECS[mk]
        # Insert the split tag: cnn_seed*.pth -> cnn_official_seed*.pth
        # (unchanged for the custom split).
        pattern = spec["pattern"].replace("_seed", f"{split_tag}_seed")
        ckpts = find_checkpoints(CHECKPOINTS_DIR, pattern)
        if args.seeds is not None:
            ckpts = [(s, p) for (s, p) in ckpts if s in args.seeds]
        if not ckpts:
            logger.warning(f"\n[WARN] No checkpoints for '{mk}' "
                           f"(pattern {pattern}). Skipped.")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"MODEL: {spec['display']}  ({len(ckpts)} seeds)")
        logger.info(f"{'='*70}")

        per_model_seed[mk] = []
        preds_by_model_seed[mk] = {}

        for i, (seed, ckpt_path) in enumerate(ckpts, 1):
            logger.info(f"\n[{i}/{len(ckpts)}] {spec['display']} seed={seed}: {ckpt_path.name}")
            try:
                model, ck, config = load_checkpoint_model(
                    ckpt_path, spec["kind"], script09, script11a, device
                )
            except Exception as e:
                logger.error(f"  Error loading checkpoint: {e}")
                import traceback
                traceback.print_exc()
                continue

            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"    Params: {n_params:,}  |  best_epoch(train): {ck.get('epoch','NA')}")

            try:
                y_true, y_pred, probs = evaluate_with_probs(model, test_loader, device)
            except Exception as e:
                logger.error(f"  Error during inference: {e}")
                import traceback
                traceback.print_exc()
                continue

            # The test set (and its order) must be identical across models
            if seed in ytrue_by_seed:
                if not np.array_equal(ytrue_by_seed[seed], y_true):
                    logger.warning(f"    [WARN] y_true differs for seed {seed} across models. "
                                   f"Non-deterministic DataLoader order?")
            else:
                ytrue_by_seed[seed] = y_true

            cm = confusion_matrix_4x4(y_true, y_pred)
            per_class = per_class_full_metrics(y_true, y_pred)
            # Macro (earlier scripts 09/11a/14/16) and official (ICBHI 2017) scores
            icbhi_macro, se_macro, sp_macro = icbhi_macro_from_perclass(per_class)
            icbhi_off, se_off, sp_off = icbhi_official_from_confusion(cm)
            acc = float((y_true == y_pred).mean())
            binary = binary_screening_metrics(y_true, y_pred, probs)

            logger.info(f"    OFFICIAL: ICBHI={icbhi_off:.4f}  Se={se_off:.4f}  Sp={sp_off:.4f}")
            logger.info(f"    Macro:    ICBHI={icbhi_macro:.4f}  (earlier metric; "
                        f"inflation vs official = {icbhi_macro - icbhi_off:+.4f})")
            logger.info(f"    Accuracy: {acc*100:.2f}%")
            logger.info(f"    Binary:   Se={binary['sensitivity']:.4f}  "
                        f"Sp={binary['specificity']:.4f}  "
                        f"AUC={binary['auc_roc']:.4f}" if binary['auc_roc'] is not None
                        else f"    Binary:   Se={binary['sensitivity']:.4f}  AUC=NA")
            logger.info(f"    Per-class precision: " +
                        ", ".join(f"{pc['class']}={pc['precision']:.3f}" for pc in per_class))

            # Arrays reused downstream (figures, bootstrap, TOST)
            save_arrays(mk, seed, y_true, y_pred, probs, cm)

            per_model_seed[mk].append({
                "seed": seed,
                "checkpoint": ckpt_path.name,
                "params": n_params,
                "best_epoch_training": ck.get("epoch", None),
                "icbhi_official": icbhi_off,
                "se_official": se_off,
                "sp_official": sp_off,
                "icbhi_macro": icbhi_macro,
                "sensitivity": se_macro,    # macro Se (compatible with earlier scripts)
                "specificity": sp_macro,    # macro Sp
                "accuracy": acc,
                "per_class": per_class,
                "binary": binary,
                "confusion": cm,
            })
            preds_by_model_seed[mk][seed] = y_pred

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not per_model_seed:
        logger.error("\n[ERROR] No checkpoint could be evaluated.")
        sys.exit(1)

    # Per-model aggregation
    logger.info(f"\n{'='*70}")
    logger.info("STATISTICAL AGGREGATION (mean ± std)")
    logger.info(f"{'='*70}")
    aggregated = {}
    for mk, runs in per_model_seed.items():
        if not runs:
            continue
        agg = aggregate_model(runs)
        aggregated[mk] = agg
        disp = MODEL_SPECS[mk]["display"]
        logger.info(f"\n{disp} ({agg['n_seeds']} seeds):")
        off = agg["icbhi_official"]; mac = agg["icbhi_macro"]
        logger.info(f"  ICBHI OFFICIAL: {off['mean']:.4f} ± {off['std']:.4f}   <- reported metric")
        logger.info(f"  ICBHI Macro:   {mac['mean']:.4f} ± {mac['std']:.4f}   "
                    f"(inflation {agg['inflation_macro_minus_official']:+.4f})")
        logger.info(f"  Official Se:   {agg['se_official']['mean']:.4f} ± {agg['se_official']['std']:.4f}")
        logger.info(f"  Official Sp:   {agg['sp_official']['mean']:.4f} ± {agg['sp_official']['std']:.4f}")
        b = agg["binary"]
        if b["auc_roc"]:
            logger.info(f"  Binary AUC:    {b['auc_roc']['mean']:.4f} ± {b['auc_roc']['std']:.4f}")
        logger.info(f"  Binary Se:     {b['sensitivity']['mean']:.4f} ± {b['sensitivity']['std']:.4f}")
        logger.info(f"  Binary Sp:     {b['specificity']['mean']:.4f} ± {b['specificity']['std']:.4f}")
        logger.info("  Per-class precision (mean):")
        for pc in agg["per_class"]:
            logger.info(f"    {pc['class']:10s}: Se={pc['sensitivity']['mean']:.3f}  "
                        f"PPV={pc['precision']['mean']:.3f}  F1={pc['f1']['mean']:.3f}")

    # McNemar CNN vs Hybrid
    mcnemar_results = {}
    if "cnn" in preds_by_model_seed and "hybrid" in preds_by_model_seed:
        logger.info(f"\n{'='*70}")
        logger.info("PAIRED McNEMAR: CNN vs CfC-Hybrid")
        logger.info(f"{'='*70}")
        mcnemar_results = mcnemar_pooled_and_perseed(
            preds_by_model_seed["cnn"],
            preds_by_model_seed["hybrid"],
            ytrue_by_seed,
        )
        pooled = mcnemar_results["pooled"]
        logger.info(f"  Pooled: b(CNN)={pooled['b_AonlyCorrect']}, "
                    f"c(Hybrid)={pooled['c_BonlyCorrect']}, "
                    f"χ²={pooled['chi2']:.3f}")
        if pooled.get("p_value_chi2") is not None:
            sig = "SIGNIFICANT" if pooled["p_value_chi2"] < 0.05 else "not significant"
            logger.info(f"  p-value (χ², Edwards): {pooled['p_value_chi2']:.4f} ({sig})")
    else:
        logger.info("\n[INFO] McNemar skipped: requires evaluating 'cnn' and 'hybrid' together.")

    logger.info(f"\n{'='*70}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*70}")
    md_path = write_outputs(per_model_seed, aggregated, mcnemar_results, sysinfo)

    logger.info(f"\n{'='*70}")
    logger.info("FULL RE-EVALUATION FINISHED")
    logger.info(f"{'='*70}")
    logger.info(f"  Summary report: {md_path}")
    logger.info(f"  Arrays (.npy/.npz) in: {ARRAYS_DIR}")
    logger.info(f"  All outputs in: {REEVAL_DIR}")


if __name__ == "__main__":
    main()
