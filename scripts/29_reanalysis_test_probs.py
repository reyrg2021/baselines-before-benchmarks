#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 29: Statistical reanalysis of the saved test probabilities.

Reanalyses, without any new inference or training, the per-cycle softmax
probabilities saved by scripts 18, 21 and 22 for the six configurations
(5 seeds each, 2733 cycles of the official post-QC test set):
  - trivial baselines (always-Normal, majority abnormal class, random, device prior);
  - paired TOST equivalence test per seed (pre-specified margin delta = 0.02);
  - patient-clustered bootstrap confidence intervals;
  - stratification by recording equipment and patient-level aggregation;
  - operating points on the binary ROC (thresholds fixed on test, so they
     characterise the curve and are not deployment points).

Integrity: before computing anything, each {run}_probs.npy is aligned row by
row with test_cycles.csv (y_true must match cycle_class) and the per-seed
official ICBHI score and binary AUC are recomputed and checked against
master_results.json. If either check fails the script aborts, because no
downstream figure would be valid. --skip_verify exists only for smoke tests.

Equivalence: a paired t-test with p > 0.05 does not show equivalence. TOST
(Schuirmann 1987; Lakens 2017) tests H0: |diff| >= delta with two one-sided
tests; equivalence holds only if both reject at level alpha, i.e. the
(1 - 2*alpha) CI of the difference lies inside [-delta, +delta]. The margin
was fixed before running this script and is applied to both ICBHI and AUC.

Bootstrap: the unit of dependence is the patient (many cycles share device,
chest location and pathophysiology), so patients are resampled with
replacement (B = 2000) and the same replicate is applied to all
(model, seed) runs, making model differences paired.

Threshold rule with tau on P(abnormal) = 1 - P(Normal):
    y_pred(tau) = Normal              if P(abnormal) < tau
                = 1 + argmax P(1:4)   otherwise
which keeps the official (exact-subclass) ICBHI score well defined for every tau.

Inputs:  results/metrics/full_reeval/arrays/, data/splits/strategy1_official/,
         results/master_results.json, data/raw/ICBHI_Challenge_diagnosis.txt
Outputs: results/metrics/reanalysis/ (reanalysis_results.json, CSV tables per
         section, reanalysis_summary.md), reanalysis_log.txt
Usage:
    python 29_reanalysis_test_probs.py [--n_boot 2000] [--delta 0.02]
                                       [--alpha 0.05] [--boot_seed 2026] [--skip_verify]
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


# Paths

BASE_DIR_FIXED = Path(__file__).resolve().parent.parent

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent

if BASE_DIR_FIXED.exists():
    BASE_DIR = BASE_DIR_FIXED
elif (SCRIPTS_DIR.parent / "models").exists():
    BASE_DIR = SCRIPTS_DIR.parent
else:
    BASE_DIR = SCRIPTS_DIR

METRICS_DIR = BASE_DIR / "results" / "metrics"
ARRAYS_DIR = METRICS_DIR / "full_reeval" / "arrays"
SPLIT_CSV = BASE_DIR / "data" / "splits" / "strategy1_official" / "test_cycles.csv"
TRAIN_CSV = BASE_DIR / "data" / "splits" / "strategy1_official" / "train_cycles.csv"
MASTER_JSON = BASE_DIR / "results" / "master_results.json"   # written by script 24
DIAGNOSIS_TXT = BASE_DIR / "data" / "raw" / "ICBHI_Challenge_diagnosis.txt"
OUT_DIR = METRICS_DIR / "reanalysis"

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_MAP = {"Normal": 0, "Crackle": 1, "Wheeze": 2, "Both": 3}

# Same seed set for all configurations
SEEDS = [42, 123, 7, 99, 314]

# Configuration -> array prefix and master_results key. Order defines table
# order; the CNN goes first because it is the reference for all paired comparisons.
MODELS = {
    "CNN":        {"display": "CNN-RespiratoryNet (FP32)",
                   "prefix": "cnn",
                   "master_key": "CNN_FP32"},
    "CNN_INT8":   {"display": "CNN-RespiratoryNet (INT8, FP32 classifier)",
                   "prefix": "cnn_int8",
                   "master_key": "CNN_INT8", "optional": True},
    "MNv3_IN":    {"display": "MobileNetV3-Small (ImageNet)",
                   "prefix": "tl_mobilenet_v3_small_pretrained_official",
                   "master_key": "MobileNetV3_ImageNet"},
    "MNv3_SC":    {"display": "MobileNetV3-Small (scratch)",
                   "prefix": "tl_mobilenet_v3_small_official",
                   "master_key": "MobileNetV3_scratch"},
    "PANNs":      {"display": "PANNs CNN6 (AudioSet)",
                   "prefix": "panns_cnn6_official",
                   "master_key": "PANNs_CNN6_AudioSet"},
    "PANNs_tune": {"display": "PANNs CNN6 (AudioSet, tune)",
                   "prefix": "panns_cnn6_tune_official",
                   "master_key": "PANNs_CNN6_tune"},
    "CfC":        {"display": "CfC-Hybrid (mmRNN)",
                   "prefix": "hybrid",
                   "master_key": "CfC_Hybrid"},
    # Recurrent baselines from script 31: not in master_results.json (the
    # integrity check skips them with a warning); included if their arrays exist.
    "LSTM":       {"display": "LSTM-Hybrid (h=103)",
                   "prefix": "lstm_official",
                   "master_key": "LSTM_Hybrid", "optional": True},
    "GRU":        {"display": "GRU-Hybrid (h=134)",
                   "prefix": "gru_official",
                   "master_key": "GRU_Hybrid", "optional": True},
    "CfC_pure":   {"display": "CfC without mixed memory",
                   "prefix": "cfcpure_official",
                   "master_key": "CfC_pure", "optional": True},
}
REFERENCE = "CNN"

# Operating points characterised on the binary ROC (section 6)
OPERATING_POINTS = [
    ("argmax",   "4-class argmax (reported operating point)"),
    ("se_eq_sp", "Se = Sp (equal error)"),
    ("youden",   "Youden maximum (Se + Sp - 1)"),
    ("sp_0.70",  "Sp >= 0.70 (30 % referral budget)"),
    ("sp_0.80",  "Sp >= 0.80 (20 % referral budget)"),
    ("se_0.80",  "Se >= 0.80 (misses <= 20 % of abnormal cycles)"),
]


# Logging

logger = logging.getLogger(__name__)


def setup_logging():
    """Log to stdout and to reanalysis_log.txt in BASE_DIR."""
    log_file = BASE_DIR / "reanalysis_log.txt"
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


# Metrics (same definitions as script 18)

def confusion_from_preds(y_true, y_pred, n_classes=4):
    """Confusion matrix, rows = true, columns = predicted (vectorised)."""
    idx = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def icbhi_official_from_confusion(cm):
    """
    Official ICBHI 2017 score (exact-subclass):
        Sp = CM[0,0] / sum(CM[0,:])
        Se = (CM[1,1] + CM[2,2] + CM[3,3]) / sum(CM[1:,:])
        Score = (Se + Sp) / 2
    Returns (score, se, sp).
    """
    cm = np.asarray(cm, dtype=float)
    normal_total = cm[0, :].sum()
    abn_total = cm[1:, :].sum()
    sp = cm[0, 0] / normal_total if normal_total > 0 else 0.0
    se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / abn_total if abn_total > 0 else 0.0
    return (se + sp) / 2.0, float(se), float(sp)


def auc_from_scores(y_bin, scores):
    """
    Rank-based ROC AUC (Mann-Whitney U), exactly equivalent to sklearn's AUC and
    several times faster inside the bootstrap. Returns NaN if a class is missing.
    """
    pos = y_bin == 1
    n_pos = int(pos.sum())
    n_neg = int(len(y_bin) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)          # ties get the average rank
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_se_sp(y_true, y_pred):
    """Binary Se and Sp, Normal (0) vs Abnormal (1,2,3), from the argmax prediction."""
    yt = y_true != 0
    yp = y_pred != 0
    se = float((yt & yp).sum() / yt.sum()) if yt.sum() > 0 else 0.0
    sp = float((~yt & ~yp).sum() / (~yt).sum()) if (~yt).sum() > 0 else 0.0
    return se, sp


def all_metrics(y_true, y_pred, p_abn):
    """All metrics of one run on a subset of cycles."""
    cm = confusion_from_preds(y_true, y_pred)
    score, se, sp = icbhi_official_from_confusion(cm)
    bse, bsp = binary_se_sp(y_true, y_pred)
    auc = auc_from_scores((y_true != 0).astype(int), p_abn)
    return {"icbhi": score, "se_off": se, "sp_off": sp,
            "bin_se": bse, "bin_sp": bsp, "auc": auc}


def preds_at_threshold(probs, p_abn, tau):
    """
    Decision rule with threshold tau on P(abnormal) = 1 - P(Normal): Normal if
    P(abnormal) < tau, otherwise the most probable abnormal subclass. Keeps the
    official (exact-subclass) ICBHI score well defined. p_abn is passed
    explicitly so that the same float32-derived score as the ROC is used.
    """
    sub = 1 + np.argmax(probs[:, 1:], axis=1)
    return np.where(p_abn >= tau, sub, 0)


def mean_sd(x):
    """Mean and sample standard deviation (ddof = 1), ignoring NaN."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


# Section 1: loading, alignment and integrity

def load_split():
    """Read test_cycles.csv and derive the columns used by the later sections."""
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Not found: {SPLIT_CSV}")
    df = pd.read_csv(SPLIT_CSV)
    needed = ["cycle_id", "patient_id", "cycle_class", "recording_equipment",
              "chest_location", "acquisition_mode", "filename_original"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in test_cycles.csv: {missing}")
    df["y_true"] = df["cycle_class"].map(CLASS_MAP)
    if df["y_true"].isna().any():
        raise ValueError("cycle_class has values outside {Normal,Crackle,Wheeze,Both}")
    df["y_true"] = df["y_true"].astype(int)
    logger.info(f"test_cycles.csv: {len(df)} cycles, "
                f"{df['patient_id'].nunique()} patients, "
                f"{df['filename_original'].nunique()} recordings")
    logger.info("Class distribution: " +
                ", ".join(f"{c}={int((df['y_true'] == i).sum())}"
                          for i, c in enumerate(CLASS_NAMES)))
    logger.info("Equipment: " + ", ".join(
        f"{k}={v}" for k, v in df["recording_equipment"].value_counts().items()))
    return df


def load_runs(df):
    """
    Load probs and y_true/y_pred of every run and verify row-by-row alignment
    with the CSV. Optional models with missing arrays are removed from MODELS.
    Returns dict[model][seed] = {probs, y_pred, p_abn}.
    """
    n = len(df)
    y_csv = df["y_true"].values
    runs = {}
    for key in list(MODELS.keys()):
        spec = MODELS[key]
        missing = [s for s in SEEDS if not (ARRAYS_DIR / f"{spec['prefix']}_seed{s}_probs.npy").exists()]
        if missing and spec.get("optional"):
            logger.warning(f"  {spec['display']}: missing arrays for seeds {missing}; skipped")
            del MODELS[key]
            continue
        runs[key] = {}
        for seed in SEEDS:
            stem = f"{spec['prefix']}_seed{seed}"
            p_path = ARRAYS_DIR / f"{stem}_probs.npy"
            l_path = ARRAYS_DIR / f"{stem}_labels_preds.npz"
            if not p_path.exists() or not l_path.exists():
                raise FileNotFoundError(f"Missing arrays for {stem} in {ARRAYS_DIR}")
            probs32 = np.load(p_path)                  # float32, as saved
            # P(abnormal) is computed in float32, as in script 18. Doing it in
            # float64 changes ties between scores and shifts the AUC by ~1e-4,
            # which breaks the exact match with master_results.json.
            p_abn = (np.float32(1.0) - probs32[:, 0]).astype(np.float64)
            probs = probs32.astype(np.float64)
            lp = np.load(l_path)
            y_true, y_pred = lp["y_true"].astype(int), lp["y_pred"].astype(int)

            # Alignment with the CSV
            if probs.shape != (n, 4):
                raise ValueError(f"{stem}: probs {probs.shape} != ({n}, 4)")
            if len(y_true) != n or not np.array_equal(y_true, y_csv):
                n_diff = int((y_true != y_csv).sum()) if len(y_true) == n else -1
                raise ValueError(f"{stem}: y_true does not match cycle_class in the CSV "
                                 f"({n_diff} differing rows). Loader order is broken.")
            # Internal consistency: y_pred must equal argmax(probs)
            if not np.array_equal(y_pred, np.argmax(probs, axis=1)):
                n_diff = int((y_pred != np.argmax(probs, axis=1)).sum())
                logger.warning(f"{stem}: y_pred differs from argmax(probs) in {n_diff} "
                               f"rows; using argmax(probs) for consistency with tau")
                y_pred = np.argmax(probs, axis=1)
            runs[key][seed] = {"probs": probs, "y_pred": y_pred, "p_abn": p_abn}
        logger.info(f"  {spec['display']}: {len(SEEDS)} seeds aligned")
    return runs


def verify_against_master(df, runs, tol_icbhi=1e-6, tol_auc=1e-5):
    """
    Recompute the official ICBHI score and AUC per seed and check them against
    master_results.json; abort on mismatch.

    Tolerances differ by metric:
      - ICBHI comes from argmax counts and must match exactly (1e-6).
      - AUC comes from continuous scores. For MobileNet/PANNs the stored value
        comes from the training *_metrics.json (first test pass) and the arrays
        from a second --eval pass; two non-deterministic cuDNN passes differ by
        ~1e-6 without changing any argmax. 1e-5 is allowed, three orders below
        the third reported decimal; anything larger means different arrays.
    """
    if not MASTER_JSON.exists():
        raise FileNotFoundError(f"Not found: {MASTER_JSON}")
    master = json.load(open(MASTER_JSON, encoding="utf-8"))["accuracy"]
    y_true = df["y_true"].values
    n_checked = 0
    for key, spec in MODELS.items():
        mk = spec["master_key"]
        if mk not in master:
            logger.warning(f"  {mk} is not in master_results.json; check skipped")
            continue
        m_seeds = master[mk]["seeds"]
        m_icbhi = master[mk]["icbhi_official"]["values"]
        m_auc = master[mk]["auc_binary"]["values"]
        for s, mi, ma in zip(m_seeds, m_icbhi, m_auc):
            r = runs[key][int(s)]
            met = all_metrics(y_true, r["y_pred"], r["p_abn"])
            d_i, d_a = abs(met["icbhi"] - mi), abs(met["auc"] - ma)
            if d_a > 0:
                logger.info(f"    {mk} seed {s}: ΔAUC = {d_a:.2e} (ΔICBHI = {d_i:.1e})")
            if d_i > tol_icbhi or d_a > tol_auc:
                raise AssertionError(
                    f"{mk} seed {s}: ICBHI {met['icbhi']:.6f} vs master {mi:.6f} "
                    f"(Δ={d_i:.2e}); AUC {met['auc']:.6f} vs master {ma:.6f} "
                    f"(Δ={d_a:.2e}). Arrays and master_results do not correspond.")
            n_checked += 1
    logger.info(f"  Check against master_results.json: {n_checked} (model, seed) "
                f"match in ICBHI (tol {tol_icbhi:.0e}) and AUC (tol {tol_auc:.0e})")


# Section 2: trivial baselines

def trivial_baselines(df):
    """
    Degenerate classifiers on the same test set (2733 cycles):
      - always-Normal: Se = 0, Sp = 1 -> ICBHI 0.500, AUC 0.5 (no discrimination)
      - always-<most frequent abnormal class>: Sp = 0, Se = n_c / n_abn
      - stratified random (analytic expectation): each class predicted with its
        test prior; E[Se] = sum_c pi_c * n_c / n_abn, E[Sp] = pi_Normal, AUC = 0.5
      - uniform random (expectation): Se = 1/4, Sp = 1/4 -> 0.250
      - device prior (see device_prior_baseline): measures the "equipment -> class"
        shortcut available in the official split.
    always-Normal shows that ICBHI 0.500 requires no discrimination at all.
    """
    y = df["y_true"].values
    n = len(y)
    counts = np.bincount(y, minlength=4).astype(float)
    n_abn = counts[1:].sum()
    prior = counts / n
    rows = []

    # always-Normal
    rows.append({"baseline": "always-Normal", "se_off": 0.0, "sp_off": 1.0,
                 "icbhi": 0.5, "bin_se": 0.0, "bin_sp": 1.0, "auc": 0.5})
    # always-<majority abnormal class>
    c_max = 1 + int(np.argmax(counts[1:]))
    se = counts[c_max] / n_abn
    rows.append({"baseline": f"always-{CLASS_NAMES[c_max]}", "se_off": float(se),
                 "sp_off": 0.0, "icbhi": float(se / 2), "bin_se": 1.0,
                 "bin_sp": 0.0, "auc": 0.5})
    # stratified random (expectation)
    se = float(np.sum(prior[1:] * counts[1:]) / n_abn)
    sp = float(prior[0])
    rows.append({"baseline": "random-stratified (E)", "se_off": se, "sp_off": sp,
                 "icbhi": (se + sp) / 2, "bin_se": float(1 - prior[0]),
                 "bin_sp": float(prior[0]), "auc": 0.5})
    # uniform random (expectation)
    rows.append({"baseline": "random-uniform (E)", "se_off": 0.25, "sp_off": 0.25,
                 "icbhi": 0.25, "bin_se": 0.75, "bin_sp": 0.25, "auc": 0.5})

    # device prior (estimated on TRAIN, applied to TEST)
    dev = device_prior_baseline(df)
    if dev is not None:
        rows.append(dev)

    for r in rows:
        logger.info(f"  {r['baseline']:<26} ICBHI {r['icbhi']:.3f} "
                    f"(Se {r['se_off']:.3f} / Sp {r['sp_off']:.3f})  AUC {r['auc']:.3f}")
    return rows


def device_prior_baseline(df):
    """
    Classifier that only looks at the recording equipment. In the official split
    equipment correlates with class (AKG C417L: 66 % abnormal in train; Meditron:
    29 %), so a model can learn the microphone signature as a shortcut to the
    label. This baseline quantifies how far that shortcut goes on test:
      - prediction = TRAIN majority class for that equipment (official ICBHI)
      - AUC score  = P(abnormal | equipment) estimated on TRAIN
    Equipment absent from train gets the global train prior. If this AUC
    approaches a model's AUC, much of that model's discrimination may be
    device recognition.
    """
    if not TRAIN_CSV.exists():
        logger.warning(f"  {TRAIN_CSV} not found; device-prior baseline skipped")
        return None
    tr = pd.read_csv(TRAIN_CSV)
    tr["y"] = tr["cycle_class"].map(CLASS_MAP)
    g_prior = float((tr["y"] != 0).mean())
    g_major = int(tr["y"].value_counts().idxmax())
    p_abn_dev, major_dev = {}, {}
    for eq, sub in tr.groupby("recording_equipment"):
        p_abn_dev[eq] = float((sub["y"] != 0).mean())
        major_dev[eq] = int(sub["y"].value_counts().idxmax())
        logger.info(f"    train prior {eq:<10} P(abnormal)={p_abn_dev[eq]:.3f}  "
                    f"majority={CLASS_NAMES[major_dev[eq]]}  n={len(sub)}")
    eq_test = df["recording_equipment"].values
    y_true = df["y_true"].values
    y_pred = np.array([major_dev.get(e, g_major) for e in eq_test])
    score = np.array([p_abn_dev.get(e, g_prior) for e in eq_test])
    met = all_metrics(y_true, y_pred, score)
    return {"baseline": "device-prior (train)", "se_off": met["se_off"],
            "sp_off": met["sp_off"], "icbhi": met["icbhi"], "bin_se": met["bin_se"],
            "bin_sp": met["bin_sp"], "auc": met["auc"]}


# Section 3: paired TOST per seed

def per_seed_metric_table(df, runs):
    """dict[model][metric] = array (n_seeds,) in SEEDS order."""
    y_true = df["y_true"].values
    table = {}
    for key in MODELS:
        vals = {m: [] for m in ["icbhi", "se_off", "sp_off", "bin_se", "bin_sp", "auc"]}
        for seed in SEEDS:
            met = all_metrics(y_true, runs[key][seed]["y_pred"], runs[key][seed]["p_abn"])
            for m in vals:
                vals[m].append(met[m])
        table[key] = {m: np.array(v) for m, v in vals.items()}
    return table


def tost_paired(a, b, delta, alpha):
    """
    TOST on paired differences d = a - b (same seeds).
      p_lower : H0 d <= -delta  vs H1 d > -delta
      p_upper : H0 d >= +delta  vs H1 d < +delta
      p_tost  = max(p_lower, p_upper); equivalent if p_tost < alpha.
    Equivalent to the (1 - 2 alpha) CI lying inside [-delta, delta].
    Also returns the two-sided paired t-test and Wilcoxon p-values.
    """
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha, n - 1)
    ci_lo, ci_hi = mean - t_crit * se, mean + t_crit * se
    p_lower = stats.ttest_1samp(d, -delta, alternative="greater").pvalue
    p_upper = stats.ttest_1samp(d, +delta, alternative="less").pvalue
    p_tost = max(p_lower, p_upper)
    p_two = stats.ttest_rel(a, b).pvalue
    try:
        p_wil = stats.wilcoxon(a, b).pvalue
    except ValueError:      # all differences are zero
        p_wil = float("nan")
    return {"diff_mean": float(mean), "diff_sd": float(sd), "n": n,
            "ci90_lo": float(ci_lo), "ci90_hi": float(ci_hi),
            "p_ttest_two_sided": float(p_two), "p_wilcoxon": float(p_wil),
            "p_tost_lower": float(p_lower), "p_tost_upper": float(p_upper),
            "p_tost": float(p_tost), "equivalent": bool(p_tost < alpha),
            "ci_within_margin": bool(ci_lo > -delta and ci_hi < delta)}


def run_tost(table, delta, alpha):
    """TOST of the reference model (CNN) against every other configuration, ICBHI and AUC."""
    rows = []
    for key in MODELS:
        if key == REFERENCE:
            continue
        for metric in ["icbhi", "auc"]:
            res = tost_paired(table[REFERENCE][metric], table[key][metric], delta, alpha)
            res.update({"comparison": f"{REFERENCE} - {key}", "metric": metric,
                        "delta": delta, "alpha": alpha})
            rows.append(res)
            logger.info(
                f"  {REFERENCE} - {key:<10} {metric:<6} Δ={res['diff_mean']:+.4f} "
                f"CI90 [{res['ci90_lo']:+.4f}, {res['ci90_hi']:+.4f}]  "
                f"t p={res['p_ttest_two_sided']:.3f}  Wilcoxon p={res['p_wilcoxon']:.3f}  "
                f"TOST p={res['p_tost']:.3f} -> "
                f"{'EQUIVALENT' if res['equivalent'] else 'not equivalent'} (±{delta})")
    return rows


# Section 4: patient-clustered bootstrap

def patient_bootstrap(df, runs, n_boot, alpha, delta, rng):
    """
    Resample patients with replacement; each replicate rebuilds the test set
    from all cycles of the drawn patients and is applied to every run.
    Metric per (model, replicate) = mean over seeds. Differences against the
    reference are paired on the same replicate.
    """
    y_true = df["y_true"].values
    patients = df["patient_id"].values
    uniq = np.unique(patients)
    n_pat = len(uniq)
    pat_idx = {p: np.where(patients == p)[0] for p in uniq}
    logger.info(f"  {n_pat} test patients; B = {n_boot} replicates")

    metrics = ["icbhi", "se_off", "sp_off", "auc"]
    boot = {key: {m: np.empty(n_boot) for m in metrics} for key in MODELS}

    for b in range(n_boot):
        draw = rng.choice(uniq, size=n_pat, replace=True)
        idx = np.concatenate([pat_idx[p] for p in draw])
        yt = y_true[idx]
        for key in MODELS:
            acc = {m: 0.0 for m in metrics}
            for seed in SEEDS:
                r = runs[key][seed]
                met = all_metrics(yt, r["y_pred"][idx], r["p_abn"][idx])
                for m in metrics:
                    acc[m] += met[m]
            for m in metrics:
                boot[key][m][b] = acc[m] / len(SEEDS)
        if (b + 1) % 500 == 0:
            logger.info(f"    replicate {b + 1}/{n_boot}")

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)          # 95 % CI
    lo90, hi90 = 100 * alpha, 100 * (1 - alpha)               # 90 % CI (TOST)

    rows_model, rows_diff = [], []
    for key in MODELS:
        row = {"model": key, "n_patients": n_pat, "n_boot": n_boot}
        for m in metrics:
            v = boot[key][m]
            row[f"{m}_mean"] = float(np.nanmean(v))
            row[f"{m}_ci_lo"] = float(np.nanpercentile(v, lo))
            row[f"{m}_ci_hi"] = float(np.nanpercentile(v, hi))
        rows_model.append(row)
        logger.info(f"  {key:<10} ICBHI {row['icbhi_mean']:.3f} "
                    f"[{row['icbhi_ci_lo']:.3f}, {row['icbhi_ci_hi']:.3f}]  "
                    f"AUC {row['auc_mean']:.3f} "
                    f"[{row['auc_ci_lo']:.3f}, {row['auc_ci_hi']:.3f}]")
        if key == REFERENCE:
            continue
        drow = {"comparison": f"{REFERENCE} - {key}", "n_boot": n_boot, "delta": delta}
        for m in ["icbhi", "auc"]:
            d = boot[REFERENCE][m] - boot[key][m]
            drow[f"{m}_diff_mean"] = float(np.nanmean(d))
            drow[f"{m}_ci95_lo"] = float(np.nanpercentile(d, lo))
            drow[f"{m}_ci95_hi"] = float(np.nanpercentile(d, hi))
            drow[f"{m}_ci90_lo"] = float(np.nanpercentile(d, lo90))
            drow[f"{m}_ci90_hi"] = float(np.nanpercentile(d, hi90))
            drow[f"{m}_ci90_within_margin"] = bool(
                drow[f"{m}_ci90_lo"] > -delta and drow[f"{m}_ci90_hi"] < delta)
            # Two-sided bootstrap p: 2 * min(P(d<=0), P(d>=0))
            drow[f"{m}_p_boot"] = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
        rows_diff.append(drow)
        logger.info(
            f"    Δ vs {key:<10} ICBHI {drow['icbhi_diff_mean']:+.4f} "
            f"CI90 [{drow['icbhi_ci90_lo']:+.4f}, {drow['icbhi_ci90_hi']:+.4f}] "
            f"{'inside' if drow['icbhi_ci90_within_margin'] else 'OUTSIDE'} (margin ±{delta})  |  "
            f"AUC {drow['auc_diff_mean']:+.4f} "
            f"CI90 [{drow['auc_ci90_lo']:+.4f}, {drow['auc_ci90_hi']:+.4f}] "
            f"{'inside' if drow['auc_ci90_within_margin'] else 'OUTSIDE'}")
    return rows_model, rows_diff


# Section 5: stratification by equipment and patient level

def by_equipment(df, runs):
    """
    Metrics per recording equipment, mean ± sd over seeds. If a stratum lacks
    Normal or abnormal cycles, AUC and the affected Se/Sp are NaN. Relevant
    because the QC exclusion is skewed toward the AKG C417L.
    """
    y_true = df["y_true"].values
    rows = []
    for equip, sub in df.groupby("recording_equipment"):
        idx = sub.index.values
        yt = y_true[idx]
        n_norm, n_abn = int((yt == 0).sum()), int((yt != 0).sum())
        for key in MODELS:
            per_seed = {m: [] for m in ["icbhi", "se_off", "sp_off", "auc"]}
            for seed in SEEDS:
                r = runs[key][seed]
                met = all_metrics(yt, r["y_pred"][idx], r["p_abn"][idx])
                for m in per_seed:
                    per_seed[m].append(met[m])
            row = {"equipment": equip, "model": key, "n_cycles": len(idx),
                   "n_patients": int(sub["patient_id"].nunique()),
                   "n_normal": n_norm, "n_abnormal": n_abn}
            for m, v in per_seed.items():
                mu, sd = mean_sd(v)
                row[f"{m}_mean"], row[f"{m}_sd"] = mu, sd
            rows.append(row)
        cnn = [r for r in rows if r["equipment"] == equip and r["model"] == REFERENCE][0]
        logger.info(f"  {equip:<10} n={len(idx):>5} pat={cnn['n_patients']:>3} "
                    f"N/A={n_norm}/{n_abn}  CNN ICBHI {cnn['icbhi_mean']:.3f}±{cnn['icbhi_sd']:.3f}  "
                    f"AUC {cnn['auc_mean']:.3f}±{cnn['auc_sd']:.3f}")
    return rows


def load_diagnosis():
    """patient_id -> diagnosis (ICBHI_Challenge_diagnosis.txt), or None if missing."""
    cands = [DIAGNOSIS_TXT, BASE_DIR / "data" / "ICBHI_Challenge_diagnosis.txt",
             BASE_DIR / "ICBHI_Challenge_diagnosis.txt"]
    cands += list(BASE_DIR.rglob("ICBHI_Challenge_diagnosis.txt"))
    for p in cands:
        if p.exists():
            d = pd.read_csv(p, sep="\t", header=None, names=["patient_id", "diagnosis"])
            logger.info(f"  ICBHI diagnoses: {p}")
            return dict(zip(d["patient_id"].astype(int), d["diagnosis"].astype(str)))
    logger.warning("  ICBHI_Challenge_diagnosis.txt not found; the diagnosis-based "
                   "patient definition is skipped (cycle-based only)")
    return None


def patient_level(df, runs):
    """
    Patient-level aggregation with two definitions of a positive patient:
      (a) cycle-based: > 50 % of the patient's test cycles are abnormal;
      (b) ICBHI diagnosis-based: diagnosis != Healthy.
    Patient score = mean P(abnormal) over the patient's cycles. Reports per-seed
    patient-level AUC (mean ± sd) and, with a 0.5 threshold on the fraction of
    cycles predicted abnormal, patient-level Se/Sp. The number of patients is
    small, so these results are descriptive, not inferential.
    """
    y_true = df["y_true"].values
    diag = load_diagnosis()
    groups = df.groupby("patient_id").indices          # patient -> row indices
    pids = sorted(groups.keys())
    frac_abn_true = np.array([(y_true[groups[p]] != 0).mean() for p in pids])
    label_cycles = (frac_abn_true > 0.5).astype(int)
    label_diag = None
    if diag is not None:
        label_diag = np.array([0 if diag.get(int(p), "NA") == "Healthy" else 1 for p in pids])
        n_unknown = sum(1 for p in pids if int(p) not in diag)
        if n_unknown:
            logger.warning(f"  {n_unknown} patients without a diagnosis in the ICBHI file")

    logger.info(f"  {len(pids)} patients; cycle-based positives (>50 %): "
                f"{int(label_cycles.sum())}; "
                + (f"diagnosis-based positives (!= Healthy): {int(label_diag.sum())}"
                   if label_diag is not None else "diagnosis not available"))

    rows = []
    for key in MODELS:
        aucs_c, aucs_d, ses, sps = [], [], [], []
        for seed in SEEDS:
            r = runs[key][seed]
            score = np.array([r["p_abn"][groups[p]].mean() for p in pids])
            frac_pred = np.array([(r["y_pred"][groups[p]] != 0).mean() for p in pids])
            aucs_c.append(auc_from_scores(label_cycles, score))
            if label_diag is not None:
                aucs_d.append(auc_from_scores(label_diag, score))
            pred = (frac_pred > 0.5).astype(int)
            pos, neg = label_cycles == 1, label_cycles == 0
            ses.append(float((pred[pos] == 1).mean()) if pos.any() else np.nan)
            sps.append(float((pred[neg] == 0).mean()) if neg.any() else np.nan)
        row = {"model": key, "n_patients": len(pids),
               "n_pos_cycles_def": int(label_cycles.sum()),
               "n_pos_diag_def": int(label_diag.sum()) if label_diag is not None else None}
        for name, v in [("auc_cycles_def", aucs_c), ("auc_diag_def", aucs_d),
                        ("se_patient", ses), ("sp_patient", sps)]:
            mu, sd = mean_sd(v) if len(v) else (np.nan, np.nan)
            row[f"{name}_mean"], row[f"{name}_sd"] = mu, sd
        rows.append(row)
        logger.info(f"  {key:<10} patient AUC (cycles) {row['auc_cycles_def_mean']:.3f}"
                    f"±{row['auc_cycles_def_sd']:.3f}  "
                    f"patient AUC (diag) {row['auc_diag_def_mean']:.3f}±{row['auc_diag_def_sd']:.3f}  "
                    f"patient Se/Sp {row['se_patient_mean']:.2f}/{row['sp_patient_mean']:.2f}")
    per_patient = pd.DataFrame({"patient_id": pids, "n_cycles": [len(groups[p]) for p in pids],
                                "frac_abnormal_true": frac_abn_true,
                                "label_cycles_def": label_cycles,
                                "diagnosis": [diag.get(int(p), "NA") if diag else "NA" for p in pids]})
    return rows, per_patient


# Section 6: operating points on the binary ROC

def find_threshold(y_bin, p_abn, mode):
    """
    Return tau for the requested criterion on the ROC of the given subset.
    Candidate thresholds are the scores themselves plus 1.0 (all Normal).
    """
    taus = np.unique(np.concatenate([p_abn, [1.0 + 1e-9]]))
    pos, neg = y_bin == 1, y_bin == 0
    # Vectorised Se(tau) and Sp(tau): positive prediction if p >= tau
    n_pos, n_neg = pos.sum(), neg.sum()
    pos_sorted = np.sort(p_abn[pos]); neg_sorted = np.sort(p_abn[neg])
    se = 1 - np.searchsorted(pos_sorted, taus, side="left") / n_pos
    sp = np.searchsorted(neg_sorted, taus, side="left") / n_neg
    if mode == "se_eq_sp":
        i = int(np.argmin(np.abs(se - sp)))
    elif mode == "youden":
        i = int(np.argmax(se + sp - 1))
    elif mode.startswith("sp_"):
        target = float(mode.split("_")[1])
        ok = np.where(sp >= target)[0]
        i = int(ok[np.argmax(se[ok])]) if len(ok) else int(np.argmax(sp))
    elif mode.startswith("se_"):
        target = float(mode.split("_")[1])
        ok = np.where(se >= target)[0]
        i = int(ok[np.argmax(sp[ok])]) if len(ok) else int(np.argmax(se))
    else:
        raise ValueError(mode)
    return float(taus[i])


def operating_points(df, runs):
    """
    For each model and operating point: tau (mean ± sd over seeds), binary Se/Sp
    and the resulting official ICBHI score. 'argmax' reproduces the reported
    operating point as a control. Thresholds are fixed on TEST, so they
    characterise the curve and are not a deployment point.
    """
    y_true = df["y_true"].values
    y_bin = (y_true != 0).astype(int)
    rows = []
    for key in MODELS:
        for mode, desc in OPERATING_POINTS:
            per = {m: [] for m in ["tau", "icbhi", "se_off", "sp_off", "bin_se", "bin_sp"]}
            for seed in SEEDS:
                r = runs[key][seed]
                if mode == "argmax":
                    tau = np.nan
                    y_pred = r["y_pred"]
                else:
                    tau = find_threshold(y_bin, r["p_abn"], mode)
                    y_pred = preds_at_threshold(r["probs"], r["p_abn"], tau)
                met = all_metrics(y_true, y_pred, r["p_abn"])
                per["tau"].append(tau)
                for m in ["icbhi", "se_off", "sp_off", "bin_se", "bin_sp"]:
                    per[m].append(met[m])
            row = {"model": key, "operating_point": mode, "description": desc}
            for m, v in per.items():
                mu, sd = mean_sd(v)
                row[f"{m}_mean"], row[f"{m}_sd"] = mu, sd
            rows.append(row)
            if key == REFERENCE:
                logger.info(f"  CNN {mode:<9} tau={row['tau_mean']:.3f}  "
                            f"bin Se/Sp {row['bin_se_mean']:.3f}/{row['bin_sp_mean']:.3f}  "
                            f"official ICBHI {row['icbhi_mean']:.3f}±{row['icbhi_sd']:.3f} "
                            f"(Se {row['se_off_mean']:.3f} / Sp {row['sp_off_mean']:.3f})")
    return rows


# Outputs

def fmt(mu, sd=None, nd=3):
    if mu is None or (isinstance(mu, float) and np.isnan(mu)):
        return "n/a"
    return f"{mu:.{nd}f}" if sd is None or np.isnan(sd) else f"{mu:.{nd}f} ± {sd:.{nd}f}"


def write_outputs(args, df, table, baselines, tost_rows, boot_rows, boot_diff,
                  equip_rows, pat_rows, per_patient, op_rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    disp = {k: v["display"] for k, v in MODELS.items()}

    # CSVs
    pd.DataFrame(baselines).to_csv(OUT_DIR / "reanalysis_trivial_baselines.csv", index=False)
    pd.DataFrame(tost_rows).to_csv(OUT_DIR / "reanalysis_tost.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "reanalysis_bootstrap.csv", index=False)
    pd.DataFrame(boot_diff).to_csv(OUT_DIR / "reanalysis_bootstrap_diff.csv", index=False)
    pd.DataFrame(equip_rows).to_csv(OUT_DIR / "reanalysis_by_equipment.csv", index=False)
    pd.DataFrame(pat_rows).to_csv(OUT_DIR / "reanalysis_patient_level.csv", index=False)
    per_patient.to_csv(OUT_DIR / "reanalysis_patient_labels.csv", index=False)
    pd.DataFrame(op_rows).to_csv(OUT_DIR / "reanalysis_operating_points.csv", index=False)

    # JSON
    per_seed = {k: {m: v.tolist() for m, v in table[k].items()} for k in table}
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"seeds": SEEDS, "delta_tost": args.delta, "alpha": args.alpha,
                   "n_boot": args.n_boot, "boot_seed": args.boot_seed,
                   "delta_preregistered": "2026-09-06, fixed before running this script",
                   "n_test_cycles": int(len(df)),
                   "n_test_patients": int(df["patient_id"].nunique())},
        "models": disp,
        "per_seed_metrics": per_seed,
        "sec2_trivial_baselines": baselines,
        "sec3_tost": tost_rows,
        "sec4_bootstrap_models": boot_rows,
        "sec4_bootstrap_diffs": boot_diff,
        "sec5_by_equipment": equip_rows,
        "sec5_patient_level": pat_rows,
        "sec6_operating_points": op_rows,
    }
    with open(OUT_DIR / "reanalysis_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Markdown
    L = []
    L.append("# Statistical reanalysis of the test probabilities (Script 29)\n")
    L.append(f"_{out['timestamp']} · {len(df)} cycles · {out['config']['n_test_patients']} "
             f"patients · TOST delta = {args.delta} (pre-specified) · B = {args.n_boot}_\n")

    L.append("\n## Section 2. Models and trivial baselines\n")
    L.append("| Configuration | Official ICBHI | Official Se | Official Sp | Binary AUC |")
    L.append("|---|---|---|---|---|")
    for k in MODELS:
        t = table[k]
        L.append(f"| {disp[k]} | {fmt(*mean_sd(t['icbhi']))} | {fmt(*mean_sd(t['se_off']))} "
                 f"| {fmt(*mean_sd(t['sp_off']))} | {fmt(*mean_sd(t['auc']))} |")
    for b in baselines:
        L.append(f"| _{b['baseline']}_ | {b['icbhi']:.3f} | {b['se_off']:.3f} "
                 f"| {b['sp_off']:.3f} | {b['auc']:.3f} |")

    L.append(f"\n## Section 3. Paired TOST per seed (n = {len(SEEDS)}, "
             f"delta = ±{args.delta}, alpha = {args.alpha})\n")
    L.append("| Comparison | Metric | Mean Δ | 90 % CI | Paired t p | Wilcoxon p | TOST p | Equivalent |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in tost_rows:
        L.append(f"| {r['comparison']} | {r['metric']} | {r['diff_mean']:+.4f} "
                 f"| [{r['ci90_lo']:+.4f}, {r['ci90_hi']:+.4f}] | {r['p_ttest_two_sided']:.3f} "
                 f"| {r['p_wilcoxon']:.3f} | {r['p_tost']:.3f} | {'yes' if r['equivalent'] else 'no'} |")

    L.append(f"\n## Section 4. Patient-clustered bootstrap (95 % CI, B = {args.n_boot})\n")
    L.append("| Model | ICBHI [95 % CI] | Official Se [CI] | Official Sp [CI] | AUC [95 % CI] |")
    L.append("|---|---|---|---|---|")
    for r in boot_rows:
        L.append(f"| {disp[r['model']]} | {r['icbhi_mean']:.3f} [{r['icbhi_ci_lo']:.3f}, {r['icbhi_ci_hi']:.3f}] "
                 f"| {r['se_off_mean']:.3f} [{r['se_off_ci_lo']:.3f}, {r['se_off_ci_hi']:.3f}] "
                 f"| {r['sp_off_mean']:.3f} [{r['sp_off_ci_lo']:.3f}, {r['sp_off_ci_hi']:.3f}] "
                 f"| {r['auc_mean']:.3f} [{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}] |")
    L.append("\n**Paired differences CNN − X (same patient replicate):**\n")
    L.append("| Comparison | Δ ICBHI [95 % CI] | 90 % CI within ±δ | boot p | Δ AUC [95 % CI] | 90 % CI within ±δ | boot p |")
    L.append("|---|---|---|---|---|---|---|")
    for r in boot_diff:
        L.append(f"| {r['comparison']} | {r['icbhi_diff_mean']:+.4f} [{r['icbhi_ci95_lo']:+.4f}, {r['icbhi_ci95_hi']:+.4f}] "
                 f"| {'yes' if r['icbhi_ci90_within_margin'] else 'no'} | {r['icbhi_p_boot']:.3f} "
                 f"| {r['auc_diff_mean']:+.4f} [{r['auc_ci95_lo']:+.4f}, {r['auc_ci95_hi']:+.4f}] "
                 f"| {'yes' if r['auc_ci90_within_margin'] else 'no'} | {r['auc_p_boot']:.3f} |")

    L.append("\n## Section 5a. Stratification by recording equipment (mean ± sd, 5 seeds)\n")
    L.append("| Equipment | n cycles | n patients | Normal/Abnormal | Model | ICBHI | Official Se | Official Sp | AUC |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in equip_rows:
        L.append(f"| {r['equipment']} | {r['n_cycles']} | {r['n_patients']} | {r['n_normal']}/{r['n_abnormal']} "
                 f"| {r['model']} | {fmt(r['icbhi_mean'], r['icbhi_sd'])} | {fmt(r['se_off_mean'], r['se_off_sd'])} "
                 f"| {fmt(r['sp_off_mean'], r['sp_off_sd'])} | {fmt(r['auc_mean'], r['auc_sd'])} |")

    L.append("\n## Section 5b. Patient level (descriptive; small n)\n")
    L.append("| Model | n patients | Patient AUC (cycles >50 % def.) | Patient AUC (diagnosis def.) | Patient Se | Patient Sp |")
    L.append("|---|---|---|---|---|---|")
    for r in pat_rows:
        L.append(f"| {disp[r['model']]} | {r['n_patients']} | {fmt(r['auc_cycles_def_mean'], r['auc_cycles_def_sd'])} "
                 f"| {fmt(r['auc_diag_def_mean'], r['auc_diag_def_sd'])} "
                 f"| {fmt(r['se_patient_mean'], r['se_patient_sd'], 2)} | {fmt(r['sp_patient_mean'], r['sp_patient_sd'], 2)} |")

    L.append("\n## Section 6. Operating points on the binary ROC (threshold fixed on TEST: "
             "characterisation, not deployment)\n")
    L.append("| Model | Point | tau | Bin Se | Bin Sp | Official ICBHI | Official Se | Official Sp |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in op_rows:
        L.append(f"| {r['model']} | {r['operating_point']} | {fmt(r['tau_mean'], r['tau_sd'])} "
                 f"| {fmt(r['bin_se_mean'], r['bin_se_sd'])} | {fmt(r['bin_sp_mean'], r['bin_sp_sd'])} "
                 f"| {fmt(r['icbhi_mean'], r['icbhi_sd'])} | {fmt(r['se_off_mean'], r['se_off_sd'])} "
                 f"| {fmt(r['sp_off_mean'], r['sp_off_sd'])} |")

    with open(OUT_DIR / "reanalysis_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    logger.info(f"  Outputs in {OUT_DIR}")


# Main

def parse_args():
    p = argparse.ArgumentParser(description="Script 29: statistical reanalysis of the test probabilities")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--delta", type=float, default=0.02,
                   help="Pre-specified TOST margin. Do not change it post hoc.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--boot_seed", type=int, default=2026)
    p.add_argument("--skip_verify", action="store_true",
                   help="Skip the check against master_results.json (smoke tests only)")
    return p.parse_args()


def main():
    args = parse_args()
    log_file = setup_logging()
    logger.info("=" * 78)
    logger.info("SCRIPT 29: statistical reanalysis of the test probabilities")
    logger.info("=" * 78)
    logger.info(f"BASE_DIR:   {BASE_DIR}")
    logger.info(f"ARRAYS_DIR: {ARRAYS_DIR}")
    logger.info(f"TOST delta = {args.delta} (pre-specified)  alpha = {args.alpha}  "
                f"B = {args.n_boot}  boot_seed = {args.boot_seed}")
    if args.delta != 0.02:
        logger.warning("delta != 0.02: the pre-specified margin is 0.02. Any other "
                       "value must be reported as a sensitivity analysis, not as a result.")

    logger.info("\n[1/6] Loading, alignment and integrity")
    df = load_split()
    runs = load_runs(df)
    if args.skip_verify:
        logger.warning("  --skip_verify: NOT checked against master_results.json. "
                       "Figures from this run are not valid for reporting.")
    else:
        verify_against_master(df, runs)
    table = per_seed_metric_table(df, runs)

    logger.info("\n[2/6] Trivial baselines")
    baselines = trivial_baselines(df)

    logger.info("\n[3/6] Paired TOST per seed")
    tost_rows = run_tost(table, args.delta, args.alpha)

    logger.info("\n[4/6] Patient-clustered bootstrap")
    rng = np.random.default_rng(args.boot_seed)
    boot_rows, boot_diff = patient_bootstrap(df, runs, args.n_boot, args.alpha, args.delta, rng)

    logger.info("\n[5/6] Stratification by equipment and patient level")
    equip_rows = by_equipment(df, runs)
    pat_rows, per_patient = patient_level(df, runs)

    logger.info("\n[6/6] Operating points on the binary ROC")
    op_rows = operating_points(df, runs)

    logger.info("\nWriting outputs")
    write_outputs(args, df, table, baselines, tost_rows, boot_rows, boot_diff,
                  equip_rows, pat_rows, per_patient, op_rows)
    logger.info(f"Log: {log_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
