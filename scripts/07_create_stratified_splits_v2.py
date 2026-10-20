#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 07: Patient-level stratified train/validation/test splits (three strategies).

Strategy 1 keeps the official ICBHI 2017 test partition for comparability with the
literature and carves a patient-disjoint validation set (one of five folds, ~20%) out of
the official train partition. The official split shares two patients (156, 218) between
train and test; this leakage is documented and kept. Strategy 2 is a custom
StratifiedGroupKFold (K=7) split with strict patient separation (5/1/1 folds for
train/val/test). Strategy 3 is a patient-grouped stratified 5-fold cross-validation.
Stratification uses each patient's dominant cycle class.

Inputs:  data/features/feature_metadata.csv (or data/segments/processed_cycles.csv),
         data/raw/ICBHI_challenge_train_test.txt
Outputs: data/splits/strategy1_official/{train,val,test}_cycles.csv
         data/splits/strategy2_custom/{train,val,test}_cycles.csv
         data/splits/strategy3_kfold/fold_{k}_{train,val,test}.csv, fold_assignments.csv
         data/splits/split_config.json, results/metrics/split_report.txt,
         results/figures/split_comparison_all_strategies.png, split_summary_table.png,
         split_creation_log.txt (in the working directory)
Usage:
    python 07_create_stratified_splits_v2.py
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('split_creation_log.txt', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SEGMENTS_DIR = BASE_DIR / "data" / "segments"
FEATURES_DIR = BASE_DIR / "data" / "features"
SPLITS_DIR = BASE_DIR / "data" / "splits"
RESULTS_DIR = BASE_DIR / "results" / "metrics"
FIGURES_DIR = BASE_DIR / "results" / "figures"

# Inputs
CYCLES_CSV = SEGMENTS_DIR / "processed_cycles.csv"
FEATURE_METADATA_CSV = FEATURES_DIR / "feature_metadata.csv"

# Official ICBHI split file: 920 lines of "<recording name>\t<train|test>"
ICBHI_OFFICIAL_SPLIT = BASE_DIR / "data" / "raw" / "ICBHI_challenge_train_test.txt"

# Output directories, one per strategy
STRATEGY1_DIR = SPLITS_DIR / "strategy1_official"
STRATEGY2_DIR = SPLITS_DIR / "strategy2_custom"
STRATEGY3_DIR = SPLITS_DIR / "strategy3_kfold"

# Reports
REPORT_FILE = RESULTS_DIR / "split_report.txt"
SPLIT_CONFIG_FILE = SPLITS_DIR / "split_config.json"
COMPARISON_CSV = SPLITS_DIR / "split_comparison.csv"


SPLIT_CONFIG = {
    # Seed shared by all strategies
    "random_seed": 42,

    "strategy1": {
        "name": "Official ICBHI Split + Validation",
        "description": (
            "Uses the official ICBHI 2017 challenge split. "
            "The test set is kept identical to the official one for comparability. "
            "The official train set is subdivided into train+validation with patient "
            "separation."
        ),
        # Recorded for reference; the subdivision below takes one of five folds (~20%)
        "val_from_train_ratio": 0.20,
    },

    "strategy2": {
        "name": "StratifiedGroupKFold (K=7)",
        "description": (
            "Custom split with strict patient separation. "
            "K=7 yields folds of ~14.3%, assigning 5 to train (~71%), "
            "1 to validation (~14%) and 1 to test (~14%). "
            "No data leakage."
        ),
        "n_splits_kfold": 7,
        "train_folds": 5,   # folds 2-6
        "val_fold": 1,
        "test_fold": 0,
    },

    "strategy3": {
        "name": "Stratified K-Fold CV with Patient Separation (K=5)",
        "description": (
            "Full cross-validation with K=5 folds. Each fold is used once "
            "as test; of the remaining 4 folds, 1 is used as "
            "validation and 3 as train. Produces 5 different splits that "
            "allow training 5 models and averaging metrics to obtain "
            "confidence intervals. Strict patient separation."
        ),
        "n_folds": 5,
        # Iteration i: test = fold i, val = fold (i+1) % K, train = the rest
    },
}


def get_patient_dominant_class(df):
    """
    Return {patient_id: most frequent cycle class}, used as the stratification label.

    The split unit is the patient, so one label per patient is needed. Ties are broken
    in favour of the rarer pathological class (Both > Wheeze > Crackle > Normal) so that
    minority classes are spread across splits.
    """
    class_priority = {"Both": 4, "Wheeze": 3, "Crackle": 2, "Normal": 1}
    patient_classes = {}

    for patient_id, group in df.groupby("patient_id"):
        class_counts = group["cycle_class"].value_counts()
        max_count = class_counts.max()
        tied_classes = class_counts[class_counts == max_count].index.tolist()

        if len(tied_classes) == 1:
            patient_classes[patient_id] = tied_classes[0]
        else:
            patient_classes[patient_id] = max(
                tied_classes, key=lambda c: class_priority.get(c, 0)
            )

    return patient_classes


def verify_no_patient_leakage(df, split_col="split"):
    """Return True if no patient appears in more than one split; log any overlap."""
    split_values = df[split_col].unique()
    patient_sets = {}
    for sv in split_values:
        patient_sets[sv] = set(df[df[split_col] == sv]["patient_id"].unique())

    is_clean = True
    checked_pairs = set()
    for s1 in split_values:
        for s2 in split_values:
            if s1 != s2 and (s2, s1) not in checked_pairs:
                overlap = patient_sets[s1] & patient_sets[s2]
                if overlap:
                    logger.warning(
                        f"  DATA LEAKAGE: {len(overlap)} patient(s) in "
                        f"{s1} ∩ {s2}: {sorted(overlap)}"
                    )
                    is_clean = False
                checked_pairs.add((s1, s2))

    return is_clean


def compute_split_stats(df, split_col="split"):
    """Per-split cycle/patient counts and class distribution, used by the report and figures."""
    stats = {}
    total = len(df)
    total_patients = df["patient_id"].nunique()

    for split_name in sorted(df[split_col].unique()):
        split_df = df[df[split_col] == split_name]
        n_cycles = len(split_df)
        n_patients = split_df["patient_id"].nunique()

        class_dist = {}
        for cls in ["Normal", "Crackle", "Wheeze", "Both"]:
            count = len(split_df[split_df["cycle_class"] == cls])
            class_dist[cls] = {
                "count": count,
                "pct": 100 * count / max(n_cycles, 1)
            }

        stats[split_name] = {
            "n_cycles": n_cycles,
            "pct_cycles": 100 * n_cycles / max(total, 1),
            "n_patients": n_patients,
            "pct_patients": 100 * n_patients / max(total_patients, 1),
            "class_distribution": class_dist,
        }

    return stats


def load_official_split(filepath):
    """
    Read ICBHI_challenge_train_test.txt ("<recording name>\\t<train|test>" per line).

    Returns a DataFrame with columns filename_base, official_split, patient_id.
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                filename = parts[0]
                split = parts[1]
                try:
                    patient_id = int(filename.split('_')[0])
                except ValueError:
                    continue
                records.append({
                    "filename_base": filename,
                    "official_split": split,
                    "patient_id": patient_id,
                })

    return pd.DataFrame(records)


def strategy1_official_split(df, config):
    """
    Official ICBHI split with the official train partition subdivided into train+val.

    Each cycle is mapped to the official partition of its source recording. The test
    set is kept identical to the official one. One StratifiedGroupKFold fold (~20%) of
    the patients found only in the official train partition becomes validation. Patients
    156 and 218 appear in both official partitions; they are left in train (never in
    val) and the overlap is kept so that results stay comparable with published work.
    """
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY 1: Official ICBHI split + validation")
    logger.info("=" * 70)

    if not ICBHI_OFFICIAL_SPLIT.exists():
        logger.error(f"Not found: {ICBHI_OFFICIAL_SPLIT}")
        logger.error("Place ICBHI_challenge_train_test.txt in data/raw/")
        return None

    df_official = load_official_split(ICBHI_OFFICIAL_SPLIT)
    logger.info(f"  Official split loaded: {len(df_official)} recordings")
    logger.info(f"    Official train: {len(df_official[df_official['official_split']=='train'])}")
    logger.info(f"    Official test:  {len(df_official[df_official['official_split']=='test'])}")

    # Map each cycle to the official partition through its source recording name
    df_s1 = df.copy()

    df_s1["filename_base"] = df_s1["filename_original"].apply(
        lambda x: Path(x).stem
    )

    df_s1 = df_s1.merge(
        df_official[["filename_base", "official_split"]],
        on="filename_base",
        how="left"
    )

    unmapped = df_s1["official_split"].isna().sum()
    if unmapped > 0:
        logger.warning(f"  {unmapped} cycles not mapped to the official split (excluded)")
        df_s1 = df_s1.dropna(subset=["official_split"]).copy()

    n_train_official = len(df_s1[df_s1["official_split"] == "train"])
    n_test_official = len(df_s1[df_s1["official_split"] == "test"])
    logger.info(f"  Mapped cycles: train={n_train_official}, test={n_test_official}")

    # Document the patient overlap of the official split
    train_patients = set(
        df_s1[df_s1["official_split"] == "train"]["patient_id"].unique()
    )
    test_patients = set(
        df_s1[df_s1["official_split"] == "test"]["patient_id"].unique()
    )
    overlap = train_patients & test_patients

    if overlap:
        logger.warning(f"  DATA LEAKAGE in the official split: {len(overlap)} patient(s)")
        logger.warning(f"     Patients: {sorted(overlap)}")
        logger.warning(f"     Kept for comparability with the literature.")
    else:
        logger.info(f"  No data leakage in the official split")

    # Subdivide the official train partition into train + validation by patient.
    # Patients that also appear in the official test partition stay in train.
    train_only_patients = train_patients - overlap

    df_train_official = df_s1[df_s1["official_split"] == "train"].copy()

    patient_classes = get_patient_dominant_class(df_train_official)

    df_train_only = df_train_official[
        df_train_official["patient_id"].isin(train_only_patients)
    ].copy()
    df_train_leaked = df_train_official[
        df_train_official["patient_id"].isin(overlap)
    ].copy()

    if len(df_train_only) > 0:
        patients_array = df_train_only["patient_id"].values
        patient_dominant = np.array([
            patient_classes.get(p, "Normal") for p in patients_array
        ])

        # K=5: one fold (~20%) becomes validation
        sgkf = StratifiedGroupKFold(
            n_splits=5, shuffle=True,
            random_state=config["random_seed"]
        )

        X_dummy = np.zeros(len(df_train_only))

        # Only the first fold is used
        for train_idx, val_idx in sgkf.split(
            X_dummy, patient_dominant, groups=patients_array
        ):
            df_train_only.iloc[val_idx, df_train_only.columns.get_loc("official_split")] = "val"
            break

    df_recombined = pd.concat([df_train_only, df_train_leaked], ignore_index=True)

    df_test_official = df_s1[df_s1["official_split"] == "test"].copy()

    df_s1_final = pd.concat([df_recombined, df_test_official], ignore_index=True)
    df_s1_final = df_s1_final.rename(columns={"official_split": "split"})

    stats = compute_split_stats(df_s1_final, "split")

    for split_name, s in stats.items():
        logger.info(
            f"  {split_name.upper():5s}: {s['n_cycles']:5d} cycles "
            f"({s['pct_cycles']:5.1f}%), {s['n_patients']:3d} patients "
            f"({s['pct_patients']:5.1f}%)"
        )

    # Check the train/val subdivision (the overlap with test is documented above)
    train_val_patients_train = set(
        df_s1_final[df_s1_final["split"] == "train"]["patient_id"].unique()
    )
    train_val_patients_val = set(
        df_s1_final[df_s1_final["split"] == "val"]["patient_id"].unique()
    )
    tv_overlap = train_val_patients_train & train_val_patients_val
    if tv_overlap:
        logger.warning(f"  Train/val leakage: {sorted(tv_overlap)}")
    else:
        logger.info(f"  No leakage between train and validation")

    STRATEGY1_DIR.mkdir(parents=True, exist_ok=True)

    df_train = df_s1_final[df_s1_final["split"] == "train"]
    df_val = df_s1_final[df_s1_final["split"] == "val"]
    df_test = df_s1_final[df_s1_final["split"] == "test"]

    df_train.to_csv(STRATEGY1_DIR / "train_cycles.csv", index=False)
    df_val.to_csv(STRATEGY1_DIR / "val_cycles.csv", index=False)
    df_test.to_csv(STRATEGY1_DIR / "test_cycles.csv", index=False)

    logger.info(f"  Files saved to: {STRATEGY1_DIR}")

    return df_s1_final, stats


def strategy2_custom_split(df, config):
    """
    Custom split with strict patient separation (no patient in more than one split).

    StratifiedGroupKFold with K=7 (~14.3% per fold): fold 0 is test, fold 1 is
    validation and folds 2-6 are train (~71/14/14).
    """
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY 2: Custom StratifiedGroupKFold (K=7)")
    logger.info("=" * 70)

    s2_config = config["strategy2"]
    seed = config["random_seed"]

    patient_classes = get_patient_dominant_class(df)
    patients_array = df["patient_id"].values
    patient_dominant = np.array([patient_classes[p] for p in patients_array])

    sgkf = StratifiedGroupKFold(
        n_splits=s2_config["n_splits_kfold"],
        shuffle=True,
        random_state=seed
    )

    X_dummy = np.zeros(len(df))

    all_fold_indices = []
    for _, test_idx in sgkf.split(X_dummy, patient_dominant, groups=patients_array):
        all_fold_indices.append(test_idx)

    df_s2 = df.copy()
    df_s2["split"] = "train"  # folds 2-6 keep this default

    df_s2.iloc[all_fold_indices[s2_config["test_fold"]],
               df_s2.columns.get_loc("split")] = "test"
    df_s2.iloc[all_fold_indices[s2_config["val_fold"]],
               df_s2.columns.get_loc("split")] = "val"

    is_clean = verify_no_patient_leakage(df_s2, "split")
    if is_clean:
        logger.info("  LEAKAGE CHECK: PASSED (zero patient overlap)")
    else:
        logger.error("  DATA LEAKAGE DETECTED: check the implementation")

    stats = compute_split_stats(df_s2, "split")
    for split_name, s in stats.items():
        logger.info(
            f"  {split_name.upper():5s}: {s['n_cycles']:5d} cycles "
            f"({s['pct_cycles']:5.1f}%), {s['n_patients']:3d} patients "
            f"({s['pct_patients']:5.1f}%)"
        )

    STRATEGY2_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_df = df_s2[df_s2["split"] == split_name]
        split_df.to_csv(STRATEGY2_DIR / f"{split_name}_cycles.csv", index=False)

    logger.info(f"  Files saved to: {STRATEGY2_DIR}")

    return df_s2, stats


def strategy3_kfold_cv(df, config):
    """
    Patient-grouped stratified K-fold cross-validation (K=5).

    Iteration i uses test = fold i, val = fold (i+1) % K and train = the remaining
    K-2 folds, so every patient is in test once, in val once and in train three times.
    Writes one train/val/test CSV triplet per iteration plus fold_assignments.csv.
    """
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY 3: Patient-level K-fold cross-validation (K=5)")
    logger.info("=" * 70)

    s3_config = config["strategy3"]
    K = s3_config["n_folds"]
    seed = config["random_seed"]

    patient_classes = get_patient_dominant_class(df)
    patients_array = df["patient_id"].values
    patient_dominant = np.array([patient_classes[p] for p in patients_array])

    sgkf = StratifiedGroupKFold(
        n_splits=K, shuffle=True, random_state=seed
    )

    X_dummy = np.zeros(len(df))

    fold_indices = []
    for _, test_idx in sgkf.split(X_dummy, patient_dominant, groups=patients_array):
        fold_indices.append(test_idx)

    logger.info(f"  {K} folds created with patient separation")
    for i, idx in enumerate(fold_indices):
        n_patients_fold = df.iloc[idx]["patient_id"].nunique()
        logger.info(
            f"    Fold {i+1}: {len(idx)} cycles, {n_patients_fold} patients"
        )

    STRATEGY3_DIR.mkdir(parents=True, exist_ok=True)

    all_fold_stats = []
    fold_assignments = df[["cycle_id", "patient_id", "cycle_class"]].copy()

    for iteration in range(K):
        test_fold_idx = iteration
        val_fold_idx = (iteration + 1) % K
        train_fold_indices = [
            j for j in range(K) if j != test_fold_idx and j != val_fold_idx
        ]

        logger.info(
            f"\n  Iteration {iteration + 1}/{K}: "
            f"test=fold_{test_fold_idx+1}, val=fold_{val_fold_idx+1}, "
            f"train=folds_{[j+1 for j in train_fold_indices]}"
        )

        df_iter = df.copy()
        df_iter["split"] = "train"  # remaining folds keep this default

        df_iter.iloc[fold_indices[test_fold_idx],
                     df_iter.columns.get_loc("split")] = "test"
        df_iter.iloc[fold_indices[val_fold_idx],
                     df_iter.columns.get_loc("split")] = "val"

        is_clean = verify_no_patient_leakage(df_iter, "split")
        if not is_clean:
            logger.error(f"  Leakage in iteration {iteration + 1}!")

        stats = compute_split_stats(df_iter, "split")
        all_fold_stats.append(stats)

        for split_name, s in stats.items():
            logger.info(
                f"    {split_name.upper():5s}: {s['n_cycles']:5d} cycles, "
                f"{s['n_patients']:3d} patients"
            )

        for split_name in ["train", "val", "test"]:
            split_df = df_iter[df_iter["split"] == split_name]
            filename = f"fold_{iteration + 1}_{split_name}.csv"
            split_df.to_csv(STRATEGY3_DIR / filename, index=False)

        col_name = f"fold_{iteration + 1}_split"
        fold_assignments[col_name] = df_iter["split"].values

    fold_assignments.to_csv(STRATEGY3_DIR / "fold_assignments.csv", index=False)
    logger.info(f"\n  Files saved to: {STRATEGY3_DIR}")
    logger.info(f"  Assignment table: fold_assignments.csv")

    return fold_indices, all_fold_stats


def generate_visualizations(df, stats_s1, stats_s2, all_fold_stats_s3, config):
    """Class-distribution figure for the three strategies and a summary table figure."""
    logger.info("\n[FIGURES] Generating comparison figures...")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        classes = ["Normal", "Crackle", "Wheeze", "Both"]
        colors_class = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]
        colors_split = {"train": "#3498db", "val": "#f39c12", "test": "#e74c3c"}

        # Figure 1: class distribution per split for the three strategies
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(
            "Class Distribution: Three Split Strategies\n"
            "ICBHI 2017 Dataset",
            fontsize=14, fontweight='bold'
        )

        strategy_data = [
            ("Strategy 1\nOfficial ICBHI Split\n+ Validation", stats_s1),
            ("Strategy 2\nStratifiedGroupKFold\n(Zero Leakage)", stats_s2),
            ("Strategy 3\nK-Fold CV (K=5)\nMean of 5 folds", None),
        ]

        for ax_idx, (title, stats) in enumerate(strategy_data):
            ax = axes[ax_idx]

            if stats is not None:
                # Strategies 1 and 2: bars grouped by split
                x = np.arange(len(classes))
                width = 0.25
                splits = ["train", "val", "test"]

                for i, split_name in enumerate(splits):
                    if split_name in stats:
                        pcts = [
                            stats[split_name]["class_distribution"].get(
                                c, {"pct": 0}
                            )["pct"]
                            for c in classes
                        ]
                        ax.bar(
                            x + i * width, pcts, width,
                            label=f"{split_name} (n={stats[split_name]['n_cycles']})",
                            color=colors_split[split_name], alpha=0.8,
                            edgecolor='white'
                        )

                ax.set_xticks(x + width)
                ax.set_xticklabels(classes)
                ax.legend(fontsize=8)

            else:
                # Strategy 3: mean ± SD over the K folds
                x = np.arange(len(classes))
                width = 0.25

                for split_idx, split_name in enumerate(["train", "val", "test"]):
                    means = []
                    stds = []
                    for cls in classes:
                        fold_pcts = []
                        for fold_stats in all_fold_stats_s3:
                            if split_name in fold_stats:
                                pct = fold_stats[split_name]["class_distribution"].get(
                                    cls, {"pct": 0}
                                )["pct"]
                                fold_pcts.append(pct)
                        means.append(np.mean(fold_pcts) if fold_pcts else 0)
                        stds.append(np.std(fold_pcts) if fold_pcts else 0)

                    avg_n = int(np.mean([
                        fs[split_name]["n_cycles"]
                        for fs in all_fold_stats_s3
                        if split_name in fs
                    ])) if all_fold_stats_s3 else 0

                    ax.bar(
                        x + split_idx * width, means, width,
                        yerr=stds, capsize=3,
                        label=f"{split_name} (n≈{avg_n})",
                        color=colors_split[split_name], alpha=0.8,
                        edgecolor='white'
                    )

                ax.set_xticks(x + width)
                ax.set_xticklabels(classes)
                ax.legend(fontsize=8)

            ax.set_title(title, fontsize=11)
            ax.set_ylabel("Percentage (%)")
            ax.set_xlabel("Class")
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "split_comparison_all_strategies.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Figure saved: {fig_path}")

        # Figure 2: summary table
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.axis('off')

        col_labels = [
            "Strategy", "Train\nCycles", "Val\nCycles", "Test\nCycles",
            "Train\nPat.", "Val\nPat.", "Test\nPat.",
            "Data\nLeakage", "Comparable to\nLiterature"
        ]

        table_data = []

        if stats_s1:
            row = [
                "1. Official ICBHI",
                f"{stats_s1.get('train', {}).get('n_cycles', 0)}",
                f"{stats_s1.get('val', {}).get('n_cycles', 0)}",
                f"{stats_s1.get('test', {}).get('n_cycles', 0)}",
                f"{stats_s1.get('train', {}).get('n_patients', 0)}",
                f"{stats_s1.get('val', {}).get('n_patients', 0)}",
                f"{stats_s1.get('test', {}).get('n_patients', 0)}",
                "2 pat.", "Yes"
            ]
            table_data.append(row)

        if stats_s2:
            row = [
                "2. GroupKFold",
                f"{stats_s2.get('train', {}).get('n_cycles', 0)}",
                f"{stats_s2.get('val', {}).get('n_cycles', 0)}",
                f"{stats_s2.get('test', {}).get('n_cycles', 0)}",
                f"{stats_s2.get('train', {}).get('n_patients', 0)}",
                f"{stats_s2.get('val', {}).get('n_patients', 0)}",
                f"{stats_s2.get('test', {}).get('n_patients', 0)}",
                "None", "No"
            ]
            table_data.append(row)

        if all_fold_stats_s3:
            avg_train = int(np.mean([
                fs.get("train", {}).get("n_cycles", 0)
                for fs in all_fold_stats_s3
            ]))
            avg_val = int(np.mean([
                fs.get("val", {}).get("n_cycles", 0)
                for fs in all_fold_stats_s3
            ]))
            avg_test = int(np.mean([
                fs.get("test", {}).get("n_cycles", 0)
                for fs in all_fold_stats_s3
            ]))
            avg_train_p = int(np.mean([
                fs.get("train", {}).get("n_patients", 0)
                for fs in all_fold_stats_s3
            ]))
            avg_val_p = int(np.mean([
                fs.get("val", {}).get("n_patients", 0)
                for fs in all_fold_stats_s3
            ]))
            avg_test_p = int(np.mean([
                fs.get("test", {}).get("n_patients", 0)
                for fs in all_fold_stats_s3
            ]))

            row = [
                "3. K-Fold CV (K=5)",
                f"~{avg_train}", f"~{avg_val}", f"~{avg_test}",
                f"~{avg_train_p}", f"~{avg_val_p}", f"~{avg_test_p}",
                "None", "No"
            ]
            table_data.append(row)

        table = ax2.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)

        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#34495e')
            table[0, j].set_text_props(color='white', fontweight='bold')

        ax2.set_title(
            "Summary of the Three Split Strategies\n"
            "ICBHI 2017 Dataset",
            fontsize=13, fontweight='bold', pad=20
        )

        plt.tight_layout()
        fig2_path = FIGURES_DIR / "split_summary_table.png"
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Table figure saved: {fig2_path}")

    except Exception as e:
        logger.warning(f"  Error generating figures: {e}")
        import traceback
        traceback.print_exc()


def create_all_splits():
    """Run the three strategies, then write the figures, split_config.json and the text report."""
    logger.info("=" * 70)
    logger.info("SPLIT CREATION (THREE STRATEGIES)")
    logger.info("=" * 70)
    start_time = time.time()

    for d in [SPLITS_DIR, RESULTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("\n[LOAD] Reading cycle table...")

    if FEATURE_METADATA_CSV.exists():
        df = pd.read_csv(FEATURE_METADATA_CSV)
        logger.info(f"  Loaded feature_metadata.csv: {len(df)} cycles")
    elif CYCLES_CSV.exists():
        df = pd.read_csv(CYCLES_CSV)
        logger.info(f"  Loaded processed_cycles.csv: {len(df)} cycles")
    else:
        logger.error("Cycle CSV not found. Run scripts 05 and 06 first.")
        sys.exit(1)

    required = ["cycle_id", "patient_id", "cycle_class", "filename_original"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        sys.exit(1)

    n_cycles = len(df)
    n_patients = df["patient_id"].nunique()
    logger.info(f"  Total: {n_cycles} cycles, {n_patients} patients")

    result_s1 = strategy1_official_split(df, SPLIT_CONFIG)
    stats_s1 = result_s1[1] if result_s1 else None

    result_s2 = strategy2_custom_split(df, SPLIT_CONFIG)
    stats_s2 = result_s2[1] if result_s2 else None

    fold_indices, all_fold_stats = strategy3_kfold_cv(df, SPLIT_CONFIG)

    generate_visualizations(df, stats_s1, stats_s2, all_fold_stats, SPLIT_CONFIG)

    config_record = {
        "script": "07_create_stratified_splits_v2.py",
        "timestamp": datetime.now().isoformat(),
        "config": {
            k: v for k, v in SPLIT_CONFIG.items()
            if k != "strategy3"
        },
        "strategy3_config": SPLIT_CONFIG["strategy3"],
        "dataset": {
            "total_cycles": n_cycles,
            "total_patients": n_patients,
        },
    }

    with open(SPLIT_CONFIG_FILE, 'w') as f:
        json.dump(config_record, f, indent=2, default=str)

    elapsed = time.time() - start_time
    report_lines = [
        "=" * 70,
        "SPLIT REPORT (THREE STRATEGIES)",
        "=" * 70,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total time: {elapsed:.1f} seconds",
        f"Dataset: {n_cycles} cycles, {n_patients} patients",
        "",
        "━" * 70,
        "STRATEGY 1: Official ICBHI split + validation",
        "━" * 70,
        "Purpose: direct comparability with the literature",
        f"Source file: {ICBHI_OFFICIAL_SPLIT.name}",
        "Data leakage: 2 patients (156, 218), documented and kept",
    ]

    if stats_s1:
        for split_name in ["train", "val", "test"]:
            if split_name in stats_s1:
                s = stats_s1[split_name]
                report_lines.append(
                    f"  {split_name.upper():5s}: {s['n_cycles']:5d} cycles "
                    f"({s['pct_cycles']:5.1f}%), {s['n_patients']:3d} pat. "
                    f"({s['pct_patients']:5.1f}%)"
                )
                for cls in ["Normal", "Crackle", "Wheeze", "Both"]:
                    d = s["class_distribution"].get(cls, {"count": 0, "pct": 0})
                    report_lines.append(
                        f"         {cls:10s}: {d['count']:5d} ({d['pct']:5.1f}%)"
                    )

    report_lines.extend([
        "",
        "━" * 70,
        "STRATEGY 2: StratifiedGroupKFold (K=7, zero leakage)",
        "━" * 70,
        "Purpose: evaluation without data leakage",
        "Data leakage: NONE (verified)",
    ])

    if stats_s2:
        for split_name in ["train", "val", "test"]:
            if split_name in stats_s2:
                s = stats_s2[split_name]
                report_lines.append(
                    f"  {split_name.upper():5s}: {s['n_cycles']:5d} cycles "
                    f"({s['pct_cycles']:5.1f}%), {s['n_patients']:3d} pat. "
                    f"({s['pct_patients']:5.1f}%)"
                )

    report_lines.extend([
        "",
        "━" * 70,
        "STRATEGY 3: K-fold cross-validation (K=5, by patient)",
        "━" * 70,
        "Purpose: robust evaluation with confidence intervals",
        "Data leakage: NONE in any fold",
        f"Models to train: {SPLIT_CONFIG['strategy3']['n_folds']}",
    ])

    for i, fold_stats in enumerate(all_fold_stats):
        report_lines.append(f"\n  Fold {i+1}:")
        for split_name in ["train", "val", "test"]:
            if split_name in fold_stats:
                s = fold_stats[split_name]
                report_lines.append(
                    f"    {split_name.upper():5s}: {s['n_cycles']:5d} cycles, "
                    f"{s['n_patients']:3d} pat."
                )

    report_lines.extend([
        "",
        "=" * 70,
        "CONCLUSION",
        "=" * 70,
        "All three strategies were created.",
        "",
        "Intended use:",
        "  1. Strategy 1 for the main results (comparability)",
        "  2. Strategy 2 to check robustness (no leakage)",
        "  3. Strategy 3 to report metrics with 95% CIs",
        "",
        "Files available in:",
        f"  {STRATEGY1_DIR}",
        f"  {STRATEGY2_DIR}",
        f"  {STRATEGY3_DIR}",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)

    logger.info(f"\nReport saved: {REPORT_FILE}")

    logger.info("\n" + "=" * 70)
    logger.info("THREE SPLIT STRATEGIES COMPLETED")
    logger.info(f"   Strategy 1: official ICBHI split + validation")
    logger.info(f"   Strategy 2: StratifiedGroupKFold (zero leakage)")
    logger.info(f"   Strategy 3: K-fold CV (K=5, metrics with CIs)")
    logger.info(f"   Files in: {SPLITS_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Split creation (three strategies)")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Strategy 1: official ICBHI split + validation")
    print(f"Strategy 2: custom StratifiedGroupKFold (zero leakage)")
    print(f"Strategy 3: patient-level K-fold CV (K=5)")
    print("=" * 70 + "\n")

    found_data = False
    for csv_path in [FEATURE_METADATA_CSV, CYCLES_CSV]:
        if csv_path.exists():
            found_data = True
            break

    if not found_data:
        print("ERROR: cycle CSV not found.")
        print("Run scripts 05 and 06 first.")
        sys.exit(1)

    if not ICBHI_OFFICIAL_SPLIT.exists():
        print(f"WARNING: not found: {ICBHI_OFFICIAL_SPLIT}")
        print("Strategy 1 will not run.")
        print("Place ICBHI_challenge_train_test.txt in data/raw/")

    create_all_splits()
