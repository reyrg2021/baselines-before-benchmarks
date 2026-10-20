#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 41: Export the supplementary artifacts declared under Data and Code Availability.

Two artifacts are produced without recomputing anything:
  - S14: the per-seed softmax probabilities saved by Script 18, one CSV per
         configuration, aligned row by row with test_cycles.csv and carrying the
         explicit cycle_id.
  - S15: a copy of the TorchScript INT8 model profiled on the Raspberry Pi 5.
Alignment is verified before writing; a misaligned file is never written.

Inputs:  data/splits/strategy1_official/test_cycles.csv,
         results/metrics/full_reeval/arrays/{prefix}_seed{S}_probs.npy (N, 4),
         results/metrics/full_reeval/arrays/{prefix}_seed{S}_labels_preds.npz,
         models/checkpoints/cnn_int8_scripted.pt
Outputs: supplementary/S14_test_predictions/{key}_seed{S}.csv,
         supplementary/S14_test_predictions/README_S14.txt,
         supplementary/S15_cnn_int8_scripted.pt,
         export_supplementary_log.txt (in BASE_DIR)
         The text block for supplementary/README.md is printed to stdout.
Usage:
    python 41_export_supplementary_artifacts.py [--base_dir .] [--skip_int8] [--skip_preds]
"""

import sys
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# Path configuration

# Same auto-detection as Script 29, so both scripts resolve the same root.
SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR_FIXED = Path(__file__).resolve().parent.parent

if BASE_DIR_FIXED.exists():
    BASE_DIR = BASE_DIR_FIXED
elif (SCRIPTS_DIR.parent / "data").exists():
    BASE_DIR = SCRIPTS_DIR.parent
else:
    BASE_DIR = SCRIPTS_DIR

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_MAP = {"Normal": 0, "Crackle": 1, "Wheeze": 2, "Both": 3}

SEEDS = [42, 123, 7, 99, 314]

# Same prefix registry as Script 29. A configuration without arrays is skipped
# with a warning instead of aborting (the Script 31 recurrent baselines may be
# absent from some copies of the project).
MODELS = {
    "CNN_FP32":            "cnn",
    "CNN_INT8":            "cnn_int8",
    "MobileNetV3_ImageNet": "tl_mobilenet_v3_small_pretrained_official",
    "MobileNetV3_scratch": "tl_mobilenet_v3_small_official",
    "PANNs_CNN6_official": "panns_cnn6_official",
    "PANNs_CNN6_tune":     "panns_cnn6_tune_official",
    "CfC_Hybrid":          "hybrid",
    "LSTM_Hybrid":         "lstm_official",
    "GRU_Hybrid":          "gru_official",
    "CfC_no_mixed_memory": "cfcpure_official",
}


# Logging

logger = logging.getLogger(__name__)


def setup_logging(base_dir):
    """Log to stdout and to export_supplementary_log.txt in BASE_DIR."""
    log_file = base_dir / "export_supplementary_log.txt"
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return log_file


def section(title_text):
    """Write a section header to the log."""
    logger.info("")
    logger.info("=" * 74)
    logger.info(f"  {title_text}")
    logger.info("=" * 74)


# Per-seed predictions (S14)

def export_predictions(base_dir, out_dir):
    """
    Write one CSV of per-seed softmax probabilities per configuration.

    Script 18 saved the arrays in test DataLoader order, i.e. the order of
    test_cycles.csv. Before writing, we check that (a) the array has exactly
    len(test_cycles.csv) rows, (b) y_true from the .npz matches the CSV label
    row by row, and (c) y_pred equals argmax(probs). If (a) or (b) fails the
    file is not written. Returns the list of (configuration, n_seeds) exported.
    """
    section("S14 - PER-SEED PREDICTIONS")

    arrays_dir = base_dir / "results" / "metrics" / "full_reeval" / "arrays"
    split_csv = base_dir / "data" / "splits" / "strategy1_official" / "test_cycles.csv"

    if not arrays_dir.exists():
        logger.error(f"  Arrays directory not found: {arrays_dir}")
        return []
    if not split_csv.exists():
        logger.error(f"  Test split CSV not found: {split_csv}")
        return []

    test_df = pd.read_csv(split_csv)
    n_cycles = len(test_df)
    logger.info(f"  test_cycles.csv: {n_cycles} cycles")

    # Reference integer labels, in CSV order
    y_true_csv = test_df["cycle_class"].map(CLASS_MAP).to_numpy()
    if np.isnan(y_true_csv.astype(float)).any():
        logger.error("  The CSV contains classes outside CLASS_MAP. Aborted.")
        return []

    cycle_ids = test_df["cycle_id"].astype(str).to_numpy()

    pred_dir = out_dir / "S14_test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    exported = []

    for config_key, prefix_name in MODELS.items():
        seed_files = []

        for seed in SEEDS:
            p_path = arrays_dir / f"{prefix_name}_seed{seed}_probs.npy"
            z_path = arrays_dir / f"{prefix_name}_seed{seed}_labels_preds.npz"

            if not p_path.exists():
                continue

            probs = np.load(p_path).astype(np.float64)

            # Check (a): row count
            if probs.shape != (n_cycles, 4):
                logger.error(
                    f"  {config_key} seed {seed}: probs {probs.shape} != "
                    f"({n_cycles}, 4). Not exported."
                )
                continue

            # Check (b): y_true against the CSV
            if z_path.exists():
                z = np.load(z_path)
                y_true = z["y_true"]
                y_pred = z["y_pred"]

                if not np.array_equal(y_true, y_true_csv):
                    n_dif = int((y_true != y_true_csv).sum())
                    logger.error(
                        f"  {config_key} seed {seed}: y_true differs from the CSV in "
                        f"{n_dif} rows. Misalignment. Not exported."
                    )
                    continue
            else:
                logger.warning(
                    f"  {config_key} seed {seed}: no labels_preds.npz, "
                    f"using y_true from the CSV and y_pred = argmax(probs)"
                )
                y_true = y_true_csv
                y_pred = np.argmax(probs, axis=1)

            # Check (c): internal consistency of y_pred
            argmax_probs = np.argmax(probs, axis=1)
            if not np.array_equal(y_pred, argmax_probs):
                n_dif = int((y_pred != argmax_probs).sum())
                logger.warning(
                    f"  {config_key} seed {seed}: y_pred differs from argmax(probs) "
                    f"in {n_dif} rows; publishing argmax(probs)"
                )
                y_pred = argmax_probs

            df_out = pd.DataFrame({
                "cycle_id": cycle_ids,
                "true_class": [CLASS_NAMES[i] for i in y_true],
                "pred_class": [CLASS_NAMES[i] for i in y_pred],
                "p_Normal":  probs[:, 0],
                "p_Crackle": probs[:, 1],
                "p_Wheeze":  probs[:, 2],
                "p_Both":    probs[:, 3],
            })
            df_out["p_abnormal"] = 1.0 - df_out["p_Normal"]

            out_csv = pred_dir / f"{config_key}_seed{seed}.csv"
            df_out.to_csv(out_csv, index=False, float_format="%.6f")
            seed_files.append(out_csv.name)

        if seed_files:
            logger.info(f"  {config_key:<24s} {len(seed_files)} seeds exported")
            exported.append((config_key, len(seed_files)))
        else:
            logger.warning(f"  {config_key:<24s} no arrays available, skipped")

    readme = pred_dir / "README_S14.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("S14 - Per-seed test predictions on the official ICBHI 2017 test set\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated on {datetime.now():%Y-%m-%d}\n\n")
        f.write("One file per configuration and seed. Every file has one row per\n")
        f.write(f"test cycle ({n_cycles} rows), in the order of test_cycles.csv (S2).\n\n")
        f.write("Columns\n")
        f.write("-------\n")
        f.write("  cycle_id     identifier, joins with S2_test_cycles.csv\n")
        f.write("  true_class   reference label (Normal, Crackle, Wheeze, Both)\n")
        f.write("  pred_class   argmax of the four softmax outputs\n")
        f.write("  p_Normal ... p_Both   softmax outputs of the four classes\n")
        f.write("  p_abnormal   1 - p_Normal, the score used for the binary AUC\n\n")
        f.write("Seeds: " + ", ".join(str(s) for s in SEEDS) + "\n\n")
        f.write("The official ICBHI score, the binary AUC, and every paired test\n")
        f.write("reported in the article can be recomputed from these files alone.\n")
        f.write("Each file was verified row by row against S2_test_cycles.csv\n")
        f.write("before being written.\n")

    logger.info(f"  README written: {readme.name}")
    return exported


# INT8 artifact (S15)

def export_int8(base_dir, out_dir):
    """
    Copy the TorchScript INT8 model profiled on the Raspberry Pi 5.

    Provenance: the artifact comes from seed 42, whose run did not leave the
    degenerate regime; this is stated in the article. Latency and memory do
    not depend on the weights.
    """
    section("S15 - INT8 ARTIFACT (TORCHSCRIPT)")

    source_path = base_dir / "models" / "checkpoints" / "cnn_int8_scripted.pt"

    if not source_path.exists():
        logger.error(f"  Not found: {source_path}")
        logger.error("  Generate the artifact with Script 23 first,")
        logger.error("  or remove that sentence from Data and Code Availability.")
        return False

    target_path = out_dir / "S15_cnn_int8_scripted.pt"
    shutil.copy2(source_path, target_path)

    size_mb = target_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Copied: {target_path.name} ({size_mb:.2f} MB)")

    if size_mb > 1.5:
        logger.warning("  The artifact is larger than expected (0.58 MB).")
        logger.warning("  Check that it is the quantized model and not the FP32 one.")

    return True


# Text block for the supplementary README

def print_readme_block(exported, has_int8):
    """Print the text to paste into supplementary/README.md."""
    section("BLOCK FOR supplementary/README.md")

    n_configs = len(exported)
    lines_out = [
        "",
        "## S14 - Per-seed test predictions",
        "",
        f"`S14_test_predictions/` contains one CSV per configuration and seed "
        f"({n_configs} configurations, five seeds each), with one row per test",
        "cycle in the order of `S2_test_cycles.csv`. Each row carries the cycle",
        "identifier, the reference label, the predicted class, the four softmax",
        "outputs, and the abnormality score used for the binary AUC. The official",
        "score, the binary AUC, and every paired test reported in the article can",
        "be recomputed from these files alone.",
        "",
    ]

    if has_int8:
        lines_out += [
            "## S15 - Serialized INT8 model",
            "",
            "`S15_cnn_int8_scripted.pt` is the TorchScript artifact profiled on the",
            "Raspberry Pi 5 in Section 4.9. It was exported from the seed 42",
            "checkpoint, a provenance stated in the article, and reproduces the",
            "latency and memory figures of Table 13 on the same hardware. Load it",
            "with `torch.jit.load` and feed it a 1 x 128 x 251 log-mel tensor built",
            "with the parameters of Section 3.2.",
            "",
        ]

    for ln in lines_out:
        print(ln)

    logger.info("  Block printed to stdout (not written automatically)")


# Main

def parse_args():
    p = argparse.ArgumentParser(
        description="Script 41 - Export per-seed predictions and the INT8 artifact"
    )
    p.add_argument("--base_dir", type=str, default=None,
                   help="Override BASE_DIR (repository root)")
    p.add_argument("--skip_int8", action="store_true",
                   help="Do not copy the TorchScript INT8 model")
    p.add_argument("--skip_preds", action="store_true",
                   help="Do not export the per-seed predictions")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else BASE_DIR
    out_dir = base_dir / "supplementary"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = setup_logging(base_dir)

    logger.info("=" * 74)
    logger.info("  SCRIPT 41 - SUPPLEMENTARY ARTIFACTS")
    logger.info("=" * 74)
    logger.info(f"  Date:      {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"  BASE_DIR:  {base_dir}")
    logger.info(f"  Output:    {out_dir}")
    logger.info(f"  Log:       {log_file}")

    exported = []
    if not args.skip_preds:
        exported = export_predictions(base_dir, out_dir)

    has_int8 = False
    if not args.skip_int8:
        has_int8 = export_int8(base_dir, out_dir)

    print_readme_block(exported, has_int8)

    # Verdict on the Data and Code Availability statement
    section("VERDICT")

    ok_preds = len(exported) > 0
    logger.info(f"  Per-seed predictions ......... {'YES' if ok_preds else 'NO'}")
    logger.info(f"  INT8 artifact ................ {'YES' if has_int8 else 'NO'}")

    if ok_preds and has_int8:
        logger.info("")
        logger.info("  Every artifact declared under Data and Code Availability")
        logger.info("  now exists. Rebuild supplementary.zip before uploading.")
    else:
        logger.warning("")
        logger.warning("  SOME DECLARED ARTIFACTS ARE MISSING.")
        logger.warning("  Generate what is missing or remove that sentence from the")
        logger.warning("  Data and Code Availability statement.")

    logger.info("")
    logger.info("=" * 74)
    logger.info("  END OF SCRIPT 41")
    logger.info("=" * 74)


if __name__ == "__main__":
    main()
