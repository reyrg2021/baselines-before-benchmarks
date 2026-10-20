# -*- coding: utf-8 -*-
"""
rodri2_roc.py: ROC figure for the binary Normal-vs-abnormal problem.

Plots the mean ROC curve of the six configurations on the official ICBHI 2017 test
set (2733 cycles), averaged over five seeds. The abnormality score is 1 - p(Normal)
and the binary label is (y_true != 0). The legend AUC is the mean of the per-seed
AUCs computed on raw scores, not the area under the averaged curve; the drawn curve
interpolates each seed's ROC on a common FPR grid and averages the TPR. The AUC
reproduces the means in master_results.json to four decimals. Standard deviations
are logged as sample SD (ddof=1; master_results.json stores ddof=0) and are not
drawn, so ddof does not affect the image.

Inputs:  results/metrics/full_reeval/arrays/<model>_seed<S>_probs.npy (2733, 4),
         <model>_seed<S>_labels_preds.npz ('y_true', 'y_pred'); classes
         0 = Normal, 1 = Crackle, 2 = Wheeze, 3 = Both. Override with env ARRAYS_DIR.
Outputs: rodri2.png (600 dpi), rodri2.pdf, rodri2_roc_log.txt (in OUT_DIR, default scripts/)
Usage:
    python rodri2_roc.py
"""

import os
import sys
import logging
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Configuration
# ARRAYS_DIR holds the .npy/.npz files of the multi-seed evaluation; it can be
# overridden with the ARRAYS_DIR environment variable.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ARRAYS_DIR = os.environ.get(
    "ARRAYS_DIR",
    os.path.join(os.path.dirname(BASE_DIR), "results", "metrics", "full_reeval", "arrays"),
)
OUT_DIR    = os.environ.get("OUT_DIR", BASE_DIR)

OUT_PDF = os.path.join(OUT_DIR, "rodri2.pdf")
OUT_PNG = os.path.join(OUT_DIR, "rodri2.png")
LOG_TXT = os.path.join(OUT_DIR, "rodri2_roc_log.txt")

SEEDS = [123, 314, 42, 7, 99]   # same order as master_results.json
N_FPR_GRID = 200                # points of the common FPR grid
DPI_PNG = 600                   # PNG resolution (dpi)
DDOF = 1                        # sample SD

# The six configurations, in legend order (descending AUC).
# (file_pattern, legend_label, color, line_width, line_style)
# Legend labels match the row names of the results table.
MODEL_SPECS = [
    ("cnn_seed{s}",
     "CNN-RespiratoryNet",                 "#1f4e79", 2.4, "-"),
    ("panns_cnn6_official_seed{s}",
     "PANNs CNN6 (official)",              "#2e8b57", 1.6, "-"),
    ("tl_mobilenet_v3_small_pretrained_official_seed{s}",
     "MobileNetV3-Small (ImageNet)",       "#d97706", 1.6, "-"),
    ("tl_mobilenet_v3_small_official_seed{s}",
     "MobileNetV3-Small (from scratch)",   "#b58900", 1.4, "-"),
    ("panns_cnn6_tune_official_seed{s}",
     "PANNs CNN6 (tune)",                  "#8e44ad", 1.4, "-"),
    ("hybrid_seed{s}",
     "CfC-Hybrid",                         "#c0392b", 1.8, "-"),
]


# Logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_TXT, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)
    log.info("Start | %s", datetime.now().isoformat(timespec="seconds"))
    log.info("ARRAYS_DIR = %s", ARRAYS_DIR)
    log.info("Output     = %s", OUT_PNG)
    log.info("SD reported in the log: sample (ddof=%d)", DDOF)
    return log


# Loading and per-model ROC
def load_seed(pattern, seed):
    """Return (binary_label, abnormality_score) for one model and seed."""
    base = os.path.join(ARRAYS_DIR, pattern.format(s=seed))
    y_true = np.load(base + "_labels_preds.npz")["y_true"]
    probs  = np.load(base + "_probs.npy")
    y_bin  = (y_true != 0).astype(int)     # 0 = Normal, 1 = abnormal
    score  = 1.0 - probs[:, 0]             # abnormality probability
    return y_bin, score


def mean_roc_model(pattern, log):
    """
    Mean ROC over seeds: each ROC is interpolated on a common FPR grid and the
    TPR averaged. The AUC is computed per seed on raw scores (not on the
    interpolated curve) and then averaged.

    Returns (mean_fpr, mean_tpr, auc_mean, auc_std).
    """
    mean_fpr = np.linspace(0.0, 1.0, N_FPR_GRID)
    tprs, aucs = [], []
    for s in SEEDS:
        y_bin, score = load_seed(pattern, s)
        fpr, tpr, _ = roc_curve(y_bin, score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_score(y_bin, score))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    aucs = np.array(aucs)

    auc_mean = aucs.mean()
    auc_std  = aucs.std(ddof=DDOF)         # sample SD

    # Both SD forms are logged so the published figure can be audited.
    log.info("  %-44s AUC = %.4f ± %.4f (ddof=1) | ± %.4f (ddof=0) | per-seed %s",
             pattern, auc_mean, auc_std, aucs.std(ddof=0), np.round(aucs, 4))
    return mean_fpr, mean_tpr, auc_mean, auc_std


# Figure
def build_figure(log):
    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 0.8,
        "figure.dpi": 120,
    })
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    for pattern, label_text, color, lw, ls in MODEL_SPECS:
        fpr, tpr, auc_m, _ = mean_roc_model(pattern, log)
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=f"{label_text} ({auc_m:.3f})")

    # Chance line
    ax.plot([0, 1], [0, 1], color="0.4", lw=1.0, ls="--",
            label="Chance (0.500)")

    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("False positive rate (1 \u2212 Specificity)")
    ax.set_ylabel("True positive rate (Sensitivity)")
    ax.set_aspect("equal", adjustable="box")

    # Compact legend in the empty lower-right triangle of the ROC plot; only the
    # mean AUC is shown (the SD is in the log).
    leg = ax.legend(loc="lower right", title="Model (mean AUC)",
                    fontsize=9, title_fontsize=9, framealpha=0.95,
                    labelspacing=0.3, handlelength=1.6,
                    handletextpad=0.5, borderpad=0.5)
    leg._legend_box.align = "left"
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    fig.tight_layout()

    # 600 dpi PNG for line art, plus a vector PDF.
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=DPI_PNG)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    log.info("Figure saved: %s (%d dpi) | %s", OUT_PNG, DPI_PNG, OUT_PDF)
    plt.close(fig)


# Main
def main():
    log = setup_logging()
    if not os.path.isdir(ARRAYS_DIR):
        log.error("ARRAYS_DIR not found: %s", ARRAYS_DIR)
        log.error("Set the ARRAYS_DIR environment variable or edit the path configuration.")
        sys.exit(1)
    build_figure(log)
    log.info("End | OK")


if __name__ == "__main__":
    main()
