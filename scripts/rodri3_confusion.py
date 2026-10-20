# -*- coding: utf-8 -*-
"""
Figure 3: Confusion matrix of CNN-RespiratoryNet on the official ICBHI 2017 test set.

Sums the per-seed 4x4 confusion matrices (rows = true class, columns = predicted;
classes 0 = Normal, 1 = Crackle, 2 = Wheeze, 3 = Both) over the five seeds and
normalizes by row, so the diagonal is the per-class sensitivity. Proportions are
computed on the summed matrix (more stable estimate); each cell also shows the
mean count per seed (sum / 5). The diagonal is outlined in red.

Inputs:  results/metrics/full_reeval/arrays/cnn_seed<S>_confusion.npy (or ARRAYS_DIR)
Outputs: rodri3.png (600 dpi), rodri3.pdf, rodri3_confusion_log.txt (in OUT_DIR)
Usage:
    python scripts/rodri3_confusion.py
"""

import os
import sys
import logging
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ARRAYS_DIR = os.environ.get("ARRAYS_DIR", os.path.join(os.path.dirname(BASE_DIR), "results", "metrics", "full_reeval", "arrays"))
OUT_DIR    = os.environ.get("OUT_DIR", BASE_DIR)

OUT_PNG = os.path.join(OUT_DIR, "rodri3.png")
OUT_PDF = os.path.join(OUT_DIR, "rodri3.pdf")
LOG_TXT = os.path.join(OUT_DIR, "rodri3_confusion_log.txt")

SEEDS   = [123, 314, 42, 7, 99]
CLASS_LABELS  = ["Normal", "Crackle", "Wheeze", "Both"]
DPI_PNG = 600   # PNG resolution for submission


def setup_logging():
    """Log to stdout and to LOG_TXT."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(LOG_TXT, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)
    log.info("Start | %s", datetime.now().isoformat(timespec="seconds"))
    log.info("ARRAYS_DIR = %s", ARRAYS_DIR)
    return log


def aggregated_matrix(log):
    """Sum the 4x4 confusion matrices of the five seeds."""
    row_sum = np.zeros((4, 4), dtype=np.float64)
    for s in SEEDS:
        cm = np.load(os.path.join(ARRAYS_DIR, f"cnn_seed{s}_confusion.npy"))
        row_sum += cm
    sens = np.diag(row_sum) / row_sum.sum(axis=1)
    log.info("Total cycles (5 seeds) = %d", int(row_sum.sum()))
    log.info("Per-class sensitivity = %s", dict(zip(CLASS_LABELS, np.round(sens, 3))))
    return row_sum


def build_figure(row_sum, log):
    """Plot the row-normalized matrix with percentage and mean count per cell."""
    row_total = row_sum.sum(axis=1, keepdims=True)
    prop = row_sum / row_total          # row proportion (colour and %)
    mean_count = np.round(row_sum / len(SEEDS)).astype(int)   # mean count per seed

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(prop, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

    for i in range(4):
        for j in range(4):
            color = "white" if prop[i, j] > 0.55 else "black"
            ax.text(j, i - 0.12, f"{prop[i, j]*100:.0f}%",
                    ha="center", va="center", color=color, fontsize=12, fontweight="bold")
            ax.text(j, i + 0.20, f"{mean_count[i, j]}",
                    ha="center", va="center", color=color, fontsize=9)

    for k in range(4):
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                               edgecolor="#c0392b", lw=2.0))

    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion (sensitivity on diagonal)")
    fig.tight_layout()

    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=DPI_PNG)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    log.info("Figure saved: %s (%d dpi) | %s", OUT_PNG, DPI_PNG, OUT_PDF)
    plt.close(fig)


def main():
    log = setup_logging()
    if not os.path.isdir(ARRAYS_DIR):
        log.error("ARRAYS_DIR does not exist: %s", ARRAYS_DIR); sys.exit(1)
    row_sum = aggregated_matrix(log)
    build_figure(row_sum, log)
    log.info("End | OK")


if __name__ == "__main__":
    main()
