# -*- coding: utf-8 -*-
"""
rodri5_arch.py: Architecture figure of CNN-RespiratoryNet.

Draws the four convolutional blocks, the adaptive pooling and the classifier head
with the tensor shape at each stage, plus an inset with the internal ConvBlock.
All values follow 09_cnn_training.py: base_filters=16 and n_conv_blocks=4 (16, 32,
64, 128 filters); ConvBlock = [Conv2D 3x3 -> BN -> ReLU] x2 -> MaxPool2D 2x2;
AdaptiveAvgPool2d((4,4)) -> flatten 2048; Linear(2048,128) -> ReLU -> Dropout(0.5)
-> Linear(128,4); 556,500 parameters (master_results.json). Spatial shapes follow
from a 1x128x251 input with padding=1 and 2x2 max pooling (integer halving):
1x128x251 -> 16x64x125 -> 32x32x62 -> 64x16x31 -> 128x8x15 -> (pool) 128x4x4.

Outputs: rodri5.png (600 dpi), rodri5.pdf, rodri5_arch_log.txt (in OUT_DIR, default scripts/)
Usage:
    python rodri5_arch.py
"""

import os
import sys
import logging
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.environ.get("OUT_DIR", BASE_DIR)

OUT_PNG = os.path.join(OUT_DIR, "rodri5.png")
OUT_PDF = os.path.join(OUT_DIR, "rodri5.pdf")
LOG_TXT = os.path.join(OUT_DIR, "rodri5_arch_log.txt")

DPI_PNG = 600   # PNG resolution (dpi)

# Palette (same as rodri1)
BLUE_FILL, BLUE_EDGE   = "#dbe9f6", "#1f4e79"   # convolutional backbone
GREEN_FILL, GREEN_EDGE = "#cfe8dc", "#2e8b57"   # input / output
HEAD_FILL, HEAD_EDGE   = "#eef2f7", "#6a7a8c"   # head (pool + classifier)
SHAPE_COLOR            = "#3a3a3a"              # tensor-shape text

# Main column: (title, tensor shape, type); type sets the color: "in" | "conv" | "head" | "out"
STAGES = [
    ("Log-mel input",                          "1 \u00d7 128 \u00d7 251",  "in"),
    ("Conv block 1 \u2014 16 filters",         "16 \u00d7 64 \u00d7 125",  "conv"),
    ("Conv block 2 \u2014 32 filters",         "32 \u00d7 32 \u00d7 62",   "conv"),
    ("Conv block 3 \u2014 64 filters",         "64 \u00d7 16 \u00d7 31",   "conv"),
    ("Conv block 4 \u2014 128 filters",        "128 \u00d7 8 \u00d7 15",   "conv"),
    ("Adaptive avg pooling 4\u00d74",          "128 \u00d7 4 \u00d7 4",    "head"),
    ("Flatten",                                "2048",                     "head"),
    ("FC 2048 \u2192 128 \u00b7 ReLU \u00b7 Dropout 0.5", "128",           "head"),
    ("FC 128 \u2192 4  (logits)",              "4",                        "out"),
]

# Internal ConvBlock detail (top to bottom)
CONVBLOCK_CHIPS = [
    "Conv 3\u00d73", "BatchNorm", "ReLU",
    "Conv 3\u00d73", "BatchNorm", "ReLU",
    "MaxPool 2\u00d72",
]

KIND_COLORS = {
    "in":   (GREEN_FILL, GREEN_EDGE),
    "conv": (BLUE_FILL,  BLUE_EDGE),
    "head": (HEAD_FILL,  HEAD_EDGE),
    "out":  (GREEN_FILL, GREEN_EDGE),
}

FOOTER = "RespiratoryNet  \u00b7  ~556.5 K parameters  \u00b7  trained from scratch"


# Logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(LOG_TXT, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)
    log.info("Start | %s", datetime.now().isoformat(timespec="seconds"))
    return log


# Figure
def build_figure(log):
    n = len(STAGES)
    box_w, box_h = 0.50, 0.62      # box width/height (data coordinates)
    gap = 0.30                     # vertical gap between boxes
    step_info = box_h + gap

    cx_main   = 0.36               # main column center
    cx_inset  = 0.82               # ConvBlock inset center
    inset_w   = 0.30

    fig, ax = plt.subplots(figsize=(6.6, 9.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.35, n * step_info + 0.1)
    ax.axis("off")

    # Main column
    centers_y = []
    for k, (title_text, shape_text, kind) in enumerate(STAGES):
        cy = (n * step_info) - (k * step_info) - box_h / 2      # top (k=0) to bottom
        centers_y.append(cy)
        fill, edge = KIND_COLORS[kind]

        box = FancyBboxPatch((cx_main - box_w / 2, cy - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2)
        ax.add_patch(box)
        ax.text(cx_main, cy + 0.10, title_text, ha="center", va="center",
                fontsize=9.8, fontweight="bold", color=edge, zorder=3)
        ax.text(cx_main, cy - 0.14, shape_text, ha="center", va="center",
                fontsize=8.0, style="italic", color=SHAPE_COLOR, zorder=3)
        log.info("  Stage %d: %-34s | %s", k + 1, title_text, shape_text)

    # Downward arrows
    for k in range(n - 1):
        y0 = centers_y[k] - box_h / 2
        y1 = centers_y[k + 1] + box_h / 2
        ax.add_patch(FancyArrowPatch((cx_main, y0), (cx_main, y1),
                                     arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.6, color="#555555", zorder=1))

    # ConvBlock inset, aligned with the conv blocks (indices 1..4 of STAGES)
    y_top    = centers_y[1] + box_h / 2
    y_bottom = centers_y[4] - box_h / 2
    cont_h   = y_top - y_bottom
    cont = FancyBboxPatch((cx_inset - inset_w / 2, y_bottom), inset_w, cont_h,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.3, edgecolor=BLUE_EDGE, facecolor="#ffffff",
                          linestyle="--", zorder=2)
    ax.add_patch(cont)
    ax.text(cx_inset, y_top - 0.16, "ConvBlock  (\u00d74)", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=BLUE_EDGE, zorder=3)

    # Inner chips
    m = len(CONVBLOCK_CHIPS)
    chip_w, chip_h = 0.24, 0.20
    top_chip = y_top - 0.40
    bot_chip = y_bottom + 0.14
    span = top_chip - bot_chip
    chip_cys = [top_chip - (span * i / (m - 1)) for i in range(m)]
    for i, (label, ccy) in enumerate(zip(CONVBLOCK_CHIPS, chip_cys)):
        chip = FancyBboxPatch((cx_inset - chip_w / 2, ccy - chip_h / 2), chip_w, chip_h,
                              boxstyle="round,pad=0.01,rounding_size=0.03",
                              linewidth=1.0, edgecolor=BLUE_EDGE, facecolor=BLUE_FILL,
                              zorder=3)
        ax.add_patch(chip)
        ax.text(cx_inset, ccy, label, ha="center", va="center",
                fontsize=7.4, color=BLUE_EDGE, zorder=4)
        if i < m - 1:
            ax.add_patch(FancyArrowPatch((cx_inset, ccy - chip_h / 2),
                                         (cx_inset, chip_cys[i + 1] + chip_h / 2),
                                         arrowstyle="-|>", mutation_scale=9,
                                         linewidth=1.0, color="#888888", zorder=2))

    # Dashed connector: conv blocks -> inset
    x_brace = cx_main + box_w / 2 + 0.015
    y_mid   = (centers_y[1] + centers_y[4]) / 2
    ax.plot([x_brace, x_brace], [y_bottom + 0.10, y_top - 0.10],
            linestyle="--", linewidth=1.2, color=BLUE_EDGE, zorder=1)
    ax.plot([x_brace, cx_inset - inset_w / 2], [y_mid, y_mid],
            linestyle="--", linewidth=1.2, color=BLUE_EDGE, zorder=1)

    # Footer
    ax.text(0.5, -0.22, FOOTER, ha="center", va="center",
            fontsize=8.4, color="#444444")

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=DPI_PNG)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    log.info("Figure saved: %s (%d dpi) | %s", OUT_PNG, DPI_PNG, OUT_PDF)
    plt.close(fig)


# Main
def main():
    log = setup_logging()
    build_figure(log)
    log.info("End | OK")


if __name__ == "__main__":
    main()
