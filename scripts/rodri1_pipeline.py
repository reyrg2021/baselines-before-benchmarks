# -*- coding: utf-8 -*-
"""
Figure 1: Preprocessing and feature-extraction pipeline diagram.

Draws the five stages that take a raw ICBHI 2017 recording to the model input
tensor: recording (920 WAV, 126 subjects), conditioning (mono, 16 kHz,
50-2000 Hz band), per-cycle segmentation (6,729 cycles, 4 classes), 8 s crop
with normalized log-mel spectrogram (128 bands, FFT 2048, hop 512, [0, 1]),
and the 1 x 128 x 251 input tensor.

Outputs: rodri1.png (600 dpi), rodri1.pdf, rodri1_pipeline_log.txt (in OUT_DIR)
Usage:
    python scripts/rodri1_pipeline.py
"""

import os
import sys
import logging
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.environ.get("OUT_DIR", BASE_DIR)

OUT_PNG = os.path.join(OUT_DIR, "rodri1.png")
OUT_PDF = os.path.join(OUT_DIR, "rodri1.pdf")
LOG_TXT = os.path.join(OUT_DIR, "rodri1_pipeline_log.txt")

DPI_PNG = 600   # PNG resolution for submission

# Each stage: (title, subtitle, fill colour, edge colour)
STAGES = [
    ("ICBHI 2017 recordings",        "920 WAV \u00b7 126 subjects",                       "#dbe9f6", "#1f4e79"),
    ("Conditioning",           "mono \u00b7 16 kHz \u00b7 50\u20132000 Hz band",  "#dbe9f6", "#1f4e79"),
    ("Cycle segmentation",      "annotations \u2192 6,729 cycles \u00b7 4 classes",  "#dbe9f6", "#1f4e79"),
    ("8 s crop + log-mel + norm.","128 bands \u00b7 FFT 2048 \u00b7 hop 512 \u00b7 [0,1]", "#dbe9f6", "#1f4e79"),
    ("Input tensor",           "1 \u00d7 128 \u00d7 251 \u2192 to the models",     "#cfe8dc", "#2e8b57"),
]


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
    return log


def build_figure(log):
    """Draw the stage boxes top to bottom, connected by arrows, and save PNG/PDF."""
    n = len(STAGES)
    box_w, box_h = 0.86, 0.66      # box width/height (data coordinates)
    gap = 0.34                     # vertical gap between boxes
    step_info = box_h + gap

    fig, ax = plt.subplots(figsize=(4.4, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n * step_info + 0.1)
    ax.axis("off")

    cx = 0.5
    centers_y = []
    for k, (title_text, sub, fill, edge) in enumerate(STAGES):
        cy = (n * step_info) - (k * step_info) - box_h / 2
        centers_y.append(cy)
        box = FancyBboxPatch((cx - box_w / 2, cy - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2)
        ax.add_patch(box)
        ax.text(cx, cy + 0.11, title_text, ha="center", va="center",
                fontsize=11, fontweight="bold", color=edge, zorder=3)
        ax.text(cx, cy - 0.13, sub, ha="center", va="center",
                fontsize=8.6, color="#222222", zorder=3)
        log.info("  Stage %d: %s", k + 1, title_text)

    for k in range(n - 1):
        y0 = centers_y[k] - box_h / 2
        y1 = centers_y[k + 1] + box_h / 2
        ax.add_patch(FancyArrowPatch((cx, y0), (cx, y1),
                                     arrowstyle="-|>", mutation_scale=16,
                                     linewidth=1.6, color="#555555", zorder=1))

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=DPI_PNG)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    log.info("Figure saved: %s (%d dpi) | %s", OUT_PNG, DPI_PNG, OUT_PDF)
    plt.close(fig)


def main():
    log = setup_logging()
    build_figure(log)
    log.info("End | OK")


if __name__ == "__main__":
    main()
