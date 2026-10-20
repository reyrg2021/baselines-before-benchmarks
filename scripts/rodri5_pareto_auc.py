#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
rodri5_pareto_auc.py: Accuracy-efficiency plane on the Raspberry Pi 5 (seven configurations).

X axis: end-to-end latency per cycle at 4 threads (front end + forward, p50, ms), read
from the Script 32 JSON. Y axis: binary Normal-vs-abnormal AUC (mean of 5 seeds) or,
with --metric auc_wd, the post hoc within-device AUC. Bubble area: parameter and buffer
memory (MB), from the same JSON. The front end added is the librosa mel for models that
use the shared log-mel, and the 16->32 kHz resampling for PANNs, whose mel is internal.
The AUC values (Scripts 29, 33 and 35) are listed in AUC_BIN and AUC_WD; update only
those two tables if they are recomputed. Supersedes rodri4_pareto.py.

Inputs:  results/edge_benchmark/pi5_extended/edge_extended_<label>_4h_<stamp>.json (Script 32)
Outputs: rodri5.png, rodri5.pdf, rodri5_pareto_log.txt, rodri5_pareto_data.csv (in --out_dir)
Usage:
    python rodri5_pareto_auc.py \
        --edge_json results/edge_benchmark/pi5_extended/edge_extended_RPi5_performance_v7_ext_4h_20260907_232133.json \
        [--metric {auc,auc_wd}] [--out_dir .] [--dpi 600]
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Data not contained in the Script 32 JSON
# Pooled binary AUC (Script 29; Script 33 for INT8).
AUC_BIN = {
    "cnn": 0.572, "cnn_int8": 0.572, "hybrid": 0.501,
    "mnv3_imagenet": 0.543, "mnv3_scratch": 0.528,
    "panns": 0.546, "panns_tune": 0.516,
}
# Within-device AUC, post hoc analysis (Script 35).
AUC_WD = {
    "cnn": 0.614, "cnn_int8": 0.613, "hybrid": 0.540,
    "mnv3_imagenet": 0.600, "mnv3_scratch": 0.555,
    "panns": 0.575, "panns_tune": 0.538,
}
DISPLAY = {
    "cnn": "CNN-RespNet FP32", "cnn_int8": "CNN-RespNet INT8", "hybrid": "CfC-Hybrid",
    "mnv3_imagenet": "MobileNetV3 (ImageNet)", "mnv3_scratch": "MobileNetV3 (scratch)",
    "panns": "PANNs CNN6", "panns_tune": "PANNs CNN6 (tune)",
}
COLOR = {
    "cnn": "#2e8b57", "cnn_int8": "#1f4e79", "hybrid": "#c0392b",
    "mnv3_imagenet": "#8e44ad", "mnv3_scratch": "#b07aa1",
    "panns": "#e07b39", "panns_tune": "#d9a441",
}
# Models whose front end is resampling, not the shared mel.
FRONTEND_RESAMPLE = {"panns", "panns_tune"}
# Label placement relative to each bubble; the offset is computed from the bubble
# radius so the text never overlaps it.
LABEL_SIDE = {
    "cnn_int8": "above", "cnn": "right", "mnv3_imagenet": "right",
    "mnv3_scratch": "below", "hybrid": "below", "panns": "above",
    "panns_tune": "below",
}
LABEL_PAD_PT = 5.0          # gap between bubble edge and text

logger = logging.getLogger(__name__)


def setup_logging(out_dir):
    log_file = out_dir / "rodri5_pareto_log.txt"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(log_file, mode="w", encoding="utf-8")],
                        force=True)
    return log_file


# Reading the Script 32 JSON
def load_points(edge_json, threads, metric):
    """Return the points (key, end-to-end latency, metric, MB) and the raw JSON."""
    d = json.loads(Path(edge_json).read_text(encoding="utf-8"))
    pre = d["preprocessing"]
    mel_ms = float(pre["mel_16k_librosa"]["p50_ms"])
    res_ms = float(pre["resample_16k_to_32k"]["p50_ms"])
    logger.info(f"  front end: mel {mel_ms:.2f} ms · resampling {res_ms:.2f} ms")
    table = AUC_BIN if metric == "auc" else AUC_WD
    points = []
    for key, m in d["models"].items():
        if key not in table:
            logger.warning(f"  {key} is not in the AUC table; skipped")
            continue
        bt = m["by_threads"].get(str(threads))
        if bt is None:
            raise KeyError(f"{key}: the JSON has no measurement at {threads} threads")
        fwd = float(bt["latency_ms_p50"])
        front = res_ms if key in FRONTEND_RESAMPLE else mel_ms
        points.append({"key": key, "display": DISPLAY[key], "forward_ms": fwd,
                       "frontend_ms": front, "e2e_ms": fwd + front,
                       "metric": float(table[key]), "mem_mb": float(m["model_memory_mb"])})
    if not points:
        raise SystemExit("The JSON contains none of the expected models")
    return sorted(points, key=lambda p: p["e2e_ms"]), d


def pareto_front(points):
    """Non-dominated points: lower latency and higher metric (no strict ties)."""
    front = []
    for p in points:
        dominated = any((q["e2e_ms"] <= p["e2e_ms"] and q["metric"] >= p["metric"] and
                         (q["e2e_ms"] < p["e2e_ms"] or q["metric"] > p["metric"]))
                        for q in points if q is not p)
        p["pareto"] = not dominated
        if not dominated:
            front.append(p)
    return front


# Figure
def draw(points, front, metric, out_dir, dpi):
    ylab = ("Binary AUC (Normal vs. abnormal, 5 seeds)" if metric == "auc"
            else "Within-device AUC (post hoc, 5 seeds)")
    fig, ax = plt.subplots(figsize=(7.16, 3.8))
    ax.axhline(0.5, color="#888888", lw=1.0, ls="--", zorder=1)
    ax.text(0.01, 0.5, "chance", transform=ax.get_yaxis_transform(),
            va="bottom", ha="left", fontsize=8, color="#666666")

    xs = [p["e2e_ms"] for p in front]
    ys = [p["metric"] for p in front]
    if len(front) > 1:
        ax.plot(xs, ys, color="#1f4e79", lw=1.2, ls=":", zorder=2, label="Pareto frontier")
    else:
        ax.scatter([], [], s=90, facecolor="none", edgecolor="black", linewidth=1.6,
                   label="not dominated (thick edge)")

    for i, p in enumerate(points):
        size = 40 + 120 * max(p["mem_mb"], 0.2) ** 0.5
        radius_pt = (size / 3.141592653589793) ** 0.5      # s is area in pt^2
        ax.scatter(p["e2e_ms"], p["metric"], s=size, color=COLOR[p["key"]], alpha=0.60,
                   edgecolor="black", linewidth=1.6 if p["pareto"] else 0.6, zorder=3)
        side = LABEL_SIDE.get(p["key"], "above")
        off = radius_pt + LABEL_PAD_PT
        xy, ha, va = {"above": ((0, off), "center", "bottom"),
                      "below": ((0, -off), "center", "top"),
                      "right": ((off, 0), "left", "center"),
                      "left": ((-off, 0), "right", "center")}[side]
        ax.annotate(f"{p['display']}\n{p['mem_mb']:.2f} MB · {p['e2e_ms']:.0f} ms",
                    (p["e2e_ms"], p["metric"]), textcoords="offset points",
                    xytext=xy, ha=ha, va=va, fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.8))

    ax.set_xscale("log")
    lo = min(p["e2e_ms"] for p in points); hi = max(p["e2e_ms"] for p in points)
    ax.set_xlim(lo * 0.48, hi * 1.45)
    ticks = [t for t in (20, 30, 40, 60, 100, 150, 200, 300, 400) if lo * 0.48 <= t <= hi * 1.45]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("End-to-end latency per cycle on Raspberry Pi 5 — 4 threads, p50 (ms)")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, which="both")
    ax.set_ylim(0.47, max(p["metric"] for p in points) + 0.035)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    png, pdf = out_dir / "rodri5.png", out_dir / "rodri5.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


# Main
def main():
    ap = argparse.ArgumentParser(description="AUC vs end-to-end latency figure")
    ap.add_argument("--edge_json", required=True, help="Script 32 JSON (the 4-thread run)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--metric", choices=["auc", "auc_wd"], default="auc")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(out_dir)
    logger.info("=" * 78)
    logger.info("AUC vs end-to-end latency plane (Pi 5)")
    logger.info("=" * 78)
    logger.info(f"  JSON: {args.edge_json}  ({args.threads} threads, metric {args.metric})")

    points, raw = load_points(args.edge_json, args.threads, args.metric)
    front = pareto_front(points)
    for p in points:
        logger.info(f"  {p['display']:<24} forward {p['forward_ms']:7.2f} + front {p['frontend_ms']:5.2f} "
                    f"= {p['e2e_ms']:7.2f} ms | {args.metric} {p['metric']:.3f} | {p['mem_mb']:6.2f} MB"
                    f"{'  <- Pareto' if p['pareto'] else ''}")
    logger.info(f"  Non-dominated: {', '.join(p['display'] for p in front)}")

    with open(out_dir / "rodri5_pareto_data.csv", "w", encoding="utf-8") as f:
        f.write("key,display,forward_ms,frontend_ms,e2e_ms,metric,metric_name,mem_mb,pareto\n")
        for p in points:
            f.write(f"{p['key']},{p['display']},{p['forward_ms']:.3f},{p['frontend_ms']:.3f},"
                    f"{p['e2e_ms']:.3f},{p['metric']:.4f},{args.metric},{p['mem_mb']:.3f},{p['pareto']}\n")

    png, pdf = draw(points, front, args.metric, out_dir, args.dpi)
    logger.info(f"  Figure: {png}  and  {pdf}")
    logger.info(f"  Data:   {out_dir / 'rodri5_pareto_data.csv'}")
    logger.info(f"  Log:    {log_file}")
    logger.info(f"  Benchmark system: {raw['system'].get('rpi_model', 'n/a')} · "
                f"torch {raw['system'].get('torch_version', 'n/a')} · {datetime.now():%Y-%m-%d}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
