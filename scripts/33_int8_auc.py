#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 33: Binary AUC and per-cycle probabilities of the INT8 CNN.

Script 23 measured the INT8 model only with the official ICBHI score (argmax), so the
INT8 row had no binary AUC (Normal vs abnormal) and the AUC Pareto frontier could not
include the deployed point (15 ms, 0.58 MB). Two evaluations are produced:
  A. Deployed artifact: models/checkpoints/cnn_int8_scripted.pt (the file run on the
     Pi, exported from cnn_official_seed42), evaluated on test.
  B. Per seed: each cnn_official_seed{S}.pth is re-quantized with Script 23's
     quantize_int8_static() (full-train MinMax calibration, final Linear kept in FP32
     because quantizing it collapses sensitivity), giving mean ± sd over 5 seeds; the
     ICBHI of each seed is cross-checked against cnn_int8_quant_report.json.
On x86 the INT8 kernels run with fbgemm or qnnpack; probabilities may differ by ~1e-3
from the Pi (ARM, qnnpack) without changing argmax.

Inputs:  models/checkpoints/cnn_int8_scripted.pt, models/checkpoints/cnn_official_seed{S}.pth,
         data/splits/strategy1_official/test_cycles.csv,
         results/quantization/cnn_int8_quant_report.json
Outputs: results/metrics/reanalysis/int8_auc.json (int8_auc_qlast.json with --quantize_last),
         results/metrics/full_reeval/arrays/cnn_int8_seed{S}_probs.npy, _labels_preds.npz,
         results/metrics/full_reeval/arrays/cnn_int8_scripted_probs.npy, _labels_preds.npz,
         script33_log.txt (in BASE_DIR)
Usage:
    python 33_int8_auc.py                    # A + B
    python 33_int8_auc.py --only_scripted    # A only
    python 33_int8_auc.py --quantize_last    # sensitivity: classifier also INT8 (Se collapses)
"""

import sys
import json
import argparse
import logging
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR_FIXED = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = BASE_DIR_FIXED if BASE_DIR_FIXED.exists() else (
    SCRIPTS_DIR.parent if (SCRIPTS_DIR.parent / "models").exists() else SCRIPTS_DIR)
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
ARRAYS_DIR = BASE_DIR / "results" / "metrics" / "full_reeval" / "arrays"
OUT_DIR = BASE_DIR / "results" / "metrics" / "reanalysis"
SPLIT_DIR = BASE_DIR / "data" / "splits" / "strategy1_official"
QUANT_REPORT = BASE_DIR / "results" / "quantization" / "cnn_int8_quant_report.json"
SEEDS = [42, 123, 7, 99, 314]
CLASS_MAP = {"Normal": 0, "Crackle": 1, "Wheeze": 2, "Both": 3}

logger = logging.getLogger(__name__)


def import_script(name, filename):
    """Import a numbered script from SCRIPTS_DIR as a module."""
    path = SCRIPTS_DIR / filename
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def icbhi_official(y, p):
    """Official ICBHI score (Se + Sp) / 2; returns (score, Se, Sp)."""
    cm = np.zeros((4, 4))
    for t, q in zip(y, p):
        cm[t, q] += 1
    sp = cm[0, 0] / cm[0].sum(); se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / cm[1:].sum()
    return (se + sp) / 2, se, sp


def auc_bin(y, probs32):
    """Binary Normal-vs-abnormal AUC; p_abnormal is formed in float32 so ties match the stored arrays."""
    p_abn = (np.float32(1.0) - probs32[:, 0]).astype(np.float64)
    yb = (y != 0).astype(int); pos = yb == 1; n_pos = pos.sum(); n_neg = len(yb) - n_pos
    r = stats.rankdata(p_abn)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def infer(model, loader):
    """Return (labels, float32 softmax probabilities) over a loader."""
    import torch
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(torch.softmax(model(x).float(), 1).cpu().numpy().astype(np.float32))
            ys.append(np.asarray(y))
    return np.concatenate(ys).astype(int), np.concatenate(ps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only_scripted", action="store_true")
    p.add_argument("--quantize_last", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(BASE_DIR / "script33_log.txt", mode="a", encoding="utf-8")], force=True)
    import torch
    logger.info("SCRIPT 33 - AUC of the INT8 CNN")
    s23 = import_script("script23", "23_quantize_cnn_int8.py")
    s23.logger = logging.getLogger("script23")
    data = s23.get_dataloaders(split_strategy="strategy1_official", batch_size=32)
    y_csv = pd.read_csv(SPLIT_DIR / "test_cycles.csv")["cycle_class"].map(CLASS_MAP).values
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.load(open(QUANT_REPORT)) if QUANT_REPORT.exists() else None
    rep_by_ckpt = {r["checkpoint"]: r for r in report["per_seed"]} if report else {}
    out = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "quantized_engine": torch.backends.quantized.engine, "quantize_last": args.quantize_last}

    # A: deployed artifact
    pt = CHECKPOINTS_DIR / "cnn_int8_scripted.pt"
    if pt.exists():
        try:
            if "qnnpack" in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = "qnnpack"
        except Exception:
            pass
        m = torch.jit.load(str(pt), map_location="cpu"); m.eval()
        y, probs = infer(m, data["test_loader"])
        assert np.array_equal(y, y_csv), "test set not aligned"
        s, se, sp = icbhi_official(y, probs.argmax(1)); a = auc_bin(y, probs)
        np.save(ARRAYS_DIR / "cnn_int8_scripted_probs.npy", probs)
        np.savez(ARRAYS_DIR / "cnn_int8_scripted_labels_preds.npz", y_true=y, y_pred=probs.argmax(1))
        out["scripted"] = {"file": pt.name, "engine": torch.backends.quantized.engine,
                           "icbhi": s, "se_off": se, "sp_off": sp, "auc": a}
        logger.info(f"  [A] {pt.name} ({torch.backends.quantized.engine}): ICBHI {s:.4f} (Se {se:.3f}/Sp {sp:.3f})  AUC {a:.4f}")
    else:
        logger.warning(f"  not found: {pt}")

    # B: per seed
    if not args.only_scripted:
        rows = []
        for seed in SEEDS:
            ckpt = CHECKPOINTS_DIR / f"cnn_official_seed{seed}.pth"
            fp32 = s23.load_fp32_cnn(ckpt, "cpu")
            y32, p32 = infer(fp32, data["test_loader"])
            int8 = s23.quantize_int8_static(fp32, data["train_loader"], "cpu", n_calib_batches=0,
                                            observer="minmax", quantize_last=args.quantize_last)
            y, probs = infer(int8, data["test_loader"])
            assert np.array_equal(y, y_csv), "test set not aligned"
            s, se, sp = icbhi_official(y, probs.argmax(1)); a = auc_bin(y, probs)
            s32 = icbhi_official(y32, p32.argmax(1))[0]; a32 = auc_bin(y32, p32)
            ref = rep_by_ckpt.get(ckpt.name, {}).get("int8_icbhi")
            tag = "_qlast" if args.quantize_last else ""
            np.save(ARRAYS_DIR / f"cnn_int8{tag}_seed{seed}_probs.npy", probs)
            np.savez(ARRAYS_DIR / f"cnn_int8{tag}_seed{seed}_labels_preds.npz", y_true=y, y_pred=probs.argmax(1))
            rows.append({"seed": seed, "fp32_icbhi": s32, "fp32_auc": a32, "int8_icbhi": s, "int8_se": se,
                         "int8_sp": sp, "int8_auc": a, "delta_icbhi": s - s32, "delta_auc": a - a32,
                         "report_int8_icbhi": ref, "delta_vs_report": (s - ref) if ref is not None else None})
            logger.info(f"  [B] seed {seed}: FP32 ICBHI {s32:.4f} AUC {a32:.4f} | INT8 ICBHI {s:.4f} AUC {a:.4f} "
                        f"(ΔAUC {a - a32:+.4f})" + (f" | vs Script 23 report: {s - ref:+.4f}" if ref is not None else ""))
        df = pd.DataFrame(rows)
        agg = {k: {"mean": float(df[k].mean()), "sd": float(df[k].std(ddof=1))}
               for k in ["fp32_icbhi", "fp32_auc", "int8_icbhi", "int8_auc", "delta_icbhi", "delta_auc"]}
        out["per_seed"] = rows; out["aggregated"] = agg
        logger.info(f"  INT8 5 seeds: ICBHI {agg['int8_icbhi']['mean']:.3f} ± {agg['int8_icbhi']['sd']:.3f}  "
                    f"AUC {agg['int8_auc']['mean']:.3f} ± {agg['int8_auc']['sd']:.3f}  "
                    f"ΔAUC vs FP32 {agg['delta_auc']['mean']:+.4f} ± {agg['delta_auc']['sd']:.4f}")
    suffix = "_qlast" if args.quantize_last else ""
    with open(OUT_DIR / f"int8_auc{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Done.")


if __name__ == "__main__":
    main()
