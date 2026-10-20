#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Script 38: Per-seed fit to the training set.

For every (model, seed) run, reports the number of epochs executed, the epoch
selected by the validation loss, and the training accuracy and loss at the end
and at the selected epoch. These figures only exist in the training histories,
not in master_results.json or full_reeval_results.json (which keep only the best
epoch). They document, for instance, that a run can stay in a degenerate regime
on the training set and still obtain a competitive test AUC.

The histories live in two kinds of sources:
  - Script 09 (CNN):          results/metrics/{run_tag}_history.csv
  - Script 11a (CfC family):  results/metrics/{run_name}_training_log.csv
  - Scripts 21 and 22:        only the per-epoch line of {run_tag}_log.txt,
                              "Epoch  12/80 (...) | Train: loss=..., acc=...% |
                              Val: loss=..., acc=...% | ..."
When several sources exist for a run, the one that reproduces the checkpoint's
best epoch (full_reeval_results.json) is used.

Per model, the script reports the range of final training accuracy against the
range of test AUC and their Pearson correlation over the seeds, with and without
degenerate runs. A run is flagged as degenerate when its final training accuracy
does not exceed --degenerate_acc (25 %, chance level with 4 classes). The flag is
descriptive only: no run is excluded from any average. Nothing is trained or
evaluated; the script only reads CSV, log, and JSON files.

Inputs:  results/metrics/**/*.csv, *_log.txt, results/master_results.json,
         results/metrics/full_reeval/full_reeval_results.json,
         results/metrics/reanalysis/val_thresholds.json (Script 30, optional)
Outputs: results/metrics/reanalysis/seed_fit_runs.csv, seed_fit_summary.csv,
         seed_fit_results.json, seed_fit_summary.md; script38_log.txt
Usage:
    python scripts/38_seed_fit.py [--degenerate_acc 25]
    python scripts/38_seed_fit.py --list_only     # only list the histories found
"""

import re
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR_FIXED = Path(__file__).resolve().parent.parent
BASE_DIR = BASE_DIR_FIXED if BASE_DIR_FIXED.exists() else (
    SCRIPTS_DIR.parent if (SCRIPTS_DIR.parent / "results").exists() else SCRIPTS_DIR)
RESULTS_DIR = BASE_DIR / "results"
OUT_DIR = RESULTS_DIR / "metrics" / "reanalysis"
MASTER_JSON = RESULTS_DIR / "master_results.json"
FULL_REEVAL_JSON = RESULTS_DIR / "metrics" / "full_reeval" / "full_reeval_results.json"
VAL_JSON = OUT_DIR / "val_thresholds.json"

SEEDS = [42, 123, 7, 99, 314]

# key -> (run_tag aliases, tokens that exclude a file, display name,
#         key in master_results.json, in full_reeval_results.json, in val_thresholds.json)
MODELS = {
    "CNN":        (["cnn_official", "cnn"], ["aug", "int8", "_ce", "best_model"],
                   "CNN-RespiratoryNet (FP32)", "CNN_FP32", "cnn", "CNN"),
    "CfC":        (["hybrid_official", "cfc_v7_mel", "hybrid"], ["pure", "lstm", "gru"],
                   "CfC-Hybrid", "CfC_Hybrid", "hybrid", "CfC"),
    "MNv3_IN":    (["tl_mobilenet_v3_small_pretrained_official", "mobilenet_v3_small_pretrained"],
                   ["noaug"], "MobileNetV3-Small (ImageNet)", "MobileNetV3_ImageNet", None, "MNv3_IN"),
    "MNv3_SC":    (["tl_mobilenet_v3_small_official", "mobilenet_v3_small_scratch"], ["pretrained"],
                   "MobileNetV3-Small (scratch)", "MobileNetV3_scratch", None, "MNv3_SC"),
    "PANNs":      (["panns_cnn6_official"], ["tune"],
                   "PANNs CNN6 (official)", "PANNs_CNN6_AudioSet", None, "PANNs"),
    "PANNs_tune": (["panns_cnn6_tune_official", "panns_cnn6_tune"], [],
                   "PANNs CNN6 (tune)", "PANNs_CNN6_tune", None, "PANNs_tune"),
    "LSTM":       (["lstm_official"], [], "LSTM-Hybrid", None, None, None),
    "GRU":        (["gru_official"], [], "GRU-Hybrid", None, None, None),
    "CfC_pure":   (["cfcpure_official", "cfcpure"], [], "CfC without mixed memory", None, None, None),
    # Recipe ablation of Script 34 (no metrics in master_results.json); used to
    # verify that its arm A did not converge.
    "CNN_aug":    (["cnn_aug_official"], [], "CNN-RespiratoryNet with augmentation", None, None, None),
    "MNv3_noaug": (["tl_mobilenet_v3_small_pretrained_noaug_official", "mnv3_noaug"], [],
                   "MobileNetV3-ImageNet without augmentation", None, None, None),
}

# Per-epoch line of the logs written by Scripts 09, 21 and 22.
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*/\s*(\d+).*?Train:\s*loss=([0-9.]+),?\s*acc=\s*([0-9.]+)\s*%.*?"
    r"Val:\s*loss=([0-9.]+),?\s*acc=\s*([0-9.]+)\s*%")

logger = logging.getLogger(__name__)


def setup_logging():
    log_file = BASE_DIR / "script38_log.txt"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(log_file, mode="w", encoding="utf-8")],
                        force=True)
    return log_file


def classify(stem):
    """Return (model, seed) for a file name stem, or None (longest matching alias wins)."""
    stem = stem.lower()
    seed = next((x for x in SEEDS if f"seed{x}" in stem), None)
    if seed is None:
        return None
    best = None
    for key, (aliases, excludes, *_rest) in MODELS.items():
        if any(x in stem for x in excludes):
            continue
        for a in aliases:
            if a in stem and (best is None or len(a) > best[1]):
                best = (key, len(a))
    return (best[0], seed) if best else None


def is_history_csv(path):
    """True if the CSV has a validation-loss column."""
    try:
        head = pd.read_csv(path, nrows=1)
    except Exception:
        return False
    return col(head, "val_loss", "valid_loss", "vloss") is not None


def find_sources():
    """
    Collect all candidate sources per (model, seed): CSVs with a validation-loss
    column and per-run logs. The choice among them is made later, by comparing
    the selected epoch with the checkpoint's.
    """
    cand, unmatched = {}, []
    for path in sorted(RESULTS_DIR.rglob("*.csv")):
        hit = classify(path.stem)
        if hit is None:
            if "history" in path.stem.lower() or "training_log" in path.stem.lower():
                unmatched.append(("csv", path))
            continue
        if is_history_csv(path):
            cand.setdefault(hit, []).append(("csv", path))
    for base in {BASE_DIR, SCRIPTS_DIR, BASE_DIR / "logs"}:
        for path in sorted(base.glob("*_log.txt")):
            hit = classify(path.stem)
            if hit is None:
                unmatched.append(("log", path))
            else:
                cand.setdefault(hit, []).append(("log", path))
    return cand, unmatched


def col(df, *names):
    """First column matching any of the names (case-insensitive)."""
    lower = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def as_percent(x):
    """Return accuracy as a percentage, whether given as a fraction or a percentage."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    return float(x) * 100.0 if float(x) <= 1.0 else float(x)


def read_log(path):
    """Rebuild the history from the per-epoch log lines (Scripts 09, 21, 22)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = [(int(m.group(1)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)))
            for m in EPOCH_RE.finditer(text)]
    if not rows:
        raise ValueError(f"{path.name}: no per-epoch lines "
                         f"(probably the log of an --eval_only run)")
    df = pd.DataFrame(rows, columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
    return df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch").reset_index(drop=True)


def read_run(path, kind="csv"):
    """Extract the per-run fit figures from a history (CSV or log)."""
    df = read_log(path) if kind == "log" else pd.read_csv(path)
    c_vl = col(df, "val_loss", "valid_loss", "vloss")
    c_ta = col(df, "train_acc", "train_accuracy", "acc_train")
    c_tl = col(df, "train_loss", "loss", "tloss")
    c_va = col(df, "val_acc", "val_accuracy")
    if c_vl is None:
        raise ValueError(f"{path.name}: validation-loss column not found")
    best_idx = int(np.argmin(df[c_vl].values))
    return {
        "source": kind,
        "history_file": path.name,
        "epochs_run": int(len(df)),
        "best_epoch": best_idx + 1,
        "val_loss_min": float(df[c_vl].values[best_idx]),
        "train_acc_final_pct": as_percent(df[c_ta].values[-1]) if c_ta else float("nan"),
        "train_acc_best_pct": as_percent(df[c_ta].values[best_idx]) if c_ta else float("nan"),
        "train_loss_final": float(df[c_tl].values[-1]) if c_tl else float("nan"),
        "train_loss_best": float(df[c_tl].values[best_idx]) if c_tl else float("nan"),
        "val_acc_best_pct": as_percent(df[c_va].values[best_idx]) if c_va else float("nan"),
    }


def load_reference_metrics():
    """Per-seed test metrics (master), best epoch (full_reeval) and validation AUC (Script 30)."""
    test, best_ep, val_auc = {}, {}, {}
    if MASTER_JSON.exists():
        acc = json.load(open(MASTER_JSON, encoding="utf-8"))["accuracy"]
        for key, (_a, _e, _d, mkey, *_r) in MODELS.items():
            if mkey and mkey in acc:
                v = acc[mkey]
                for i, s in enumerate(v["seeds"]):
                    test[(key, int(s))] = {"test_icbhi": v["icbhi_official"]["values"][i],
                                           "test_auc": v["auc_binary"]["values"][i]}
    else:
        logger.warning(f"  file not found: {MASTER_JSON}")
    if FULL_REEVAL_JSON.exists():
        fr = json.load(open(FULL_REEVAL_JSON, encoding="utf-8"))["per_model_seed"]
        for key, (_a, _e, _d, _m, fkey, *_r) in MODELS.items():
            if fkey and fkey in fr:
                for r in fr[fkey]:
                    best_ep[(key, int(r["seed"]))] = int(r["best_epoch_training"]) + 1
    if VAL_JSON.exists():
        vm = json.load(open(VAL_JSON, encoding="utf-8"))["val_metrics_per_seed"]
        for key, (_a, _e, _d, _m, _f, vkey) in MODELS.items():
            if vkey and vkey in vm:
                for i, s in enumerate(SEEDS):
                    val_auc[(key, s)] = vm[vkey]["val_auc"][i]
    return test, best_ep, val_auc


def main():
    p = argparse.ArgumentParser(description="Per-seed fit to the training set")
    p.add_argument("--degenerate_acc", type=float, default=25.0,
                   help="Training accuracy (%%) at or below which a run is flagged as degenerate")
    p.add_argument("--list_only", action="store_true", help="Only list the histories found")
    args = p.parse_args()

    log_file = setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 78)
    logger.info("SCRIPT 38: Per-seed fit to the training set")
    logger.info("=" * 78)
    logger.info(f"  BASE_DIR: {BASE_DIR}")

    cand, unmatched = find_sources()
    logger.info(f"\n[1/3] Runs with at least one source: {len(cand)}")
    for (key, seed), lst in sorted(cand.items(), key=lambda kv: (list(MODELS).index(kv[0][0]),
                                                                SEEDS.index(kv[0][1]))):
        logger.info(f"  {key:<11} seed {seed:>3}: " + ", ".join(f"[{k}] {pp.name}" for k, pp in lst))
    for kind, path in unmatched:
        logger.info(f"  (unclassified) [{kind}] {path.name}")
    missing = [(k, x) for k in MODELS for x in SEEDS if (k, x) not in cand]
    if missing:
        logger.warning(f"  no source: " + ", ".join(f"{k}/{x}" for k, x in missing))
    if args.list_only:
        logger.info("  --list_only: done.")
        return

    test, best_ep, val_auc = load_reference_metrics()

    logger.info("\n[2/3] Reading and choosing sources")
    logger.info("  Rule: among the sources of a run, use the one that reproduces the epoch")
    logger.info("  selected by the checkpoint (full_reeval_results.json). Without a reference, the CSV.")
    rows, unresolved = [], []
    for (key, seed), lst in sorted(cand.items(), key=lambda kv: (list(MODELS).index(kv[0][0]),
                                                                SEEDS.index(kv[0][1]))):
        ref = best_ep.get((key, seed))
        parsed = []
        for kind, path in lst:
            try:
                r = read_run(path, kind)
                r["candidate"] = f"[{kind}] {path.name}"
                parsed.append(r)
            except Exception as e:
                logger.info(f"  {key}/{seed}: discarding {path.name}: {e}")
        if not parsed:
            unresolved.append((key, seed, "no readable source"))
            continue
        matching = [r for r in parsed if ref is not None and r["best_epoch"] == ref]
        if matching:
            chosen = sorted(matching, key=lambda r: 0 if r["source"] == "csv" else 1)[0]
            status = "matches the checkpoint"
        elif ref is not None:
            chosen = sorted(parsed, key=lambda r: 0 if r["source"] == "csv" else 1)[0]
            status = "does NOT match the checkpoint"
            unresolved.append((key, seed, "no source reproduces the checkpoint epoch: "
                              + ", ".join(f"{r['candidate']} selects {r['best_epoch']}" for r in parsed)
                              + f"; checkpoint {ref}"))
        else:
            chosen = sorted(parsed, key=lambda r: 0 if r["source"] == "csv" else 1)[0]
            status = "no reference"
        r = dict(chosen)
        r.update({"model": key, "display": MODELS[key][2], "seed": seed,
                  "best_epoch_reference": ref if ref is not None else float("nan"),
                  "best_epoch_matches_reference": bool(ref is None or r["best_epoch"] == ref),
                  "reference_status": status,
                  "n_candidates": len(parsed),
                  "candidates": "; ".join(f"{x['candidate']}:ep{x['best_epoch']}" for x in parsed)})
        r.update(test.get((key, seed), {"test_icbhi": float("nan"), "test_auc": float("nan")}))
        r["val_auc"] = val_auc.get((key, seed), float("nan"))
        r["degenerate"] = bool(r["train_acc_final_pct"] <= args.degenerate_acc)
        rows.append(r)
        flag = "" if r["best_epoch_matches_reference"] else f"   <- {status} (checkpoint {ref})"
        logger.info(f"  {key:<11} seed {seed:>3} [{r['source']}]: {r['epochs_run']:>3} ep., "
                    f"selected {r['best_epoch']:>3}, final train acc {r['train_acc_final_pct']:>5.1f} %, "
                    f"test AUC {r['test_auc']:.4f}"
                    + ("   <- degenerate" if r["degenerate"] else "") + flag)
    if unresolved:
        logger.warning("\n  Runs without a verified source:")
        for key, seed, why in unresolved:
            logger.warning(f"    {key}/{seed}: {why}")

    if not rows:
        logger.error("No readable histories. Run with --list_only and check the patterns.")
        return
    df = pd.DataFrame(rows)

    logger.info("\n[3/3] Per-model summary")
    summary = []
    for key in MODELS:
        sub = df[df["model"] == key]
        if sub.empty:
            continue
        s = {"model": key, "display": MODELS[key][2], "n_runs": int(len(sub)),
             "n_degenerate": int(sub["degenerate"].sum()),
             "n_verified": int(sub["best_epoch_matches_reference"].sum()),
             "epochs_min": int(sub["epochs_run"].min()), "epochs_max": int(sub["epochs_run"].max()),
             "best_epoch_min": int(sub["best_epoch"].min()), "best_epoch_max": int(sub["best_epoch"].max()),
             "train_acc_min_pct": float(sub["train_acc_final_pct"].min()),
             "train_acc_max_pct": float(sub["train_acc_final_pct"].max()),
             "test_auc_min": float(sub["test_auc"].min()), "test_auc_max": float(sub["test_auc"].max())}
        for label, d in [("all", sub), ("trained_only", sub[~sub["degenerate"]])]:
            x, y = d["train_acc_final_pct"].values, d["test_auc"].values
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= 3 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                r_, p_ = stats.pearsonr(x[ok], y[ok])
                s[f"pearson_{label}"], s[f"pearson_p_{label}"], s[f"pearson_n_{label}"] = float(r_), float(p_), int(ok.sum())
            else:
                s[f"pearson_{label}"], s[f"pearson_p_{label}"], s[f"pearson_n_{label}"] = float("nan"), float("nan"), int(ok.sum())
        summary.append(s)
        logger.info(f"  {key:<11} train acc {s['train_acc_min_pct']:.1f}-{s['train_acc_max_pct']:.1f} % | "
                    f"test AUC {s['test_auc_min']:.3f}-{s['test_auc_max']:.3f} | "
                    f"Pearson {s['pearson_all']:+.2f} (p {s['pearson_p_all']:.2f}, n={s['pearson_n_all']}) | "
                    f"degenerate {s['n_degenerate']}")

    df.to_csv(OUT_DIR / "seed_fit_runs.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT_DIR / "seed_fit_summary.csv", index=False)
    with open(OUT_DIR / "seed_fit_results.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "degenerate_acc_threshold_pct": args.degenerate_acc,
                   "runs": rows, "summary": summary,
                   "missing": [f"{k}/{s}" for k, s in missing]}, f, indent=2, ensure_ascii=False, default=float)

    fmt = lambda x, n=3: "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{n}f}"
    with open(OUT_DIR / "seed_fit_summary.md", "w", encoding="utf-8") as f:
        f.write("# Per-seed fit to the training set (Script 38)\n\n")
        f.write(f"_{datetime.now().isoformat(timespec='seconds')} · degenerate-run threshold: "
                f"training accuracy <= {args.degenerate_acc:.0f} %_\n\n")
        f.write("## Per run\n\n")
        f.write("| Model | Seed | Source | Verified | Epochs | Selected ep. | Final train acc "
                "| Train acc at selected ep. | Final train loss | Val AUC | Test ICBHI | Test AUC "
                "| Degenerate |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['display']} | {r['seed']} | {r['source']} "
                    f"| {'yes' if r['best_epoch_matches_reference'] else 'NO'} "
                    f"| {r['epochs_run']} | {r['best_epoch']} "
                    f"| {fmt(r['train_acc_final_pct'],1)} % | {fmt(r['train_acc_best_pct'],1)} % "
                    f"| {fmt(r['train_loss_final'],3)} | {fmt(r['val_auc'])} | {fmt(r['test_icbhi'])} "
                    f"| {fmt(r['test_auc'])} | {'yes' if r['degenerate'] else 'no'} |\n")
        f.write("\n## Per model\n\n")
        f.write("| Model | Verified runs | Epochs | Selected ep. | Train acc | Test AUC "
                "| Pearson (5 seeds) | Pearson (without degenerate) | Degenerate |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for s in summary:
            f.write(f"| {s['display']} | {s['n_verified']}/{s['n_runs']} | {s['epochs_min']}-{s['epochs_max']} "
                    f"| {s['best_epoch_min']}-{s['best_epoch_max']} "
                    f"| {fmt(s['train_acc_min_pct'],1)}-{fmt(s['train_acc_max_pct'],1)} % "
                    f"| {fmt(s['test_auc_min'])}-{fmt(s['test_auc_max'])} "
                    f"| {fmt(s['pearson_all'],2)} (p {fmt(s['pearson_p_all'],2)}, n={s['pearson_n_all']}) "
                    f"| {fmt(s['pearson_trained_only'],2)} (p {fmt(s['pearson_p_trained_only'],2)}, "
                    f"n={s['pearson_n_trained_only']}) | {s['n_degenerate']} |\n")
        if missing:
            f.write(f"\n_No source: {', '.join(f'{k}/{s}' for k, s in missing)}._\n")
        if unresolved:
            f.write("\n_Runs whose source does not reproduce the checkpoint epoch:_\n\n")
            for key, seed, why in unresolved:
                f.write(f"- {key}/{seed}: {why}\n")
        f.write("\n_No run is excluded from the reported averages; the flag is descriptive._\n")

    logger.info(f"\nOutputs in {OUT_DIR}")
    logger.info(f"Log: {log_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
