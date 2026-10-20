#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 36: Paired contrasts for the pretraining and training-recipe comparisons.

Scripts 29 and 35 only computed CNN - X contrasts. The question "does pretraining help?"
also needs contrasts that do not involve the CNN (MNv3-ImageNet vs MNv3-scratch, PANNs
official vs tune, vision vs audio pretraining) and a comparison of the recipe ablation of
script 34 (MobileNetV3 without augmentation or mixup) against the CNN. Two of these
differences pass the per-seed paired t-test, so they require the patient-level bootstrap
to be called "supported" (t p < 0.05 and 95% bootstrap CI excluding 0).

For each pair and metric (official ICBHI, pooled binary AUC, post hoc within-device AUC)
it computes the per-seed paired t-test, TOST with delta = 0.02 (90% CI), and a
patient-level bootstrap 95% CI and p-value using the same replicates as scripts 29 and 35
(seed 2026, B = 2000). Only saved test arrays are used (no training or inference). As an
integrity control it reproduces the CNN - MNv3_IN contrasts of script 29 and aborts if
they differ, and checks the MNv3_noaug means against recipe_ablation_summary.md.

Inputs:  saved test arrays loaded through scripts 29 and 35 (must be in scripts/),
         master_results.json
Outputs: results/metrics/reanalysis/pretrain_contrasts_results.json,
         results/metrics/reanalysis/pretrain_contrasts.csv,
         results/metrics/reanalysis/pretrain_contrasts_absolute.csv,
         results/metrics/reanalysis/pretrain_contrasts_summary.md, script36_log.txt
Usage:
    python scripts/36_pretraining_contrasts.py [--n_boot 2000] [--boot_seed 2026] [--delta 0.02]
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


SCRIPTS_DIR = Path(__file__).resolve().parent


def import_script(name, filename):
    """Import a numbered script as a module."""
    path = SCRIPTS_DIR / filename
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s35 = import_script("script35", "35_device_confound_auc.py")
s29 = s35.s29
BASE_DIR = s29.BASE_DIR
OUT_DIR = s29.OUT_DIR
SEEDS = s29.SEEDS

# Arm B of script 34, added as an optional model (not present in master_results.json)
s29.MODELS["MNv3_noaug"] = {
    "display": "MobileNetV3-Small (ImageNet, no augmentation or mixup)",
    "prefix": "tl_mobilenet_v3_small_pretrained_noaug_official",
    "master_key": "MobileNetV3_ImageNet_noaug",
    "optional": True,
}

PAIRS = [
    ("P1", "MNv3_IN", "MNv3_SC", "Effect of ImageNet pretraining"),
    ("P2", "PANNs", "PANNs_tune", "Effect of the PANNs adaptation regime"),
    ("P3", "MNv3_IN", "PANNs", "Vision vs audio pretraining"),
    ("P4", "MNv3_IN", "MNv3_noaug", "Contribution of augmentation to MobileNetV3"),
    ("P5", "CNN", "MNv3_noaug", "CNN vs MobileNetV3 without augmentation"),
]
CONTROL_PAIR = ("CNN", "MNv3_IN")
CONTROL_EXPECTED = {  # script 29 (reanalysis_summary.md)
    "icbhi": {"diff": -0.0027, "ci90_lo": -0.0202, "ci90_hi": 0.0148},
    "auc": {"diff": 0.0286, "ci90_lo": -0.0156, "ci90_hi": 0.0728},
}
NOAUG_EXPECTED = {"icbhi": 0.439, "auc": 0.512}  # recipe_ablation_summary.md
METRICS = ["icbhi", "auc", "auc_wd"]

logger = logging.getLogger(__name__)


def setup_logging():
    """Log to stdout and to script36_log.txt in BASE_DIR."""
    log_file = BASE_DIR / "script36_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, mode="w", encoding="utf-8")],
        force=True,
    )
    s29.logger = logging.getLogger("script29")
    s35.logger = logging.getLogger("script35")
    return log_file


def run_metrics(y_true, y_bin, dev_codes, n_dev, run, idx):
    """Official ICBHI, pooled AUC and within-device AUC of one run on the cycles in idx."""
    yt, yb, dc = y_true[idx], y_bin[idx], dev_codes[idx]
    met = s29.all_metrics(yt, run["y_pred"][idx], run["p_abn"][idx])
    dec = s35.auc_decomposition(yb, run["p_abn"][idx], dc, n_dev)
    return {"icbhi": met["icbhi"], "auc": met["auc"], "auc_wd": dec["within"],
            "se_off": met["se_off"], "sp_off": met["sp_off"]}


def sample_means(models, runs, y_true, y_bin, dev_codes, n_dev, idx):
    """Mean over seeds of each metric, per model, on the sample idx."""
    out = {}
    for key in models:
        acc = {m: 0.0 for m in METRICS}
        for seed in SEEDS:
            r = run_metrics(y_true, y_bin, dev_codes, n_dev, runs[key][seed], idx)
            for m in METRICS:
                acc[m] += r[m]
        out[key] = {m: v / len(SEEDS) for m, v in acc.items()}
    return out


def main():
    p = argparse.ArgumentParser(description="Paired contrasts for pretraining and training recipe")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--boot_seed", type=int, default=2026)
    p.add_argument("--delta", type=float, default=0.02,
                   help="Preregistered TOST margin (do not change)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--skip_verify", action="store_true",
                   help="Skip the cross-check against master_results.json (testing only)")
    args = p.parse_args()

    log_file = setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 78)
    logger.info("SCRIPT 36 - Paired contrasts for pretraining and training recipe")
    logger.info("=" * 78)

    logger.info("\n[1/4] Loading, alignment and integrity checks")
    df = s29.load_split()
    runs = s29.load_runs(df)
    if args.skip_verify:
        logger.warning("  --skip_verify: figures are NOT citable")
    else:
        s29.verify_against_master(df, runs)
    available = set(s29.MODELS.keys())
    pairs = []
    for pid, a, b, desc in PAIRS:
        if a in available and b in available:
            pairs.append((pid, a, b, desc))
        else:
            logger.warning(f"  {pid} skipped: missing arrays for {a if a not in available else b}")
    models = sorted({k for _, a, b, _ in pairs for k in (a, b)} | set(CONTROL_PAIR),
                    key=list(s29.MODELS.keys()).index)
    y_true = df["y_true"].values
    y_bin = (y_true != 0).astype(int)
    dev_names = sorted(df["recording_equipment"].unique())
    dev_codes = df["recording_equipment"].map({e: i for i, e in enumerate(dev_names)}).values.astype(np.int64)
    n_dev = len(dev_names)

    logger.info("\n[2/4] Per-seed metrics (full test set)")
    full = np.arange(len(df))
    per_seed = {key: {seed: run_metrics(y_true, y_bin, dev_codes, n_dev, runs[key][seed], full)
                      for seed in SEEDS} for key in models}
    abs_rows = []
    for key in models:
        row = {"model": key, "display": s29.MODELS[key]["display"]}
        for m in METRICS + ["se_off", "sp_off"]:
            vals = np.array([per_seed[key][s][m] for s in SEEDS])
            row[f"{m}_mean"], row[f"{m}_sd"] = float(vals.mean()), float(vals.std(ddof=1))
        abs_rows.append(row)
        logger.info(f"  {key:<11} ICBHI {row['icbhi_mean']:.4f} ± {row['icbhi_sd']:.4f}  "
                    f"AUC {row['auc_mean']:.4f} ± {row['auc_sd']:.4f}  "
                    f"AUC_wd {row['auc_wd_mean']:.4f} ± {row['auc_wd_sd']:.4f}  "
                    f"Se {row['se_off_mean']:.3f}  Sp {row['sp_off_mean']:.3f}")

    if "MNv3_noaug" in models:
        r = next(x for x in abs_rows if x["model"] == "MNv3_noaug")
        for m, ref in NOAUG_EXPECTED.items():
            if abs(r[f"{m}_mean"] - ref) > 5e-4:
                logger.warning(f"  MNv3_noaug {m} {r[f'{m}_mean']:.4f} differs from "
                               f"recipe_ablation_summary.md ({ref}); check the script 34 arrays")
            else:
                logger.info(f"  OK: MNv3_noaug {m} matches script 34 ({ref})")

    # Per-seed paired t-test and TOST (function from script 29)
    def seed_test(a, b, m):
        va = np.array([per_seed[a][s][m] for s in SEEDS])
        vb = np.array([per_seed[b][s][m] for s in SEEDS])
        return s29.tost_paired(va, vb, args.delta, args.alpha)

    ctrl = {m: seed_test(*CONTROL_PAIR, m) for m in ["icbhi", "auc"]}
    for m, exp in (CONTROL_EXPECTED.items() if not args.skip_verify else []):
        got = ctrl[m]
        ok = (round(got["diff_mean"], 4) == exp["diff"] and round(got["ci90_lo"], 4) == exp["ci90_lo"]
              and round(got["ci90_hi"], 4) == exp["ci90_hi"])
        if not ok:
            raise AssertionError(f"Control CNN − MNv3_IN {m}: {got['diff_mean']:+.4f} "
                                 f"[{got['ci90_lo']:+.4f}, {got['ci90_hi']:+.4f}] does not reproduce script 29")
    if args.skip_verify:
        logger.warning("  --skip_verify: CNN − MNv3_IN control skipped")
    else:
        logger.info("  OK: control CNN − MNv3_IN reproduces script 29 (ICBHI and AUC)")

    # Patient-level bootstrap with the same replicates as scripts 29 and 35
    logger.info(f"\n[3/4] Patient-level bootstrap (B = {args.n_boot}, seed {args.boot_seed})")
    rng = np.random.default_rng(args.boot_seed)
    patients = df["patient_id"].values
    uniq = np.unique(patients)
    pat_idx = {pp: np.where(patients == pp)[0] for pp in uniq}
    boot = {key: {m: np.empty(args.n_boot) for m in METRICS} for key in models}
    for b in range(args.n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pat_idx[pp] for pp in draw])
        res = sample_means(models, runs, y_true, y_bin, dev_codes, n_dev, idx)
        for key in models:
            for m in METRICS:
                boot[key][m][b] = res[key][m]
        if (b + 1) % 250 == 0:
            logger.info(f"    replicate {b + 1}/{args.n_boot}")

    for row in abs_rows:
        for m in METRICS:
            row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = s35.ci(boot[row["model"]][m])

    logger.info("\n[4/4] Paired contrasts")
    rows = []
    for pid, a, b, desc in pairs:
        for m in METRICS:
            t = seed_test(a, b, m)
            d = boot[a][m] - boot[b][m]
            lo, hi = s35.ci(d)
            pb = s35.p_boot(d)
            supported = bool(t["p_ttest_two_sided"] < args.alpha and (lo > 0 or hi < 0))
            rows.append({"pair": pid, "a": a, "b": b, "description": desc, "metric": m,
                         "diff_mean": t["diff_mean"], "ci90_lo": t["ci90_lo"], "ci90_hi": t["ci90_hi"],
                         "p_t": t["p_ttest_two_sided"], "p_tost": t["p_tost"], "equivalent": t["equivalent"],
                         "boot_ci_lo": lo, "boot_ci_hi": hi, "p_boot": pb, "supported": supported})
            logger.info(f"  {pid} {a} − {b:<10} {m:<6} Δ {t['diff_mean']:+.4f} "
                        f"CI90 [{t['ci90_lo']:+.4f}, {t['ci90_hi']:+.4f}] t p {t['p_ttest_two_sided']:.3f} "
                        f"TOST p {t['p_tost']:.3f} | boot [{lo:+.4f}, {hi:+.4f}] p {pb:.3f} "
                        f"| supported: {'YES' if supported else 'no'}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "pretrain_contrasts.csv", index=False)
    pd.DataFrame(abs_rows).to_csv(OUT_DIR / "pretrain_contrasts_absolute.csv", index=False)
    with open(OUT_DIR / "pretrain_contrasts_results.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "n_boot": args.n_boot, "boot_seed": args.boot_seed,
                   "delta": args.delta, "alpha": args.alpha,
                   "absolute": abs_rows, "contrasts": rows,
                   "per_seed": {k: {str(s): v for s, v in d.items()} for k, d in per_seed.items()}},
                  f, indent=2, ensure_ascii=False)

    fmt = lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001"
    with open(OUT_DIR / "pretrain_contrasts_summary.md", "w", encoding="utf-8") as f:
        f.write("# Pretraining and training-recipe contrasts (Script 36)\n\n")
        f.write(f"_{datetime.now().isoformat(timespec='seconds')} · B = {args.n_boot} · "
                f"seed {args.boot_seed} · delta = {args.delta}_\n\n")
        f.write("## Absolute metrics (mean ± SD over seeds; patient-level 95% CI)\n\n")
        f.write("| Model | ICBHI [CI] | AUC [CI] | Within-device AUC [CI] | Off. Se | Off. Sp |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in abs_rows:
            f.write(f"| {r['display']} "
                    f"| {r['icbhi_mean']:.3f} ± {r['icbhi_sd']:.3f} [{r['icbhi_ci_lo']:.3f}, {r['icbhi_ci_hi']:.3f}] "
                    f"| {r['auc_mean']:.3f} ± {r['auc_sd']:.3f} [{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}] "
                    f"| {r['auc_wd_mean']:.3f} ± {r['auc_wd_sd']:.3f} [{r['auc_wd_ci_lo']:.3f}, {r['auc_wd_ci_hi']:.3f}] "
                    f"| {r['se_off_mean']:.3f} | {r['sp_off_mean']:.3f} |\n")
        f.write("\n## Paired contrasts (A − B)\n\n")
        f.write("| Pair | A − B | Metric | Δ | 90% CI (seeds) | t p | TOST p | 95% CI (patients) | boot p | Supported |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['pair']} | {r['a']} − {r['b']} | {r['metric']} | {r['diff_mean']:+.4f} "
                    f"| [{r['ci90_lo']:+.4f}, {r['ci90_hi']:+.4f}] | {fmt(r['p_t'])} | {fmt(r['p_tost'])} "
                    f"| [{r['boot_ci_lo']:+.4f}, {r['boot_ci_hi']:+.4f}] | {fmt(r['p_boot'])} "
                    f"| {'yes' if r['supported'] else 'no'} |\n")
        f.write("\n_Within-device AUC: post hoc analysis. Supported = t p < 0.05 and 95% bootstrap CI excluding 0._\n")

    logger.info(f"\nOutputs in {OUT_DIR}")
    logger.info(f"Log: {log_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
