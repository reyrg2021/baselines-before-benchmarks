#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 35: Recording-device confound in the binary (Normal vs Abnormal) AUC.

The device-only prior baseline of Script 29 (AUC 0.494) depends on a single device:
Litt3200 has only 13 training cycles, and its train-set abnormal rate ranks it above
AKG C417L while the order reverses in test. Giving Litt3200 the global train prior
(AUC 0.581), or scoring "is AKG C417L?" (AUC 0.582), the microphone identity alone
reaches the pooled AUC of the models (CNN 0.572), because half of the (abnormal, normal)
test pairs mix two devices and in those pairs the device is highly informative.
This script quantifies device-only baselines and splits each model's AUC into the part
that can and cannot come from device identity, with patient-level bootstrap CIs. It uses
only saved arrays (no training or inference):
  1. Device-only baselines (constant score per device): B1 train prior, B2 train prior
     with devices under --min_train training cycles shrunk to the global prior, B3
     majority-device indicator, B4 test prevalence per device (oracle upper bound only).
  2. Exact Mann-Whitney decomposition U_total = U_within + U_between: AUC_within uses only
     same-device pairs (device-stratified AUC, cannot exploit device identity),
     AUC_between only cross-device pairs, AUC_devmean replaces each score by the model's
     mean score on its device.
  3. Paired patient-bootstrap contrasts and, on AUC_within, per-seed paired t-test plus
     TOST with the pre-registered margin of Script 29 (delta = 0.02).
Linear baselines of Script 30 are decomposed too when their scores exist, since mel-band
means are the spectral signature of a microphone. The decomposition is a post hoc
analysis, motivated by an audio-free baseline, and should be reported as such.

Inputs:  outputs of Script 29 (test split, saved probabilities, master_results.json),
         results/metrics/reanalysis/linear_{logreg,linsvm}_selauc_test_pabn.npy (optional)
Outputs: results/metrics/reanalysis/device_confound_{results.json,baselines.csv,
         decomposition.csv,contrasts.csv,tost_within.csv,summary.md}, script35_log.txt
Usage:
    python 35_device_confound_auc.py [--min_train 50] [--n_boot 2000]
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


# Paths and loading/alignment/integrity checks are reused from Script 29
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


s29 = import_script("script29", "29_reanalysis_test_probs.py")
BASE_DIR = s29.BASE_DIR
OUT_DIR = s29.OUT_DIR
TRAIN_CSV = s29.TRAIN_CSV
SEEDS = s29.SEEDS
REFERENCE = s29.REFERENCE

LINEAR_FILES = {
    "logreg": "linear_logreg_selauc_test_pabn.npy",
    "linsvm": "linear_linsvm_selauc_test_pabn.npy",
}
LINEAR_DISPLAY = {"logreg": "Logistic regression (C selected by AUC)",
                  "linsvm": "Linear SVM (C selected by AUC)"}

logger = logging.getLogger(__name__)


def setup_logging():
    """Log to stdout and to script35_log.txt in BASE_DIR."""
    log_file = BASE_DIR / "script35_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, mode="w", encoding="utf-8")],
        force=True,
    )
    s29.logger = logging.getLogger("script29")
    return log_file


def u_statistic(y_bin, scores):
    """
    Mann-Whitney U over (abnormal, normal) pairs: abnormal wins + 0.5 per tie.
    Returns (U, number of pairs); AUC = U / pairs.
    """
    pos = y_bin == 1
    n_pos = int(pos.sum())
    n_neg = int(len(y_bin) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0
    ranks = stats.rankdata(scores)
    return float(ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0), n_pos * n_neg


def auc_decomposition(y_bin, scores, dev_codes, n_dev):
    """
    Split the pooled AUC into same-device and cross-device pairs.
    Exact identity: U_total = sum_d U_d + U_between.
    Returns a dict with pooled, within, between and the fraction of within-device pairs.
    """
    u_tot, p_tot = u_statistic(y_bin, scores)
    u_w, p_w = 0.0, 0
    for d in range(n_dev):
        m = dev_codes == d
        if m.any():
            u, p = u_statistic(y_bin[m], scores[m])
            u_w += u
            p_w += p
    p_b, u_b = p_tot - p_w, u_tot - u_w
    nan = float("nan")
    return {"pooled": u_tot / p_tot if p_tot else nan,
            "within": u_w / p_w if p_w else nan,
            "between": u_b / p_b if p_b else nan,
            "frac_within": p_w / p_tot if p_tot else nan}


def device_mean_scores(scores, dev_codes, n_dev):
    """Replace each score by the mean score of its device."""
    sums = np.bincount(dev_codes, weights=scores, minlength=n_dev)
    counts = np.bincount(dev_codes, minlength=n_dev)
    means = np.divide(sums, counts, out=np.zeros(n_dev), where=counts > 0)
    return means[dev_codes]


def device_baselines(df, dev_names, min_train):
    """
    Constant per-device scores of baselines B1-B4 for every test cycle.
    All are estimated on TRAIN except B4 (oracle, test prevalence).
    """
    tr = pd.read_csv(TRAIN_CSV)
    tr["y"] = tr["cycle_class"].map(s29.CLASS_MAP)
    g_prior = float((tr["y"] != 0).mean())
    n_by_dev = tr["recording_equipment"].value_counts()
    prior_by_dev = tr.groupby("recording_equipment")["y"].apply(lambda s: float((s != 0).mean()))
    major_dev = n_by_dev.idxmax()

    logger.info(f"  Global train prior P(abnormal) = {g_prior:.4f}")
    for eq in n_by_dev.index:
        logger.info(f"    train {eq:<10} n={int(n_by_dev[eq]):>5}  P(abnormal)={prior_by_dev[eq]:.4f}")
    logger.info(f"  Majority device in train: {major_dev}")

    te_prev = df.groupby("recording_equipment")["y_true"].apply(lambda s: float((s != 0).mean()))
    eq_test = df["recording_equipment"].values

    b1 = {e: float(prior_by_dev.get(e, g_prior)) for e in dev_names}
    b2 = {e: (float(prior_by_dev[e]) if e in prior_by_dev and n_by_dev[e] >= min_train
              else g_prior) for e in dev_names}
    b3 = {e: 1.0 if e == major_dev else 0.0 for e in dev_names}
    b4 = {e: float(te_prev[e]) for e in dev_names}

    spec = {
        "B1_train_prior": ("Device prior (train)", b1),
        "B2_train_prior_shrunk": (f"Device prior (train, n<{min_train} -> global)", b2),
        "B3_majority_device": (f"Indicator: {major_dev}", b3),
        "B4_oracle_test_prev": ("ORACLE: test prevalence per device", b4),
    }
    scores = {k: np.array([m[e] for e in eq_test], dtype=np.float64)
              for k, (_, m) in spec.items()}
    for k, (disp, m) in spec.items():
        logger.info(f"  {k:<24} " + ", ".join(f"{e}={v:.3f}" for e, v in m.items()))
    meta = {"global_train_prior": g_prior, "majority_train_device": str(major_dev),
            "min_train": min_train, "scores_by_device": {k: m for k, (_, m) in spec.items()},
            "display": {k: d for k, (d, _) in spec.items()}}
    return scores, meta


def check_b1_against_script29(auc_b1):
    """B1 must reproduce the device-prior AUC of Script 29."""
    path = OUT_DIR / "reanalysis_trivial_baselines.csv"
    if not path.exists():
        logger.warning(f"  {path.name} not found; B1 is not cross-checked against Script 29")
        return None
    tb = pd.read_csv(path)
    row = tb[tb["baseline"].astype(str).str.startswith("device-prior")]
    if row.empty:
        logger.warning("  device-prior not found in the Script 29 CSV; B1 is not cross-checked")
        return None
    ref = float(row["auc"].iloc[0])
    delta = abs(auc_b1 - ref)
    if delta > 1e-9:
        raise AssertionError(f"B1 AUC {auc_b1:.6f} does not reproduce Script 29 ({ref:.6f}); Δ={delta:.2e}")
    logger.info(f"  B1 reproduces the Script 29 device-prior: AUC {auc_b1:.4f}")
    return ref


def evaluate_sample(idx, y_bin, dev_codes, n_dev, runs, models, base_scores, linear):
    """
    Metrics for one sample (full test set or one bootstrap replicate):
      baselines: pooled AUC
      models:    pooled/within/between/devmean, averaged over seeds
      linear:    pooled/within/between/devmean
    """
    yb, dc = y_bin[idx], dev_codes[idx]
    out = {"baselines": {}, "models": {}, "linear": {}}
    for k, sc in base_scores.items():
        out["baselines"][k] = u_to_auc(*u_statistic(yb, sc[idx]))
    for key in models:
        acc = {"pooled": 0.0, "within": 0.0, "between": 0.0, "devmean": 0.0}
        for seed in SEEDS:
            s = runs[key][seed]["p_abn"][idx]
            dec = auc_decomposition(yb, s, dc, n_dev)
            for m in ["pooled", "within", "between"]:
                acc[m] += dec[m]
            acc["devmean"] += u_to_auc(*u_statistic(yb, device_mean_scores(s, dc, n_dev)))
        out["models"][key] = {m: v / len(SEEDS) for m, v in acc.items()}
    for name, sc in linear.items():
        s = sc[idx]
        dec = auc_decomposition(yb, s, dc, n_dev)
        dec["devmean"] = u_to_auc(*u_statistic(yb, device_mean_scores(s, dc, n_dev)))
        out["linear"][name] = {m: dec[m] for m in ["pooled", "within", "between", "devmean"]}
    frac = auc_decomposition(yb, np.zeros(len(yb)), dc, n_dev)["frac_within"]
    out["frac_within"] = frac
    return out


def u_to_auc(u, pairs):
    return u / pairs if pairs else float("nan")


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, dtype=float)
    return float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))


def p_boot(d):
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    return float(2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else float("nan")


def main():
    p = argparse.ArgumentParser(description="Recording-device confound in the binary AUC")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--boot_seed", type=int, default=2026)
    p.add_argument("--delta", type=float, default=0.02,
                   help="TOST margin (pre-registered in Script 29; do not change)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min_train", type=int, default=50,
                   help="Minimum training cycles for a device to keep its own prior (B2)")
    p.add_argument("--skip_verify", action="store_true",
                   help="Skip the cross-check against master_results.json (testing only)")
    args = p.parse_args()

    log_file = setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 78)
    logger.info("SCRIPT 35: Recording-device confound in the binary AUC")
    logger.info("=" * 78)

    logger.info("\n[1/4] Loading, alignment and integrity")
    df = s29.load_split()
    runs = s29.load_runs(df)
    if args.skip_verify:
        logger.warning("  --skip_verify: figures are NOT citable")
    else:
        s29.verify_against_master(df, runs)
    models = list(s29.MODELS.keys())
    y_bin = (df["y_true"].values != 0).astype(int)
    dev_names = sorted(df["recording_equipment"].unique())
    dev_index = {e: i for i, e in enumerate(dev_names)}
    dev_codes = df["recording_equipment"].map(dev_index).values.astype(np.int64)
    n_dev = len(dev_names)

    linear = {}
    for name, fn in LINEAR_FILES.items():
        path = OUT_DIR / fn
        if path.exists():
            arr = np.load(path).astype(np.float64)
            if len(arr) != len(df):
                raise ValueError(f"{fn}: {len(arr)} scores != {len(df)} cycles")
            linear[name] = arr
            logger.info(f"  Linear scores: {fn}")
        else:
            logger.warning(f"  {fn} not found; skipping {name}")

    logger.info("\n[2/4] Device-only baselines")
    base_scores, base_meta = device_baselines(df, dev_names, args.min_train)

    logger.info("\n[3/4] Point estimate on the full test set")
    full = np.arange(len(df))
    point = evaluate_sample(full, y_bin, dev_codes, n_dev, runs, models, base_scores, linear)
    check_b1_against_script29(point["baselines"]["B1_train_prior"])
    logger.info(f"  Fraction of (abnormal, normal) pairs from the same device: {point['frac_within']:.4f}")
    for k, v in point["baselines"].items():
        logger.info(f"  {k:<24} AUC {v:.4f}")
    per_seed_rows = []
    for key in models:
        for seed in SEEDS:
            s = runs[key][seed]["p_abn"]
            dec = auc_decomposition(y_bin, s, dev_codes, n_dev)
            dec["devmean"] = u_to_auc(*u_statistic(y_bin, device_mean_scores(s, dev_codes, n_dev)))
            per_seed_rows.append({"model": key, "seed": seed, **dec})
    per_seed = pd.DataFrame(per_seed_rows)
    for key in models:
        sub = per_seed[per_seed["model"] == key]
        logger.info(f"  {key:<11} pooled {sub['pooled'].mean():.4f}  within {sub['within'].mean():.4f} "
                    f"± {sub['within'].std(ddof=1):.4f}  between {sub['between'].mean():.4f}  "
                    f"devmean {sub['devmean'].mean():.4f}")
    for name, v in point["linear"].items():
        logger.info(f"  {name:<11} pooled {v['pooled']:.4f}  within {v['within']:.4f}  "
                    f"between {v['between']:.4f}  devmean {v['devmean']:.4f}")

    # Per-seed paired t-test and TOST on AUC_within (Script 29 implementation)
    tost_rows = []
    ref_w = per_seed[per_seed["model"] == REFERENCE].set_index("seed").loc[SEEDS, "within"].values
    for key in models:
        if key == REFERENCE:
            continue
        x_w = per_seed[per_seed["model"] == key].set_index("seed").loc[SEEDS, "within"].values
        res = s29.tost_paired(ref_w, x_w, args.delta, args.alpha)
        res.update({"comparison": f"{REFERENCE} - {key}", "metric": "auc_within"})
        tost_rows.append(res)
        logger.info(f"  TOST within {REFERENCE} - {key:<10} Δ {res['diff_mean']:+.4f} "
                    f"CI90 [{res['ci90_lo']:+.4f}, {res['ci90_hi']:+.4f}]  t p {res['p_ttest_two_sided']:.3f}  "
                    f"TOST p {res['p_tost']:.3f}")

    logger.info(f"\n[4/4] Patient-level bootstrap (B = {args.n_boot}, seed {args.boot_seed})")
    rng = np.random.default_rng(args.boot_seed)
    patients = df["patient_id"].values
    uniq = np.unique(patients)
    pat_idx = {pp: np.where(patients == pp)[0] for pp in uniq}
    reps = []
    for b in range(args.n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pat_idx[pp] for pp in draw])
        reps.append(evaluate_sample(idx, y_bin, dev_codes, n_dev, runs, models, base_scores, linear))
        if (b + 1) % 250 == 0:
            logger.info(f"    replicate {b + 1}/{args.n_boot}")

    def series(getter):
        return np.array([getter(r) for r in reps], dtype=float)

    base_rows = []
    for k, v in point["baselines"].items():
        lo, hi = ci(series(lambda r, k=k: r["baselines"][k]))
        base_rows.append({"baseline": k, "display": base_meta["display"][k], "auc": v,
                          "auc_ci_lo": lo, "auc_ci_hi": hi,
                          "oracle": k.startswith("B4")})

    dec_rows = []
    for key in models:
        sub = per_seed[per_seed["model"] == key]
        row = {"model": key, "display": s29.MODELS[key]["display"]}
        for m in ["pooled", "within", "between", "devmean"]:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_sd"] = float(sub[m].std(ddof=1))
            row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = ci(series(lambda r, key=key, m=m: r["models"][key][m]))
        dec_rows.append(row)
    for name, v in point["linear"].items():
        row = {"model": name, "display": LINEAR_DISPLAY[name]}
        for m in ["pooled", "within", "between", "devmean"]:
            row[f"{m}_mean"], row[f"{m}_sd"] = v[m], float("nan")
            row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = ci(series(lambda r, name=name, m=m: r["linear"][name][m]))
        dec_rows.append(row)

    # Paired contrasts (same patient replicate for every quantity)
    contrasts = []

    def add_contrast(label, point_d, d_series):
        lo, hi = ci(d_series)
        contrasts.append({"contrast": label, "diff": point_d, "ci_lo": lo, "ci_hi": hi,
                          "p_boot": p_boot(d_series)})
        logger.info(f"  {label:<46} Δ {point_d:+.4f} [{lo:+.4f}, {hi:+.4f}]  p {p_boot(d_series):.3f}")

    ref = REFERENCE
    add_contrast(f"{ref} pooled - B3 indicator",
                 point["models"][ref]["pooled"] - point["baselines"]["B3_majority_device"],
                 series(lambda r: r["models"][ref]["pooled"] - r["baselines"]["B3_majority_device"]))
    add_contrast(f"{ref} pooled - B2 shrunk prior",
                 point["models"][ref]["pooled"] - point["baselines"]["B2_train_prior_shrunk"],
                 series(lambda r: r["models"][ref]["pooled"] - r["baselines"]["B2_train_prior_shrunk"]))
    add_contrast(f"{ref} within - 0.5",
                 point["models"][ref]["within"] - 0.5,
                 series(lambda r: r["models"][ref]["within"] - 0.5))
    for key in models:
        if key == ref:
            continue
        add_contrast(f"{ref} within - {key} within",
                     point["models"][ref]["within"] - point["models"][key]["within"],
                     series(lambda r, key=key: r["models"][ref]["within"] - r["models"][key]["within"]))
    for name in point["linear"]:
        for m in ["pooled", "within"]:
            add_contrast(f"{ref} {m} - {name} {m}",
                         point["models"][ref][m] - point["linear"][name][m],
                         series(lambda r, name=name, m=m: r["models"][ref][m] - r["linear"][name][m]))

    pd.DataFrame(base_rows).to_csv(OUT_DIR / "device_confound_baselines.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(OUT_DIR / "device_confound_decomposition.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT_DIR / "device_confound_contrasts.csv", index=False)
    pd.DataFrame(tost_rows).to_csv(OUT_DIR / "device_confound_tost_within.csv", index=False)
    with open(OUT_DIR / "device_confound_results.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "n_cycles": int(len(df)), "n_patients": int(len(uniq)),
                   "devices_test": dev_names, "frac_pairs_within_device": point["frac_within"],
                   "n_boot": args.n_boot, "boot_seed": args.boot_seed,
                   "baseline_meta": base_meta, "baselines": base_rows,
                   "per_seed": per_seed_rows, "decomposition": dec_rows,
                   "contrasts": contrasts, "tost_within": tost_rows,
                   "delta": args.delta, "alpha": args.alpha}, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "device_confound_summary.md", "w", encoding="utf-8") as f:
        f.write("# Recording-device confound in the binary AUC (Script 35)\n\n")
        f.write(f"_{datetime.now().isoformat(timespec='seconds')} · {len(df)} cycles · "
                f"{len(uniq)} patients · B = {args.n_boot} · "
                f"same-device pairs = {point['frac_within']:.4f}_\n\n")
        f.write("## Device-only baselines (patient-level 95 % CI)\n\n")
        f.write("| Baseline | AUC [95 % CI] | Note |\n|---|---|---|\n")
        for r in base_rows:
            note_text = "oracle, not citable as a baseline" if r["oracle"] else ""
            f.write(f"| {r['display']} | {r['auc']:.3f} [{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}] | {note_text} |\n")
        f.write("\n## AUC decomposition (mean ± sd over seeds; patient-level 95 % CI)\n\n")
        f.write("| Model | Pooled [CI] | Same device [CI] | Cross-device | Device mean only [CI] |\n")
        f.write("|---|---|---|---|---|\n")
        for r in dec_rows:
            sd = (lambda m: f" ± {r[m + '_sd']:.3f}" if not np.isnan(r[m + "_sd"]) else "")
            f.write(f"| {r['display']} "
                    f"| {r['pooled_mean']:.3f}{sd('pooled')} [{r['pooled_ci_lo']:.3f}, {r['pooled_ci_hi']:.3f}] "
                    f"| {r['within_mean']:.3f}{sd('within')} [{r['within_ci_lo']:.3f}, {r['within_ci_hi']:.3f}] "
                    f"| {r['between_mean']:.3f}{sd('between')} "
                    f"| {r['devmean_mean']:.3f}{sd('devmean')} [{r['devmean_ci_lo']:.3f}, {r['devmean_ci_hi']:.3f}] |\n")
        f.write("\n## Paired contrasts (same patient replicate)\n\n")
        f.write("| Contrast | Δ [95 % CI] | p boot |\n|---|---|---|\n")
        for c in contrasts:
            f.write(f"| {c['contrast']} | {c['diff']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | {c['p_boot']:.3f} |\n")
        f.write(f"\n## Same-device AUC: per-seed paired t-test and TOST (delta = {args.delta}, alpha = {args.alpha})\n\n")
        f.write("| Comparison | Mean Δ | 90 % CI | t p | TOST p | Equivalent |\n|---|---|---|---|---|---|\n")
        for t in tost_rows:
            f.write(f"| {t['comparison']} | {t['diff_mean']:+.4f} | [{t['ci90_lo']:+.4f}, {t['ci90_hi']:+.4f}] "
                    f"| {t['p_ttest_two_sided']:.3f} | {t['p_tost']:.3f} | {'yes' if t['equivalent'] else 'no'} |\n")
        f.write("\n_Post hoc analysis: report it as such._\n")

    logger.info(f"\nOutputs in {OUT_DIR}")
    logger.info(f"Log: {log_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
