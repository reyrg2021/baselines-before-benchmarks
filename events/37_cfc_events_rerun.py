#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Script 37: Traceable re-run of the CfC study on irregular event sequences.

Re-runs the event-based CfC experiment of Script 20 with the same model, recipe and data
pipeline (Scripts 19 and 20 are imported, not modified), but with the study seeds
(42, 123, 7, 99, 314), sample standard deviations (ddof = 1), results persisted to disk,
and an explicit count of the test cycles that enter the evaluation. Design: 2 conditions
(with / without segment duration as a feature) x 3 timespan modes (real / ones / mean)
x 5 seeds = 30 runs. Each run tracks two checkpoints at once: lowest validation loss
(primary) and best validation ICBHI score (sensitivity, as in Script 20); each arm freezes
when its own patience is exhausted, so it reproduces training with its own early stopping.
Analysis: official ICBHI, Se, Sp, pooled and within-device AUC, per-class recall, and
seed-paired contrasts (real vs ones/mean; with vs without duration) with paired t-test,
TOST (delta = 0.02, 90 % CI) and patient-level bootstrap (B = 2000, seed 2026).

Inputs:  data/events/events.pkl, data/splits/strategy1_official/{train,val,test}_cycles.csv,
         19_event_dataloader.py, 20_cfc_event_training.py, torch_cfc.py (same folder)
Outputs: results/metrics/events_rerun/
           runs/events_{dur|nodur}_{real|ones|mean}_seed{S}.npz / .json
           test_index.csv, dataset_stats.json,
           events_rerun_absolute.csv, events_rerun_contrasts.csv,
           events_rerun_results.json, events_rerun_summary.md
         script37_log.txt
Usage:
    python 37_cfc_events_rerun.py --events data/events/events.pkl \
        --split-dir data/splits/strategy1_official [--dry-run] [--analyze-only]
"""

import sys
import json
import time
import pickle
import argparse
import logging
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR_FIXED = Path(__file__).resolve().parent.parent
BASE_DIR = BASE_DIR_FIXED if BASE_DIR_FIXED.exists() else SCRIPT_DIR

PAPER_SEEDS = [42, 123, 7, 99, 314]
MODES = ["real", "ones", "mean"]
CONDITIONS = {"dur": False, "nodur": True}          # value = no_duration
SELECTIONS = ["val_loss", "val_score"]               # primary, sensitivity
CLASS_MAP = {"Normal": 0, "Crackle": 1, "Wheeze": 2, "Both": 3}
METRICS = ["icbhi", "auc", "auc_wd"]
COND_LABEL = {"dur": "with duration", "nodur": "without duration"}

logger = logging.getLogger(__name__)


def setup_logging(dry_run):
    """Log to stdout and to script37_log.txt in BASE_DIR."""
    log_file = BASE_DIR / ("script37_dryrun_log.txt" if dry_run else "script37_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, mode="a", encoding="utf-8")],
        force=True,
    )
    return log_file


def import_script(name, filename):
    """Import a numbered script from this script's folder as a module."""
    path = SCRIPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Script 37 must sit next to 19, 20 and torch_cfc.py")
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare_data(dl, events_pkl, split_dir):
    """
    Load events.pkl once and reproduce build_loaders of Script 19 (same order, same
    train-only scaler, same collate), while keeping the test file names so predictions
    can be aligned with patients and recording devices.
    """
    with open(events_pkl, "rb") as fh:
        events = pickle.load(fh)
    split_dir = Path(split_dir)
    fnames, missing, csvs = {}, {}, {}
    for split in ("train", "val", "test"):
        csv = split_dir / f"{split}_cycles.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Not found: {csv}")
        csvs[split] = csv
        fnames[split], missing[split] = dl.load_split_filenames(csv, events)
        logger.info(f"  {split:<5}: {len(fnames[split])} cycles used "
                    f"({missing[split]} missing from events.pkl)")
    scaler = dl.fit_scaler(events, fnames["train"])
    F = int(events[fnames["train"][0]]["feats"].shape[1])
    datasets = {mode: {split: dl.EventSequenceDataset(events, fnames[split], scaler=scaler,
                                                      timespan_mode=mode)
                       for split in ("train", "val", "test")}
                for mode in MODES}

    # Dataset statistics for the Methods description
    lengths = np.array([len(events[fn]["feats"]) for s in fnames for fn in fnames[s]])
    dts = np.concatenate([np.asarray(events[fn]["dt"], dtype=float) for s in fnames for fn in fnames[s]
                          if len(events[fn]["dt"])])
    ds_stats = {
        "split_dir": str(split_dir), "events_pkl": str(events_pkl),
        "n_cycles": {s: len(fnames[s]) for s in fnames},
        "n_missing": {s: int(missing[s]) for s in missing},
        "n_features": F,
        "events_per_cycle": {"mean": float(lengths.mean()), "median": float(np.median(lengths)),
                             "p5": float(np.percentile(lengths, 5)), "p95": float(np.percentile(lengths, 95)),
                             "min": int(lengths.min()), "max": int(lengths.max())},
        "dt_seconds": {"mean": float(dts.mean()), "median": float(np.median(dts)),
                       "p5": float(np.percentile(dts, 5)), "p95": float(np.percentile(dts, 95))},
    }
    logger.info(f"  F (features per event) = {F}")
    logger.info(f"  events per cycle: median {ds_stats['events_per_cycle']['median']:.0f} "
                f"(p5 {ds_stats['events_per_cycle']['p5']:.0f}, p95 {ds_stats['events_per_cycle']['p95']:.0f})")
    logger.info(f"  Δt (s): median {ds_stats['dt_seconds']['median']:.3f} "
                f"(p5 {ds_stats['dt_seconds']['p5']:.3f}, p95 {ds_stats['dt_seconds']['p95']:.3f})")

    # Test index aligned with the official CSV (patient, device, label)
    tdf = pd.read_csv(csvs["test"])
    col = "cycle_filename" if "cycle_filename" in tdf.columns else tdf.columns[0]
    tdf = tdf.set_index(col)
    idx = pd.DataFrame({"cycle_filename": fnames["test"]})
    idx["label_events"] = [int(events[fn]["label"]) for fn in fnames["test"]]
    for c in ("patient_id", "recording_equipment", "cycle_class"):
        idx[c] = tdf.loc[fnames["test"], c].values if c in tdf.columns else None
    if idx["cycle_class"].notna().all():
        lab_csv = idx["cycle_class"].map(CLASS_MAP).values
        mism = int((lab_csv != idx["label_events"].values).sum())
        if mism:
            raise AssertionError(f"{mism} labels in events.pkl do not match test_cycles.csv")
        logger.info("  Labels in events.pkl match test_cycles.csv")
    else:
        logger.warning("  test_cycles.csv has no cycle_class: labels not cross-checked")
    if idx["patient_id"].isna().any() or idx["recording_equipment"].isna().any():
        logger.warning("  patient_id or recording_equipment missing: no bootstrap and no within-device AUC")
    return datasets, F, idx, ds_stats


def train_run(s20, dl, datasets, F, mode, no_duration, seed, args, device):
    """
    One run with dual checkpoint selection. RNG consumption order is identical to
    Script 20: set_seed -> model -> epochs (shuffling happens when iterating the DataLoader).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    s20.set_seed(seed)
    loaders = {split: DataLoader(datasets[mode][split], batch_size=args.batch_size,
                                 shuffle=(split == "train"), collate_fn=dl.collate_events,
                                 num_workers=0)
               for split in ("train", "val", "test")}
    in_feat = F - 1 if no_duration else F
    model = s20.torch_cfc.Cfc(in_features=in_feat, hidden_size=args.hidden, out_feature=4,
                              hparams=s20.HPARAMS, return_sequences=True).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))

    ytr = np.array([y for *_, y in loaders["train"].dataset.items])
    counts = np.bincount(ytr, minlength=4).astype(float)
    w = torch.tensor(counts.sum() / (4 * np.clip(counts, 1, None)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    arms = {"val_score": {"best": -np.inf, "state": None, "epoch": -1, "bad": 0, "frozen": False},
            "val_loss": {"best": np.inf, "state": None, "epoch": -1, "bad": 0, "frozen": False}}
    history = []
    t0 = time.time()
    epochs_run = 0
    for ep in range(args.epochs):
        tr_loss, _, _, _ = s20.run_epoch(model, loaders["train"], optimizer, criterion, device, True, no_duration)
        v_loss, yv, pv, _ = s20.run_epoch(model, loaders["val"], optimizer, criterion, device, False, no_duration)
        v_score = float(s20.official_metric(yv, pv)["score"])
        history.append({"epoch": ep + 1, "train_loss": float(tr_loss), "val_loss": float(v_loss),
                        "val_score": v_score})
        epochs_run = ep + 1
        snapshot = None
        # val_score: same rule as Script 20 (strict improvement)
        a = arms["val_score"]
        if not a["frozen"]:
            if v_score > a["best"]:
                snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                a.update(best=v_score, state=snapshot, epoch=ep + 1, bad=0)
            else:
                a["bad"] += 1
                a["frozen"] = a["bad"] >= args.patience
        # val_loss: same rule applied to the validation loss
        a = arms["val_loss"]
        if not a["frozen"]:
            if v_loss < a["best"]:
                snap = snapshot if snapshot is not None else \
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                a.update(best=float(v_loss), state=snap, epoch=ep + 1, bad=0)
            else:
                a["bad"] += 1
                a["frozen"] = a["bad"] >= args.patience
        if arms["val_score"]["frozen"] and arms["val_loss"]["frozen"]:
            break

    out = {"n_params": n_params, "epochs_run": epochs_run, "seconds": time.time() - t0,
           "history": history}
    preds = {}
    y_ref = None
    for sel in SELECTIONS:
        a = arms[sel]
        if a["state"] is not None:
            model.load_state_dict(a["state"])
        _, yte, pte, prob = s20.run_epoch(model, loaders["test"], optimizer, criterion, device, False, no_duration)
        if y_ref is None:
            y_ref = yte
        elif not np.array_equal(y_ref, yte):
            raise AssertionError("Test order changed between selection arms")
        m = s20.official_metric(yte, pte)
        preds[f"y_pred_{sel}"] = pte.astype(np.int64)
        preds[f"p_abn_{sel}"] = prob.astype(np.float64)
        out[sel] = {"best_epoch": a["epoch"], "best_value": a["best"], "test_icbhi": float(m["score"]),
                    "test_se": float(m["se"]), "test_sp": float(m["sp"]),
                    "test_auc": float(s20.binary_auc(yte, prob))}
    preds["y_true"] = y_ref.astype(np.int64)
    return out, preds


def run_training(args, out_dir, device_hint=None):
    """Launch the 30 runs (or the --dry-run subset), skipping runs already saved."""
    import torch
    s20 = import_script("script20", "20_cfc_event_training.py")
    dl = s20._load_dataloader_module()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  torch {torch.__version__} | device {device}")
    logger.info(f"  Script 20 HPARAMS: {s20.HPARAMS}")

    logger.info("\n[1/3] Data")
    datasets, F, idx, ds_stats = prepare_data(dl, args.events, args.split_dir)
    idx.to_csv(out_dir / "test_index.csv", index=False)
    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(ds_stats, f, indent=2, ensure_ascii=False)

    logger.info("\n[2/3] Training")
    run_dir = out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    total = len(CONDITIONS) * len(MODES) * len(args.seeds)
    k = 0
    for cond, no_dur in CONDITIONS.items():
        for mode in MODES:
            for seed in args.seeds:
                k += 1
                stem = f"events_{cond}_{mode}_seed{seed}"
                if (run_dir / f"{stem}.npz").exists() and not args.force:
                    logger.info(f"  [{k}/{total}] {stem}: already exists, skipped")
                    continue
                info, preds = train_run(s20, dl, datasets, F, mode, no_dur, seed, args, device)
                if len(preds["y_true"]) != len(idx) or not np.array_equal(preds["y_true"], idx["label_events"].values):
                    raise AssertionError(f"{stem}: y_true not aligned with test_index.csv")
                np.savez(run_dir / f"{stem}.npz", **preds)
                info.update({"condition": cond, "mode": mode, "seed": seed, "no_duration": no_dur,
                             "args": {k2: (str(v) if isinstance(v, Path) else v) for k2, v in vars(args).items()},
                             "timestamp": datetime.now().isoformat(timespec="seconds")})
                with open(run_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=1)
                logger.info(f"  [{k}/{total}] {stem}: {info['epochs_run']} ep. in {info['seconds']:.0f} s | "
                            f"params {info['n_params']:,} | "
                            f"val_loss ep {info['val_loss']['best_epoch']} ICBHI {info['val_loss']['test_icbhi']:.3f} "
                            f"AUC {info['val_loss']['test_auc']:.3f} | "
                            f"val_score ep {info['val_score']['best_epoch']} ICBHI {info['val_score']['test_icbhi']:.3f} "
                            f"AUC {info['val_score']['test_auc']:.3f}")


def official(y, p):
    """Official ICBHI score, Se and Sp from the 4x4 confusion matrix."""
    cm = np.bincount(4 * y + p, minlength=16).reshape(4, 4)
    sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else np.nan
    se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / cm[1:].sum() if cm[1:].sum() else np.nan
    return (se + sp) / 2, se, sp, cm


def u_stat(yb, s):
    pos = yb == 1
    n_pos = int(pos.sum()); n_neg = len(yb) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0
    r = stats.rankdata(s)
    return float(r[pos].sum() - n_pos * (n_pos + 1) / 2.0), n_pos * n_neg


def auc_pooled_within(yb, s, dev, n_dev):
    """Pooled AUC and within-device AUC (only pairs recorded with the same device)."""
    u, p = u_stat(yb, s)
    uw, pw = 0.0, 0
    if dev is not None:
        for d in range(n_dev):
            m = dev == d
            if m.any():
                a, b = u_stat(yb[m], s[m]); uw += a; pw += b
    return (u / p if p else np.nan), (uw / pw if pw else np.nan)


def tost_paired(a, b, delta, alpha):
    """Same as Script 29: two-sided paired t-test, TOST and (1 - 2 alpha) CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    out = {"diff_mean": float(d.mean()), "n": n}
    if n < 2 or np.allclose(d.std(ddof=1), 0):
        out.update(ci90_lo=np.nan, ci90_hi=np.nan, p_t=np.nan, p_tost=np.nan, equivalent=False)
        return out
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha, n - 1)
    p_t = 2 * stats.t.sf(abs(d.mean() / se), n - 1)
    p_lower = stats.t.sf((d.mean() + delta) / se, n - 1)
    p_upper = stats.t.cdf((d.mean() - delta) / se, n - 1)
    out.update(ci90_lo=float(d.mean() - tcrit * se), ci90_hi=float(d.mean() + tcrit * se),
               p_t=float(p_t), p_tost=float(max(p_lower, p_upper)),
               equivalent=bool(max(p_lower, p_upper) < alpha))
    return out


def load_runs(run_dir, seeds, idx):
    """Load the saved predictions. Returns dict[(cond, mode, sel)][seed] and the per-run info."""
    runs, infos = {}, {}
    y_idx = idx["label_events"].values
    for cond in CONDITIONS:
        for mode in MODES:
            for seed in seeds:
                stem = f"events_{cond}_{mode}_seed{seed}"
                f_npz = run_dir / f"{stem}.npz"
                if not f_npz.exists():
                    raise FileNotFoundError(f"Missing {f_npz}; run the training first")
                z = np.load(f_npz)
                if not np.array_equal(z["y_true"], y_idx):
                    raise AssertionError(f"{stem}: y_true not aligned with test_index.csv")
                with open(run_dir / f"{stem}.json", encoding="utf-8") as fh:
                    infos[(cond, mode, seed)] = json.load(fh)
                for sel in SELECTIONS:
                    runs.setdefault((cond, mode, sel), {})[seed] = {
                        "y_pred": z[f"y_pred_{sel}"].astype(int), "p_abn": z[f"p_abn_{sel}"].astype(float)}
    return runs, infos


def analyze(args, out_dir):
    """Absolute metrics, seed-paired contrasts and patient-level bootstrap."""
    logger.info("\n[3/3] Analysis")
    idx = pd.read_csv(out_dir / "test_index.csv")
    seeds = args.seeds
    runs, infos = load_runs(out_dir / "runs", seeds, idx)
    y = idx["label_events"].values.astype(int)
    yb = (y != 0).astype(int)
    has_pat = idx["patient_id"].notna().all() and idx["recording_equipment"].notna().all()
    if has_pat:
        devs = sorted(idx["recording_equipment"].unique())
        dev = idx["recording_equipment"].map({d: i for i, d in enumerate(devs)}).values.astype(int)
    else:
        devs, dev = [], None
    n_dev = len(devs)
    logger.info(f"  test: {len(idx)} cycles · {idx['patient_id'].nunique() if has_pat else 'n/a'} patients · "
                f"devices {devs}")

    def metrics_of(run, sel_idx):
        yy, pp, ss = y[sel_idx], run["y_pred"][sel_idx], run["p_abn"][sel_idx]
        ic, se, sp, cm = official(yy, pp)
        a, aw = auc_pooled_within(yb[sel_idx], ss, dev[sel_idx] if dev is not None else None, n_dev)
        rec = [cm[i, i] / cm[i].sum() if cm[i].sum() else np.nan for i in range(4)]
        return {"icbhi": ic, "se": se, "sp": sp, "auc": a, "auc_wd": aw,
                "rec_normal": rec[0], "rec_crackle": rec[1], "rec_wheeze": rec[2], "rec_both": rec[3]}

    full = np.arange(len(y))
    per_seed = {key: {s: metrics_of(runs[key][s], full) for s in seeds} for key in runs}

    # Patient-level bootstrap of the seed-averaged metrics
    boot = None
    if has_pat and args.n_boot > 0:
        rng = np.random.default_rng(args.boot_seed)
        patients = idx["patient_id"].values
        uniq = np.unique(patients)
        pat_idx = {p: np.where(patients == p)[0] for p in uniq}
        boot = {key: {m: np.empty(args.n_boot) for m in METRICS} for key in runs}
        for b in range(args.n_boot):
            draw = rng.choice(uniq, size=len(uniq), replace=True)
            bi = np.concatenate([pat_idx[p] for p in draw])
            for key in runs:
                acc = {m: 0.0 for m in METRICS}
                for s in seeds:
                    r = runs[key][s]
                    ic, _, _, _ = official(y[bi], r["y_pred"][bi])
                    a, aw = auc_pooled_within(yb[bi], r["p_abn"][bi], dev[bi], n_dev)
                    acc["icbhi"] += ic; acc["auc"] += a; acc["auc_wd"] += aw
                for m in METRICS:
                    boot[key][m][b] = acc[m] / len(seeds)
            if (b + 1) % 250 == 0:
                logger.info(f"    replicate {b + 1}/{args.n_boot}")

    def ci95(v):
        v = np.asarray(v, float); v = v[~np.isnan(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)

    def pboot(d):
        d = np.asarray(d, float); d = d[~np.isnan(d)]
        return float(2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else np.nan

    # Absolute metrics table
    abs_rows = []
    for sel in SELECTIONS:
        for cond in CONDITIONS:
            for mode in MODES:
                key = (cond, mode, sel)
                row = {"selection": sel, "condition": cond, "mode": mode}
                for m in ["icbhi", "se", "sp", "auc", "auc_wd", "rec_wheeze", "rec_both"]:
                    v = np.array([per_seed[key][s][m] for s in seeds], float)
                    row[f"{m}_mean"] = float(np.nanmean(v))
                    row[f"{m}_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan
                if boot is not None:
                    for m in METRICS:
                        row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = ci95(boot[key][m])
                row["best_epoch_mean"] = float(np.mean([infos[(cond, mode, s)][sel]["best_epoch"] for s in seeds]))
                row["epochs_run_mean"] = float(np.mean([infos[(cond, mode, s)]["epochs_run"] for s in seeds]))
                row["n_params"] = infos[(cond, mode, seeds[0])]["n_params"]
                abs_rows.append(row)

    # Seed-paired contrasts: Q1 = irregular timing, Q3 = duration as a feature
    pairs = []
    for sel in SELECTIONS:
        for cond in CONDITIONS:
            pairs.append((sel, f"Q1 {cond}: real - ones", (cond, "real", sel), (cond, "ones", sel)))
            pairs.append((sel, f"Q1 {cond}: real - mean", (cond, "real", sel), (cond, "mean", sel)))
        for mode in MODES:
            pairs.append((sel, f"Q3 {mode}: dur - nodur", ("dur", mode, sel), ("nodur", mode, sel)))
    contrast_rows = []
    for sel, label, ka, kb in pairs:
        for m in METRICS:
            t = tost_paired([per_seed[ka][s][m] for s in seeds], [per_seed[kb][s][m] for s in seeds],
                            args.delta, args.alpha)
            row = {"selection": sel, "contrast": label, "metric": m, **t}
            if boot is not None:
                d = boot[ka][m] - boot[kb][m]
                row["boot_ci_lo"], row["boot_ci_hi"] = ci95(d)
                row["p_boot"] = pboot(d)
                row["supported"] = bool((t["p_t"] < args.alpha) if not np.isnan(t["p_t"]) else False) and \
                    bool(row["boot_ci_lo"] > 0 or row["boot_ci_hi"] < 0)
            else:
                row.update(boot_ci_lo=np.nan, boot_ci_hi=np.nan, p_boot=np.nan, supported=False)
            contrast_rows.append(row)
            logger.info(f"  [{sel}] {label:<24} {m:<6} Δ {t['diff_mean']:+.4f} "
                        f"CI90 [{t['ci90_lo']:+.4f}, {t['ci90_hi']:+.4f}] t p {t['p_t']:.3f} "
                        f"TOST p {t['p_tost']:.3f} | boot [{row['boot_ci_lo']:+.4f}, {row['boot_ci_hi']:+.4f}] "
                        f"| supported: {'YES' if row['supported'] else 'no'}")

    # Outputs
    pd.DataFrame(abs_rows).to_csv(out_dir / "events_rerun_absolute.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(out_dir / "events_rerun_contrasts.csv", index=False)
    ds_stats = json.load(open(out_dir / "dataset_stats.json", encoding="utf-8"))
    with open(out_dir / "events_rerun_results.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"), "seeds": seeds,
                   "n_boot": args.n_boot, "boot_seed": args.boot_seed, "delta": args.delta,
                   "dataset": ds_stats, "absolute": abs_rows, "contrasts": contrast_rows,
                   "per_seed": {f"{k[0]}|{k[1]}|{k[2]}": {str(s): v for s, v in d.items()}
                                for k, d in per_seed.items()}},
                  f, indent=2, ensure_ascii=False, default=float)

    fmt = lambda x: "n/d" if (x is None or (isinstance(x, float) and np.isnan(x))) else \
        (f"{x:.3f}" if x >= 0.001 else "<0.001")
    with open(out_dir / "events_rerun_summary.md", "w", encoding="utf-8") as f:
        f.write("# Re-run of the CfC study on event sequences (Script 37)\n\n")
        f.write(f"_{datetime.now().isoformat(timespec='seconds')} · seeds {seeds} · "
                f"B = {args.n_boot} · delta = {args.delta}_\n\n")
        f.write("## Data\n\n")
        f.write(f"- Split: `{ds_stats['split_dir']}`\n")
        f.write(f"- Cycles used: train {ds_stats['n_cycles']['train']} · val {ds_stats['n_cycles']['val']} · "
                f"test {ds_stats['n_cycles']['test']} (missing from events.pkl: {ds_stats['n_missing']})\n")
        f.write(f"- Test: {len(idx)} cycles · "
                f"{idx['patient_id'].nunique() if has_pat else 'n/a'} patients\n")
        f.write(f"- Features per event F = {ds_stats['n_features']} · events per cycle median "
                f"{ds_stats['events_per_cycle']['median']:.0f} (p5 {ds_stats['events_per_cycle']['p5']:.0f}, "
                f"p95 {ds_stats['events_per_cycle']['p95']:.0f}) · Δt median {ds_stats['dt_seconds']['median']:.3f} s\n")
        f.write(f"- CfC parameters: with duration {abs_rows[0]['n_params']:,} · "
                f"without duration {[r for r in abs_rows if r['condition'] == 'nodur'][0]['n_params']:,}\n\n")
        for sel in SELECTIONS:
            tag = "PRIMARY, selection by validation loss" if sel == "val_loss" else "sensitivity, reproduces Script 20"
            f.write(f"## Absolute metrics · selection by {sel} ({tag})\n\n")
            f.write("| Condition | Δt | ICBHI | Se | Sp | AUC [CI] | Within-device AUC [CI] | Se Wheeze | Se Both | selected ep. |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for r in [r for r in abs_rows if r["selection"] == sel]:
                ci_a = f" [{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}]" if boot is not None else ""
                ci_w = f" [{r['auc_wd_ci_lo']:.3f}, {r['auc_wd_ci_hi']:.3f}]" if boot is not None else ""
                f.write(f"| {COND_LABEL[r['condition']]} | {r['mode']} "
                        f"| {r['icbhi_mean']:.3f} ± {r['icbhi_sd']:.3f} | {r['se_mean']:.3f} ± {r['se_sd']:.3f} "
                        f"| {r['sp_mean']:.3f} ± {r['sp_sd']:.3f} | {r['auc_mean']:.3f} ± {r['auc_sd']:.3f}{ci_a} "
                        f"| {r['auc_wd_mean']:.3f} ± {r['auc_wd_sd']:.3f}{ci_w} "
                        f"| {r['rec_wheeze_mean']:.3f} | {r['rec_both_mean']:.3f} | {r['best_epoch_mean']:.1f} |\n")
            f.write(f"\n## Paired contrasts · selection by {sel}\n\n")
            f.write("| Contrast | Metric | Δ | 90 % CI (seeds) | t p | TOST p | 95 % CI (patients) | p boot | Supported |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in [r for r in contrast_rows if r["selection"] == sel]:
                f.write(f"| {r['contrast']} | {r['metric']} | {r['diff_mean']:+.4f} "
                        f"| [{r['ci90_lo']:+.4f}, {r['ci90_hi']:+.4f}] | {fmt(r['p_t'])} | {fmt(r['p_tost'])} "
                        f"| [{r['boot_ci_lo']:+.4f}, {r['boot_ci_hi']:+.4f}] | {fmt(r['p_boot'])} "
                        f"| {'yes' if r['supported'] else 'no'} |\n")
            f.write("\n")
        f.write("_Sample sd (ddof = 1). Within-device AUC: post hoc. "
                "Supported = t p < 0.05 and 95 % bootstrap CI excluding zero._\n")
    logger.info(f"\nOutputs in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Traceable re-run of the CfC study on event sequences")
    ap.add_argument("--events", required=True, help="Path to events.pkl (Script 18)")
    ap.add_argument("--split-dir", required=True, help="Split folder (strategy1_official)")
    ap.add_argument("--seeds", type=int, nargs="+", default=PAPER_SEEDS)
    ap.add_argument("--epochs", type=int, default=100)      # same as Script 20
    ap.add_argument("--batch-size", type=int, default=32)   # same as Script 20
    ap.add_argument("--hidden", type=int, default=64)       # same as Script 20
    ap.add_argument("--lr", type=float, default=1e-3)       # same as Script 20
    ap.add_argument("--patience", type=int, default=15)     # same as Script 20
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=2026)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--force", action="store_true", help="Retrain even if saved runs exist")
    ap.add_argument("--analyze-only", action="store_true", help="Only analyze saved runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Smoke test: 1 seed, 2 epochs, B = 20, separate output folder")
    args = ap.parse_args()

    if args.dry_run:
        args.seeds, args.epochs, args.n_boot = args.seeds[:1], 2, 20
    out_dir = Path(args.out_dir) if args.out_dir else \
        BASE_DIR / "results" / "metrics" / ("events_rerun_dryrun" if args.dry_run else "events_rerun")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(args.dry_run)

    logger.info("=" * 78)
    logger.info("SCRIPT 37: Traceable re-run of the CfC study on event sequences")
    logger.info("=" * 78)
    logger.info(f"  events: {args.events}")
    logger.info(f"  split : {args.split_dir}")
    logger.info(f"  seeds {args.seeds} · epochs {args.epochs} · patience {args.patience} · "
                f"batch {args.batch_size} · hidden {args.hidden} · lr {args.lr}")
    logger.info(f"  output: {out_dir}")
    if "strategy1_official" not in str(args.split_dir):
        logger.warning("  Split does not look like strategy1_official: results would not be comparable with the main study")
    if args.dry_run:
        logger.warning("  --dry-run: results are NOT citable")

    if not args.analyze_only:
        run_training(args, out_dir)
    analyze(args, out_dir)
    logger.info(f"Log: {log_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
