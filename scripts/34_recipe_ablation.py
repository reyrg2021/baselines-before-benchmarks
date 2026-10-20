#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 34: Training-recipe ablation for CNN-RespiratoryNet and MobileNetV3-Small.

The compared configurations were not trained with the same recipe: the CNN (Script 09
via 14) used no augmentation, plain CE, a weighted sampler and val_loss selection, while
MobileNetV3-Small (Script 21) used the "moderate" augmentation preset plus mixup
(alpha=0.2). Performance differences could therefore reflect the protocol rather than
the architecture. The gap is closed from both sides, each arm using the original code of
its own script, so the recipe is the only factor that changes:

  Arm A "cnn_aug":    CNN with augment_preset="moderate", rest of the canonical recipe
                      unchanged (Script 09 patched in memory, as Script 14 does).
  Arm B "mnv3_noaug": MobileNetV3-ImageNet with augment_preset="none", mixup_alpha=0
                      (Script 21 CLI, then --eval_only to export the test arrays).

If neither arm moves the result beyond seed-to-seed variation, the original comparison
holds. Mixup is not replicated in the CNN (Script 09 has no mixup); Arm B covers that
direction. Each arm is compared per seed against its baseline with a paired t-test and
TOST (margin +/-0.02, the same as Script 29). New checkpoint and array prefixes ensure no
existing artifact is overwritten; Script 29 reads them through two optional MODELS entries.

Inputs:  scripts/09_cnn_training.py, scripts/21_transfer_learning.py,
         scripts/23_quantize_cnn_int8.py, data/splits/strategy1_official/,
         results/metrics/full_reeval/arrays/ (baseline arrays)
Outputs: models/checkpoints/{prefix}_seed{S}.pth,
         results/metrics/full_reeval/arrays/{prefix}_seed{S}_probs.npy, _labels_preds.npz,
         results/metrics/recipe_ablation/recipe_ablation_runs.csv,
         recipe_ablation_summary.json, recipe_ablation_summary.md, script34_log.txt
Usage:
    python 34_recipe_ablation.py --arm cnn_aug --seed 42   # one run
    python 34_recipe_ablation.py --all                     # 2 arms x 5 seeds
    python 34_recipe_ablation.py --evaluate                # evaluate existing checkpoints
    Options: [--preset moderate] [--epochs N] [--skip_existing]
"""

import sys
import json
import argparse
import logging
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR_FIXED = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if BASE_DIR_FIXED.exists():
    BASE_DIR = BASE_DIR_FIXED
elif (SCRIPTS_DIR.parent / "models").exists():
    BASE_DIR = SCRIPTS_DIR.parent
else:
    BASE_DIR = SCRIPTS_DIR

CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
ARRAYS_DIR = BASE_DIR / "results" / "metrics" / "full_reeval" / "arrays"
OUT_DIR = BASE_DIR / "results" / "metrics" / "recipe_ablation"
SPLIT_DIR = BASE_DIR / "data" / "splits" / "strategy1_official"
MASTER_JSON = BASE_DIR / "results" / "master_results.json"

SEEDS = [42, 123, 7, 99, 314]
SPLIT = "strategy1_official"
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}

ARMS = {
    "cnn_aug": {
        "display": "CNN-RespiratoryNet + augmentation",
        "prefix": "cnn_aug_official",
        "baseline_display": "CNN-RespiratoryNet (FP32)",
        "baseline_master_key": "CNN_FP32",
        "baseline_prefix": "cnn",
    },
    "mnv3_noaug": {
        "display": "MobileNetV3-ImageNet without augmentation or mixup",
        "prefix": "tl_mobilenet_v3_small_pretrained_noaug_official",
        "baseline_display": "MobileNetV3-Small (ImageNet)",
        "baseline_master_key": "MobileNetV3_ImageNet",
        "baseline_prefix": "tl_mobilenet_v3_small_pretrained_official",
    },
}

logger = logging.getLogger(__name__)


def setup_logging():
    """Log to stdout and append to script34_log.txt in BASE_DIR."""
    log_file = BASE_DIR / "script34_log.txt"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, mode="a", encoding="utf-8")], force=True)
    return log_file


def import_script(name, filename):
    """Import a numbered script from SCRIPTS_DIR as a module."""
    path = SCRIPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Metric definitions identical to Scripts 18, 29, 30, 31 and 33

def icbhi_official(y_true, y_pred):
    """Official ICBHI score (Se + Sp) / 2; Se counts exact abnormal-class matches."""
    cm = np.zeros((4, 4))
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    abn = cm[1:].sum()
    se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / abn if abn else 0.0
    return (se + sp) / 2, float(se), float(sp)


def auc_binary(y_true, probs32):
    """Normal-vs-abnormal AUC with P(abnormal) computed in float32, as in Script 18."""
    p_abn = (np.float32(1.0) - np.asarray(probs32, dtype=np.float32)[:, 0]).astype(np.float64)
    yb = (y_true != 0).astype(int)
    pos = yb == 1
    n_pos, n_neg = int(pos.sum()), int(len(yb) - pos.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = stats.rankdata(p_abn)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics_from(y_true, probs32):
    """Official ICBHI, binary Se/Sp, binary AUC and per-class sensitivity from probabilities."""
    y_pred = np.argmax(probs32, axis=1)
    s, se, sp = icbhi_official(y_true, y_pred)
    yt, yp = y_true != 0, y_pred != 0
    return {"icbhi": s, "se_off": se, "sp_off": sp,
            "bin_se": float((yt & yp).sum() / yt.sum()),
            "bin_sp": float((~yt & ~yp).sum() / (~yt).sum()),
            "auc": auc_binary(y_true, probs32),
            "per_class_se": [float(((y_true == c) & (y_pred == c)).sum() / max((y_true == c).sum(), 1))
                             for c in range(4)]}


def mean_sd(x):
    """Mean and sample SD ignoring NaN."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else 0.0)


# Arm A: CNN with augmentation (in-memory patch of Script 09, as in Script 14)

def train_cnn_aug(seed, args):
    """
    Run Script 09 in a subprocess with augment_preset injected in memory, using the same
    canonical flags and split as Script 14, and --tag so the checkpoint gets its own name
    instead of best_model.pth.
    """
    run_tag = f"{ARMS['cnn_aug']['prefix']}_seed{seed}"
    script_path = SCRIPTS_DIR / "09_cnn_training.py"
    epochs_line = f"sys.argv += ['--epochs', '{args.epochs}']" if args.epochs else ""
    code = f"""
import sys, importlib.util
from pathlib import Path

script_path = Path(r'{script_path}')
sys.path.insert(0, str(script_path.parent))
spec = importlib.util.spec_from_file_location("script09", str(script_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# In-memory patch of split and augmentation. The rest of the config is untouched, so
# the ONLY factor that differs from cnn_official_seed{seed} is augment_preset.
mod.TRAINING_CONFIG['split_strategy'] = '{SPLIT}'
mod.TRAINING_CONFIG['augment_preset'] = '{args.preset}'

# Script 09 visualization/export tools always write to fixed shared paths (best_model.onnx,
# final_model.pth, training_config.json, training_curves.png, ...). Disabled so these runs
# do not overwrite the artifacts of the non-augmented CNN; the per-seed checkpoint
# (run_tag) has its own name.
for _k in ('use_torchinfo', 'use_torchview', 'use_torchviz', 'export_onnx',
           'use_tensorboard'):
    mod.TRAINING_CONFIG[_k] = False

# Canonical Script 14 recipe (plain CE + weighted sampler + val_loss).
sys.argv = [
    '09_cnn_training.py',
    '--seed', '{seed}',
    '--loss', 'ce',
    '--sampler', 'weighted',
    '--select_metric', 'val_loss',
    '--tag', '{run_tag}',
]
{epochs_line}
print(">>> Arm A: CNN augment_preset={args.preset} seed={seed} split={SPLIT}")
mod.main()
"""
    logger.info(f"  Script 09 subprocess (augment_preset={args.preset}, seed {seed})")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(BASE_DIR))
    if r.returncode != 0:
        raise RuntimeError(f"Script 09 failed with exit code {r.returncode} (seed {seed})")
    ckpt = CHECKPOINTS_DIR / f"{run_tag}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Script 09 did not produce {ckpt}")
    return ckpt


def eval_cnn_arm(seed, s23, data, y_csv):
    """Run the Arm A checkpoint on the test set and save the probability/label arrays."""
    import torch
    run_tag = f"{ARMS['cnn_aug']['prefix']}_seed{seed}"
    ckpt = CHECKPOINTS_DIR / f"{run_tag}.pth"
    model = s23.load_fp32_cnn(ckpt, "cpu")
    ys, ps = [], []
    with torch.no_grad():
        for batch in data["test_loader"]:
            x, y = batch[0], batch[1]
            ps.append(torch.softmax(model(x).float(), 1).cpu().numpy().astype(np.float32))
            ys.append(np.asarray(y))
    y_true, probs = np.concatenate(ys).astype(int), np.concatenate(ps)
    if not np.array_equal(y_true, y_csv):
        raise ValueError(f"{run_tag}: test set not aligned with test_cycles.csv")
    np.save(ARRAYS_DIR / f"{run_tag}_probs.npy", probs)
    np.savez(ARRAYS_DIR / f"{run_tag}_labels_preds.npz",
             y_true=y_true, y_pred=np.argmax(probs, 1))
    return y_true, probs


# Arm B: MobileNetV3 without augmentation or mixup (Script 21 CLI)

def train_mnv3_noaug(seed, args):
    """Train through Script 21's own flags (arrays are exported later with --eval_only)."""
    run_tag = f"{ARMS['mnv3_noaug']['prefix']}_seed{seed}"
    script = str(SCRIPTS_DIR / "21_transfer_learning.py")
    cmd = [sys.executable, script, "--backbone", "mobilenet_v3_small", "--seed", str(seed),
           "--split", SPLIT, "--augment_preset", "none", "--mixup_alpha", "0",
           "--tag", run_tag]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    logger.info(f"  Script 21 subprocess (augment=none, mixup=0, seed {seed})")
    r = subprocess.run(cmd, cwd=str(BASE_DIR))
    if r.returncode != 0:
        raise RuntimeError(f"Script 21 failed with exit code {r.returncode} (seed {seed})")
    ckpt = CHECKPOINTS_DIR / f"{run_tag}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Script 21 did not produce {ckpt}")
    return ckpt


def eval_mnv3_arm(seed, args):
    """Let Script 21 itself export {run_tag}_probs.npy and _labels_preds.npz, then load them."""
    run_tag = f"{ARMS['mnv3_noaug']['prefix']}_seed{seed}"
    script = str(SCRIPTS_DIR / "21_transfer_learning.py")
    cmd = [sys.executable, script, "--eval_only", "--split", SPLIT, "--seed", str(seed),
           "--checkpoint", str(CHECKPOINTS_DIR / f"{run_tag}.pth"), "--tag", run_tag]
    r = subprocess.run(cmd, cwd=str(BASE_DIR))
    if r.returncode != 0:
        raise RuntimeError(f"Script 21 --eval_only failed with exit code {r.returncode} (seed {seed})")
    p = ARRAYS_DIR / f"{run_tag}_probs.npy"
    l = ARRAYS_DIR / f"{run_tag}_labels_preds.npz"
    if not p.exists() or not l.exists():
        raise FileNotFoundError(f"Script 21 did not produce the arrays for {run_tag}")
    return np.load(l)["y_true"].astype(int), np.load(p)


# Aggregation and paired comparison

def baseline_per_seed(arm, y_csv):
    """
    Baseline metrics per seed, recomputed from the stored arrays so that the comparison
    uses exactly the same code as the new arm.
    """
    pref = ARMS[arm]["baseline_prefix"]
    out = {}
    for seed in SEEDS:
        p = ARRAYS_DIR / f"{pref}_seed{seed}_probs.npy"
        if not p.exists():
            continue
        out[seed] = metrics_from(y_csv, np.load(p))
    return out


def summarize(df, y_csv):
    """Per-arm mean/SD, and per-seed paired t-test, 90% CI and TOST against the baseline."""
    agg, comp = [], []
    for arm, g in df.groupby("arm"):
        row = {"arm": arm, "display": ARMS[arm]["display"], "n_seeds": len(g)}
        for m in ["icbhi", "se_off", "sp_off", "auc", "se_Wheeze", "se_Both"]:
            row[f"{m}_mean"], row[f"{m}_sd"] = mean_sd(g[m])
        agg.append(row)
        base = baseline_per_seed(arm, y_csv)
        common = [s for s in g["seed"].tolist() if s in base]
        if len(common) < 3:
            logger.warning(f"  {arm}: only {len(common)} paired seeds; test skipped")
            continue
        gi = g.set_index("seed")
        brow = {"arm": arm, "baseline": ARMS[arm]["baseline_display"], "n_seeds": len(common)}
        for m in ["icbhi", "auc"]:
            a = np.array([gi.loc[s, m] for s in common])
            b = np.array([base[s][m] for s in common])
            d = a - b
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = stats.t.ppf(0.95, len(d) - 1)
            brow[f"{m}_baseline_mean"] = float(b.mean())
            brow[f"{m}_new_mean"] = float(a.mean())
            brow[f"{m}_diff_mean"] = float(d.mean())
            brow[f"{m}_ci90_lo"] = float(d.mean() - t * se)
            brow[f"{m}_ci90_hi"] = float(d.mean() + t * se)
            brow[f"{m}_p_ttest"] = float(stats.ttest_rel(a, b).pvalue)
            # TOST with the same preregistered margin as Script 29
            pl = stats.ttest_1samp(d, -0.02, alternative="greater").pvalue
            pu = stats.ttest_1samp(d, +0.02, alternative="less").pvalue
            brow[f"{m}_p_tost"] = float(max(pl, pu))
            brow[f"{m}_equivalent"] = bool(max(pl, pu) < 0.05)
        comp.append(brow)
        logger.info(f"  {ARMS[arm]['display']}")
        for m in ["icbhi", "auc"]:
            logger.info(f"    {m:<6} {brow[f'{m}_baseline_mean']:.3f} → {brow[f'{m}_new_mean']:.3f}  "
                        f"Δ={brow[f'{m}_diff_mean']:+.4f}  CI90 [{brow[f'{m}_ci90_lo']:+.4f}, {brow[f'{m}_ci90_hi']:+.4f}]  "
                        f"t p={brow[f'{m}_p_ttest']:.3f}  TOST p={brow[f'{m}_p_tost']:.3f} → "
                        f"{'EQUIVALENT' if brow[f'{m}_equivalent'] else 'not equivalent'} (±0.02)")
    return agg, comp


def write_outputs(df, agg, comp, args):
    """Write the per-run CSV, the JSON summary and the Markdown tables."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "recipe_ablation_runs.csv", index=False)
    with open(OUT_DIR / "recipe_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "preset_arm_a": args.preset, "split": SPLIT, "seeds": SEEDS,
                   "aggregated": agg, "paired_vs_baseline": comp}, f, indent=2)
    L = ["# Training-recipe ablation (Script 34)\n",
         f"_{datetime.now().isoformat(timespec='seconds')} · Arm A preset `{args.preset}` · "
         f"Arm B without augmentation or mixup_\n",
         "\n| Configuration | ICBHI | Se off. | Sp off. | AUC | Se Wheeze | Se Both |",
         "|---|---|---|---|---|---|---|"]
    for r in agg:
        L.append(f"| {r['display']} | {r['icbhi_mean']:.3f} ± {r['icbhi_sd']:.3f} "
                 f"| {r['se_off_mean']:.3f} ± {r['se_off_sd']:.3f} | {r['sp_off_mean']:.3f} ± {r['sp_off_sd']:.3f} "
                 f"| {r['auc_mean']:.3f} ± {r['auc_sd']:.3f} | {r['se_Wheeze_mean']:.3f} ± {r['se_Wheeze_sd']:.3f} "
                 f"| {r['se_Both_mean']:.3f} ± {r['se_Both_sd']:.3f} |")
    if comp:
        L += ["\n**Per-seed paired comparison against the original recipe:**\n",
              "| Arm | Baseline | Metric | Baseline | New | Δ | 90 % CI | t p | TOST p | Equivalent |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in comp:
            for m in ["icbhi", "auc"]:
                L.append(f"| {c['arm']} | {c['baseline']} | {m} | {c[f'{m}_baseline_mean']:.3f} "
                         f"| {c[f'{m}_new_mean']:.3f} | {c[f'{m}_diff_mean']:+.4f} "
                         f"| [{c[f'{m}_ci90_lo']:+.4f}, {c[f'{m}_ci90_hi']:+.4f}] | {c[f'{m}_p_ttest']:.3f} "
                         f"| {c[f'{m}_p_tost']:.3f} | {'yes' if c[f'{m}_equivalent'] else 'no'} |")
    with open(OUT_DIR / "recipe_ablation_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    logger.info(f"  Summary: {OUT_DIR / 'recipe_ablation_summary.md'}")


def evaluate_all(args):
    """Evaluate every available checkpoint of the requested arms/seeds and write the outputs."""
    s23 = import_script("script23", "23_quantize_cnn_int8.py")
    s23.logger = logging.getLogger("script23")
    data = s23.get_dataloaders(split_strategy=SPLIT, batch_size=32)
    y_csv = pd.read_csv(SPLIT_DIR / "test_cycles.csv")["cycle_class"].map(CLASS_MAP).values.astype(int)
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    arms = list(ARMS.keys()) if args.arm is None else [args.arm]
    seeds = SEEDS if args.seed is None else [args.seed]
    for arm in arms:
        for seed in seeds:
            run_tag = f"{ARMS[arm]['prefix']}_seed{seed}"
            ckpt = CHECKPOINTS_DIR / f"{run_tag}.pth"
            if not ckpt.exists():
                logger.warning(f"  missing {ckpt.name}; skipped")
                continue
            try:
                if arm == "cnn_aug":
                    y, probs = eval_cnn_arm(seed, s23, data, y_csv)
                else:
                    # Script 21 already exported the arrays during training or with
                    # --eval_only; the subprocess is relaunched only if they are missing.
                    pnpy = ARRAYS_DIR / f"{run_tag}_probs.npy"
                    lnpz = ARRAYS_DIR / f"{run_tag}_labels_preds.npz"
                    if pnpy.exists() and lnpz.exists():
                        y, probs = np.load(lnpz)["y_true"].astype(int), np.load(pnpy)
                    else:
                        y, probs = eval_mnv3_arm(seed, args)
                    if not np.array_equal(y, y_csv):
                        raise ValueError(f"{run_tag}: test set not aligned with test_cycles.csv")
            except Exception as e:
                logger.error(f"  {run_tag}: evaluation failed ({e})")
                continue
            m = metrics_from(y_csv, probs)
            row = {"arm": arm, "seed": seed, "icbhi": m["icbhi"], "se_off": m["se_off"],
                   "sp_off": m["sp_off"], "bin_se": m["bin_se"], "bin_sp": m["bin_sp"], "auc": m["auc"]}
            for c, v in zip(CLASS_NAMES, m["per_class_se"]):
                row[f"se_{c}"] = v
            rows.append(row)
            logger.info(f"  {run_tag:<52} ICBHI {m['icbhi']:.4f} (Se {m['se_off']:.3f}/Sp {m['sp_off']:.3f})  "
                        f"AUC {m['auc']:.4f}")
    if not rows:
        logger.warning("  No runs to evaluate.")
        return
    df = pd.DataFrame(rows)
    logger.info("\nPaired comparison against the original recipe")
    agg, comp = summarize(df, y_csv)
    write_outputs(df, agg, comp, args)


def parse_args():
    p = argparse.ArgumentParser(description="Script 34: training-recipe ablation")
    p.add_argument("--arm", choices=list(ARMS.keys()), default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all", action="store_true", help="2 arms x 5 seeds")
    p.add_argument("--evaluate", action="store_true", help="only evaluate existing checkpoints")
    p.add_argument("--preset", default="moderate",
                   help="augment_preset for Arm A (Script 21 used 'moderate')")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    log_file = setup_logging()
    logger.info("=" * 78)
    logger.info("SCRIPT 34: training-recipe ablation")
    logger.info("=" * 78)
    logger.info(f"BASE_DIR: {BASE_DIR}")

    if args.evaluate:
        evaluate_all(args)
        logger.info(f"Log: {log_file}")
        logger.info("Done.")
        return

    jobs = [(a, s) for a in ARMS for s in SEEDS] if args.all else [(args.arm, args.seed)]
    if any(a is None or s is None for a, s in jobs):
        raise SystemExit("Specify --arm and --seed, or --all, or --evaluate")

    logger.info(f"Arm A preset: {args.preset}   |   Arm B: augment=none, mixup=0")
    for i, (arm, seed) in enumerate(jobs, 1):
        run_tag = f"{ARMS[arm]['prefix']}_seed{seed}"
        logger.info(f"\n[{i}/{len(jobs)}] {ARMS[arm]['display']}, seed {seed}")
        if args.skip_existing and (CHECKPOINTS_DIR / f"{run_tag}.pth").exists():
            logger.info(f"  {run_tag}.pth exists; skipped (--skip_existing)")
            continue
        try:
            if arm == "cnn_aug":
                train_cnn_aug(seed, args)
            else:
                train_mnv3_noaug(seed, args)
        except Exception as e:
            logger.error(f"  run failed: {e}")
            continue

    logger.info("\nEvaluation")
    evaluate_all(argparse.Namespace(arm=None if args.all else args.arm,
                                    seed=None if args.all else args.seed,
                                    preset=args.preset, epochs=args.epochs))
    logger.info(f"Log: {log_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
