#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 14: Multi-seed training and evaluation of the CNN and CfC models.

Retrains each model (CNN-RespiratoryNet from script 09; CfC-Transposed, CfC-Hybrid and
the hybrid temporal-stride variants from script 11a) with several seeds, so that results
can be reported as mean +/- std and compared with paired tests (paired t-test and
Wilcoxon signed-rank). Each run is an independent subprocess and progress is stored in a
JSON file, so an interrupted run resumes where it stopped. All checkpoints are then
re-evaluated on the test set with the same DataLoader. The ICBHI score computed here is
the macro one-vs-rest average; the official score is recomputed by scripts 17 and 18.

Inputs:  data/splits/<split>/, data/features/mel_spectrograms/ (via script 08),
         scripts 09_cnn_training.py, 11a_cfc_mel_v7.py, 08_dataset_dataloader.py, torch_cfc.py
Outputs: models/checkpoints/<model>[_official]_seed<S>.pth,
         <model>[_official]_seed<S>_log.txt (in the base directory),
         results/metrics/multiseed/ (progress JSON, results JSON, per-run and aggregated CSV,
         Markdown summary), results/figures/multiseed_comparison.png
Usage:
    python 14_multiseed_runs.py [--seeds 42 123 7] [--models cnn transposed hybrid]
                                [--split strategy1_official] [--evaluate-only] [--reset-progress]
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime
from collections import defaultdict, OrderedDict

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent if (SCRIPTS_DIR.parent / "data").exists() else SCRIPTS_DIR
if not (BASE_DIR / "data").exists():
    BASE_DIR = SCRIPTS_DIR

CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
METRICS_DIR = BASE_DIR / "results" / "metrics"
FIGURES_DIR = BASE_DIR / "results" / "figures"
MULTISEED_DIR = METRICS_DIR / "multiseed"

for d in [MULTISEED_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Per model: script to run, launch method ("subprocess_with_seed_flag" for 11a,
# "wrapped_subprocess" for 09, which has no --seed flag), extra CLI arguments, and the
# checkpoint/log names the script writes (renamed per seed after each run).
MODELS = OrderedDict([
    ("cnn", {
        "display_name": "CNN-RespiratoryNet",
        "script": "09_cnn_training.py",
        "method": "wrapped_subprocess",
        "extra_args": [],
        "default_checkpoint": "best_model.pth",
        "log_filename": "best_model_log.txt",
        "color": "#1f77b4",
    }),
    ("transposed", {
        "display_name": "CfC-Transposed",
        "script": "11a_cfc_mel_v7.py",
        "method": "subprocess_with_seed_flag",
        "extra_args": ["--architecture", "transposed"],
        "default_checkpoint": "best_cfc_v7_mel_transposed.pth",
        "log_filename": "cfc_v7_mel_transposed_log.txt",
        "color": "#2ca02c",
    }),
    ("hybrid", {
        "display_name": "CfC-Hybrid",
        "script": "11a_cfc_mel_v7.py",
        "method": "subprocess_with_seed_flag",
        "extra_args": ["--architecture", "hybrid"],
        "default_checkpoint": "best_cfc_v7_mel_hybrid.pth",
        "log_filename": "cfc_v7_mel_hybrid_log.txt",
        "color": "#d62728",
    }),
    # Hybrid variants with less aggressive temporal pooling. Frequency is still pooled
    # /2 per block (cfc_in=1024, same parameter count); only the temporal stride, and
    # hence seq_length, changes. Checkpoint and log names must match the run_name built
    # by script 11a: cfc_v7_mel_hybrid_t{strides} -> best_<run_name>.pth, <run_name>_log.txt
    ("hybrid_t122", {
        "display_name": "CfC-Hybrid seq62",
        "script": "11a_cfc_mel_v7.py",
        "method": "subprocess_with_seed_flag",
        "extra_args": ["--architecture", "hybrid", "--temporal_strides", "1,2,2"],
        "default_checkpoint": "best_cfc_v7_mel_hybrid_t122.pth",
        "log_filename": "cfc_v7_mel_hybrid_t122_log.txt",
        "color": "#ff7f0e",
    }),
    ("hybrid_t112", {
        "display_name": "CfC-Hybrid seq125",
        "script": "11a_cfc_mel_v7.py",
        "method": "subprocess_with_seed_flag",
        "extra_args": ["--architecture", "hybrid", "--temporal_strides", "1,1,2"],
        "default_checkpoint": "best_cfc_v7_mel_hybrid_t112.pth",
        "log_filename": "cfc_v7_mel_hybrid_t112_log.txt",
        "color": "#9467bd",
    }),
    ("hybrid_t111", {
        "display_name": "CfC-Hybrid seq251",
        "script": "11a_cfc_mel_v7.py",
        "method": "subprocess_with_seed_flag",
        "extra_args": ["--architecture", "hybrid", "--temporal_strides", "1,1,1"],
        "default_checkpoint": "best_cfc_v7_mel_hybrid_t111.pth",
        "log_filename": "cfc_v7_mel_hybrid_t111_log.txt",
        "color": "#8c564b",
    }),
])

PROGRESS_FILE = MULTISEED_DIR / "multiseed_progress.json"

# Split, set in main() from --split. strategy2_custom has zero patient leakage;
# strategy1_official is the ICBHI 60/40 split (comparable with the literature, with a
# minor leakage of 2 patients: 156, 218). The tag is appended to checkpoints, logs,
# run keys and the progress file so that official runs never overwrite strategy2 runs.
SPLIT = "strategy2_custom"
SPLIT_TAG = ""   # "" for custom, "_official" for official


def _split_tag(split: str) -> str:
    return "_official" if split == "strategy1_official" else ""


def _insert_tag(filename: str, tag: str) -> str:
    """
    Insert the tag before the extension or the "_log.txt" suffix.

    best_cfc_v7_mel_hybrid.pth + _official -> best_cfc_v7_mel_hybrid_official.pth
    cfc_v7_mel_hybrid_log.txt  + _official -> cfc_v7_mel_hybrid_official_log.txt
    An empty tag returns the name unchanged.
    """
    if not tag:
        return filename
    if filename.endswith("_log.txt"):
        return filename[: -len("_log.txt")] + tag + "_log.txt"
    stem, dot, ext = filename.rpartition(".")
    return f"{stem}{tag}.{ext}" if dot else filename + tag


def load_progress() -> dict:
    """Load the progress state from JSON, or return an empty one."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {PROGRESS_FILE}: {e}. Starting from scratch.")
    return {"completed": [], "failed": [], "started_at": datetime.now().isoformat()}


def save_progress(progress: dict):
    """Save the progress state to JSON."""
    progress["updated_at"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def run_key(model: str, seed: int) -> str:
    """Unique identifier of a model x seed run (includes the split tag)."""
    return f"{model}{SPLIT_TAG}_seed{seed}"


def run_subprocess_with_seed_flag(model_key: str, seed: int, model_info: dict) -> bool:
    """Run a script that accepts --seed (script 11a). Return True on success."""
    script_path = SCRIPTS_DIR / model_info["script"]
    cmd = [
        sys.executable,
        str(script_path),
        *model_info["extra_args"],
        "--seed", str(seed),
        "--split", SPLIT,
    ]
    print(f"\n  Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPTS_DIR),
            check=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] subprocess failed with code {e.returncode}")
        return False
    except Exception as e:
        print(f"  [ERROR] Exception during subprocess: {e}")
        return False


def run_wrapped_subprocess(model_key: str, seed: int, model_info: dict) -> bool:
    """
    Run script 09, which has no seed/split CLI, through an inline wrapper.

    The wrapper imports the module, injects the split into TRAINING_CONFIG and calls
    main() with the canonical baseline configuration (plain cross-entropy, weighted
    sampler, checkpoint selection by val_loss, tag "best_model"). Without these flags,
    script 09 would use different defaults and the model would not be comparable with
    the baseline. Return True on success.
    """
    script_path = SCRIPTS_DIR / model_info["script"]

    wrapper_code = f"""
import sys
import importlib.util
from pathlib import Path

script_path = Path(r'{script_path}')
scripts_dir = script_path.parent
sys.path.insert(0, str(scripts_dir))

spec = importlib.util.spec_from_file_location("script09", str(script_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Script 09 has no split flag: the split is injected through its config.
if hasattr(mod, 'TRAINING_CONFIG'):
    mod.TRAINING_CONFIG['split_strategy'] = '{SPLIT}'
else:
    print("[WARN] script 09 does not expose TRAINING_CONFIG; its default split will be used")

# Canonical baseline recipe: plain CE + weighted sampler + selection by val_loss,
# passed through the flags of script 09. --tag best_model makes script 09 write
# best_model.pth and best_model_log.txt, which rename_artifacts then renames per seed.
sys.argv = [
    '09_cnn_training.py',
    '--seed', '{seed}',
    '--loss', 'ce',
    '--sampler', 'weighted',
    '--select_metric', 'val_loss',
    '--tag', 'best_model',
]
print(">>> CNN canonical config: loss=ce sampler=weighted select=val_loss "
      "seed={seed} split={SPLIT} tag=best_model")

mod.main()
"""
    print(f"\n  Wrapper subprocess for {model_info['script']} with seed={seed}")
    try:
        result = subprocess.run(
            [sys.executable, "-c", wrapper_code],
            cwd=str(BASE_DIR),
            check=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] wrapper subprocess failed with code {e.returncode}")
        return False
    except Exception as e:
        print(f"  [ERROR] Exception in wrapper: {e}")
        return False


def execute_run(model_key: str, seed: int, model_info: dict) -> bool:
    """Dispatch to the launch method of the model."""
    method = model_info["method"]
    if method == "subprocess_with_seed_flag":
        return run_subprocess_with_seed_flag(model_key, seed, model_info)
    elif method == "wrapped_subprocess":
        return run_wrapped_subprocess(model_key, seed, model_info)
    else:
        raise ValueError(f"Unknown method: {method}")


def rename_artifacts(model_key: str, seed: int, model_info: dict) -> dict:
    """Move the generated checkpoint and log to per-seed names; return the final paths."""
    artifacts = {}

    # Script 11a appends the split tag to its run_name; script 09 always writes
    # best_model.pth.
    src_tag = SPLIT_TAG if model_info["method"] == "subprocess_with_seed_flag" else ""

    src_ckpt = CHECKPOINTS_DIR / _insert_tag(model_info["default_checkpoint"], src_tag)
    dst_ckpt = CHECKPOINTS_DIR / f"{model_key}{SPLIT_TAG}_seed{seed}.pth"
    if src_ckpt.exists():
        shutil.move(str(src_ckpt), str(dst_ckpt))
        artifacts["checkpoint"] = str(dst_ckpt)
        print(f"  [OK] Checkpoint: {src_ckpt.name} -> {dst_ckpt.name}")
    else:
        print(f"  [WARN] Expected checkpoint does not exist: {src_ckpt}")
        artifacts["checkpoint"] = None

    src_log = BASE_DIR / _insert_tag(model_info["log_filename"], src_tag)
    dst_log = BASE_DIR / f"{model_key}{SPLIT_TAG}_seed{seed}_log.txt"
    if src_log.exists():
        shutil.move(str(src_log), str(dst_log))
        artifacts["log"] = str(dst_log)
    else:
        artifacts["log"] = None

    return artifacts


def run_training_phase(seeds: list, model_keys: list, progress: dict) -> dict:
    """Train every pending model x seed combination, skipping runs already completed."""
    artifacts_by_run = {}
    total_runs = len(seeds) * len(model_keys)
    completed_in_progress = set(progress.get("completed", []))

    print(f"\n{'='*70}")
    print(f"PHASE 1: MULTI-SEED TRAINING")
    print(f"{'='*70}")
    print(f"  Models:          {model_keys}")
    print(f"  Seeds:           {seeds}")
    print(f"  Total runs:      {total_runs}")
    print(f"  Already done:    {len(completed_in_progress)}")
    print(f"  Pending:         {total_runs - len(completed_in_progress)}")
    print(f"{'='*70}")

    run_idx = 0
    for model_key in model_keys:
        model_info = MODELS[model_key]
        for seed in seeds:
            run_idx += 1
            key = run_key(model_key, seed)

            print(f"\n[{run_idx}/{total_runs}] {model_info['display_name']} - seed {seed}")

            if key in completed_in_progress:
                print(f"  Already completed (per progress.json). Skipping.")
                # A run marked as completed whose checkpoint is missing is re-executed.
                expected_ckpt = CHECKPOINTS_DIR / f"{model_key}{SPLIT_TAG}_seed{seed}.pth"
                if expected_ckpt.exists():
                    artifacts_by_run[key] = {"checkpoint": str(expected_ckpt)}
                else:
                    print(f"  [WARN] Marked as completed but checkpoint does not exist: {expected_ckpt}")
                    print(f"  -> Re-running...")
                    completed_in_progress.discard(key)

            if key in completed_in_progress:
                continue

            t0 = time.time()
            ok = execute_run(model_key, seed, model_info)
            elapsed = time.time() - t0

            if not ok:
                print(f"  [FAIL] Run failed after {elapsed/60:.1f} min")
                if key not in progress.get("failed", []):
                    progress.setdefault("failed", []).append(key)
                save_progress(progress)
                continue

            print(f"  [OK] Run completed in {elapsed/60:.1f} min")

            artifacts = rename_artifacts(model_key, seed, model_info)
            artifacts_by_run[key] = artifacts

            progress.setdefault("completed", []).append(key)
            if key in progress.get("failed", []):
                progress["failed"].remove(key)
            save_progress(progress)

    print(f"\n{'='*70}")
    print(f"PHASE 1 COMPLETED")
    print(f"  Completed: {len(progress.get('completed', []))}/{total_runs}")
    print(f"  Failed:    {len(progress.get('failed', []))}")
    print(f"{'='*70}")

    return artifacts_by_run


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataloaders():
    """Create the script 08 DataLoaders (same settings as the training runs, no augmentation)."""
    script08 = import_module_from_path(
        "script08", SCRIPTS_DIR / "08_dataset_dataloader.py"
    )
    loaders = script08.create_dataloaders(
        strategy=SPLIT,
        class_mode="4class",
        augment_preset="none",
        batch_size=32,
        num_workers=0,
        use_weighted_sampling=False,
    )
    return loaders


def evaluate_checkpoint(model_key: str, seed: int, model_info: dict, test_loader) -> dict:
    """
    Load the checkpoint of one run and evaluate it on the test set.

    Returns a dict with macro ICBHI, Se, Sp, accuracy, parameter count and per-class
    one-vs-rest counts, or None if the checkpoint or model class is missing.
    """
    import torch
    import torch.nn as nn

    ckpt_path = CHECKPOINTS_DIR / f"{model_key}{SPLIT_TAG}_seed{seed}.pth"
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint does not exist: {ckpt_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if model_key == "cnn":
        script09 = import_module_from_path(
            "script09", SCRIPTS_DIR / "09_cnn_training.py"
        )
        config = ckpt.get("config", script09.TRAINING_CONFIG)
        model_class_name = None
        for name in ["RespiratoryNet", "CNN", "BaselineCNN"]:
            if hasattr(script09, name):
                model_class_name = name
                break
        if model_class_name is None:
            print(f"  [ERROR] CNN model class not found in script09")
            return None
        ModelClass = getattr(script09, model_class_name)
        model = ModelClass(config=config) if "config" in ModelClass.__init__.__code__.co_varnames \
                else ModelClass()
    elif model_key.startswith("transposed") or model_key.startswith("hybrid"):
        # The base architecture comes from the model_key prefix (this also covers the
        # hybrid_t* variants). The actual geometry (temporal_strides -> seq_length) is
        # stored in the checkpoint config, so CfCMelHybridV6 is rebuilt correctly for any
        # sequence length.
        arch = "transposed" if model_key.startswith("transposed") else "hybrid"
        # Replace argv so that importing 11a does not trip its argparse.
        old_argv = sys.argv
        sys.argv = ["dummy", "--architecture", arch]
        try:
            script11a = import_module_from_path(
                "script11a", SCRIPTS_DIR / "11a_cfc_mel_v7.py"
            )
        finally:
            sys.argv = old_argv

        config = ckpt.get("config", {})
        if arch == "transposed":
            model = script11a.CfCMelTransposedV6(n_classes=4, config=config)
        else:
            model = script11a.CfCMelHybridV6(n_classes=4, config=config)
    else:
        print(f"  [ERROR] Unknown model_key: {model_key}")
        return None

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    n_classes = 4
    all_preds, all_labels = [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            outputs = model(specs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Macro one-vs-rest Se/Sp averaged over the 4 classes (not the official ICBHI score).
    sensitivities, specificities = [], []
    per_class = []
    class_names = ["Normal", "Crackle", "Wheeze", "Both"]
    for c in range(n_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(se)
        specificities.append(sp)
        per_class.append({
            "class": class_names[c],
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "sensitivity": float(se), "specificity": float(sp),
        })

    se_avg = float(np.mean(sensitivities))
    sp_avg = float(np.mean(specificities))
    icbhi = (se_avg + sp_avg) / 2.0
    acc = float(np.mean(y_true == y_pred))

    params = sum(p.numel() for p in model.parameters())

    return {
        "model_key": model_key,
        "model_display": model_info["display_name"],
        "seed": seed,
        "icbhi": icbhi,
        "sensitivity": se_avg,
        "specificity": sp_avg,
        "accuracy": acc,
        "params": params,
        "per_class": per_class,
    }


def run_evaluation_phase(seeds: list, model_keys: list) -> dict:
    """Evaluate every available checkpoint on the test set."""
    print(f"\n{'='*70}")
    print(f"PHASE 2: CONSOLIDATED TEST SET EVALUATION")
    print(f"{'='*70}")

    print("\n[Loading test DataLoader]...")
    loaders = load_dataloaders()
    test_loader = loaders["test_loader"]
    print(f"  Test samples: {len(test_loader.dataset)}")

    results_per_run = {}
    for model_key in model_keys:
        model_info = MODELS[model_key]
        for seed in seeds:
            key = run_key(model_key, seed)
            print(f"\n  Evaluating {model_info['display_name']} seed={seed}...")
            metrics = evaluate_checkpoint(model_key, seed, model_info, test_loader)
            if metrics is None:
                continue
            results_per_run[key] = metrics
            print(f"    ICBHI={metrics['icbhi']:.4f}  "
                  f"Se={metrics['sensitivity']:.3f}  "
                  f"Sp={metrics['specificity']:.3f}  "
                  f"Acc={metrics['accuracy']*100:.2f}%  "
                  f"Params={metrics['params']:,}")

    return results_per_run


def aggregate_results(results_per_run: dict, model_keys: list) -> dict:
    """
    Compute mean +/- std (ddof=1) per model and paired tests over common seeds.

    Paired t-test and Wilcoxon signed-rank on ICBHI: CNN vs CfC-Hybrid and
    CfC-Transposed vs CfC-Hybrid, when both models are present with at least 2 seeds.
    """
    by_model = defaultdict(list)
    for key, m in results_per_run.items():
        by_model[m["model_key"]].append(m)

    aggregated = {}
    for mk in model_keys:
        runs = by_model.get(mk, [])
        if not runs:
            continue
        icbhis = [r["icbhi"] for r in runs]
        ses = [r["sensitivity"] for r in runs]
        sps = [r["specificity"] for r in runs]
        accs = [r["accuracy"] for r in runs]
        aggregated[mk] = {
            "display_name": MODELS[mk]["display_name"],
            "n_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "icbhi_mean": float(np.mean(icbhis)),
            "icbhi_std": float(np.std(icbhis, ddof=1)) if len(icbhis) > 1 else 0.0,
            "icbhi_per_seed": icbhis,
            "se_mean": float(np.mean(ses)),
            "se_std": float(np.std(ses, ddof=1)) if len(ses) > 1 else 0.0,
            "sp_mean": float(np.mean(sps)),
            "sp_std": float(np.std(sps, ddof=1)) if len(sps) > 1 else 0.0,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "params": runs[0]["params"],
            "per_class_means": _aggregate_per_class(runs),
        }

    stats_tests = {}
    if "cnn" in aggregated and "hybrid" in aggregated:
        cnn_runs = sorted(by_model["cnn"], key=lambda r: r["seed"])
        hyb_runs = sorted(by_model["hybrid"], key=lambda r: r["seed"])
        # Pair runs by seed.
        cnn_seeds = [r["seed"] for r in cnn_runs]
        hyb_seeds = [r["seed"] for r in hyb_runs]
        common = sorted(set(cnn_seeds) & set(hyb_seeds))
        if len(common) >= 2:
            cnn_paired = [next(r["icbhi"] for r in cnn_runs if r["seed"] == s) for s in common]
            hyb_paired = [next(r["icbhi"] for r in hyb_runs if r["seed"] == s) for s in common]
            stats_tests["cnn_vs_hybrid"] = _stat_tests(cnn_paired, hyb_paired, common,
                                                       "CNN", "CfC-Hybrid")

    if "transposed" in aggregated and "hybrid" in aggregated:
        trp = sorted(by_model["transposed"], key=lambda r: r["seed"])
        hyb = sorted(by_model["hybrid"], key=lambda r: r["seed"])
        common = sorted(set(r["seed"] for r in trp) & set(r["seed"] for r in hyb))
        if len(common) >= 2:
            trp_paired = [next(r["icbhi"] for r in trp if r["seed"] == s) for s in common]
            hyb_paired = [next(r["icbhi"] for r in hyb if r["seed"] == s) for s in common]
            stats_tests["transposed_vs_hybrid"] = _stat_tests(
                trp_paired, hyb_paired, common, "CfC-Transposed", "CfC-Hybrid")

    return {"per_model": aggregated, "stats_tests": stats_tests}


def _aggregate_per_class(runs: list) -> list:
    """Mean and std of per-class Se/Sp across seeds."""
    n_classes = len(runs[0]["per_class"])
    out = []
    for c in range(n_classes):
        ses = [r["per_class"][c]["sensitivity"] for r in runs]
        sps = [r["per_class"][c]["specificity"] for r in runs]
        out.append({
            "class": runs[0]["per_class"][c]["class"],
            "se_mean": float(np.mean(ses)),
            "se_std": float(np.std(ses, ddof=1)) if len(ses) > 1 else 0.0,
            "sp_mean": float(np.mean(sps)),
            "sp_std": float(np.std(sps, ddof=1)) if len(sps) > 1 else 0.0,
        })
    return out


def _stat_tests(a: list, b: list, seeds: list, name_a: str, name_b: str) -> dict:
    """Paired t-test and Wilcoxon signed-rank between two lists of ICBHI scores."""
    out = {
        "name_a": name_a, "name_b": name_b,
        "seeds": seeds, "values_a": a, "values_b": b,
        "diff_mean": float(np.mean(np.array(b) - np.array(a))),
    }
    try:
        from scipy import stats
        t_stat, t_p = stats.ttest_rel(a, b)
        out["paired_t"] = {"t": float(t_stat), "p_value": float(t_p)}
        # Wilcoxon is undefined when all differences are zero.
        diffs = np.array(b) - np.array(a)
        if np.any(diffs != 0):
            try:
                w_stat, w_p = stats.wilcoxon(a, b)
                out["wilcoxon"] = {"W": float(w_stat), "p_value": float(w_p)}
            except Exception as e:
                out["wilcoxon"] = {"error": str(e)}
        else:
            out["wilcoxon"] = {"note": "all differences are zero"}
    except ImportError:
        out["paired_t"] = {"error": "scipy not installed; install with: pip install scipy"}
        out["wilcoxon"] = {"error": "scipy not installed"}
    return out


def write_outputs(results_per_run: dict, aggregated: dict, model_keys: list):
    """Write the results JSON, per-run and aggregated CSV, Markdown summary and figure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = MULTISEED_DIR / "multiseed_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "per_run": results_per_run,
            "aggregated": aggregated,
        }, f, indent=2)
    print(f"\n  Consolidated JSON: {json_path}")

    csv_path = MULTISEED_DIR / "multiseed_runs.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,seed,icbhi,sensitivity,specificity,accuracy,params,"
                "se_normal,se_crackle,se_wheeze,se_both\n")
        for key, m in results_per_run.items():
            pc = {r["class"]: r["sensitivity"] for r in m["per_class"]}
            f.write(f"{m['model_display']},{m['seed']},{m['icbhi']:.4f},"
                    f"{m['sensitivity']:.4f},{m['specificity']:.4f},"
                    f"{m['accuracy']:.4f},{m['params']},"
                    f"{pc.get('Normal',0):.3f},{pc.get('Crackle',0):.3f},"
                    f"{pc.get('Wheeze',0):.3f},{pc.get('Both',0):.3f}\n")
    print(f"  Per-run CSV:       {csv_path}")

    csv_agg = MULTISEED_DIR / "multiseed_aggregated.csv"
    with open(csv_agg, "w", encoding="utf-8") as f:
        f.write("model,n_seeds,icbhi_mean,icbhi_std,se_mean,se_std,sp_mean,sp_std,"
                "acc_mean,acc_std,params\n")
        for mk in model_keys:
            if mk not in aggregated["per_model"]:
                continue
            a = aggregated["per_model"][mk]
            f.write(f"{a['display_name']},{a['n_seeds']},"
                    f"{a['icbhi_mean']:.4f},{a['icbhi_std']:.4f},"
                    f"{a['se_mean']:.4f},{a['se_std']:.4f},"
                    f"{a['sp_mean']:.4f},{a['sp_std']:.4f},"
                    f"{a['acc_mean']:.4f},{a['acc_std']:.4f},"
                    f"{a['params']}\n")
    print(f"  Aggregated CSV:    {csv_agg}")

    md_path = MULTISEED_DIR / "multiseed_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Multi-Seed Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Main table (test set)\n\n")
        f.write("| Model | Params | n seeds | ICBHI | Sensitivity | Specificity | Accuracy |\n")
        f.write("|---|---:|---:|---|---|---|---|\n")
        for mk in model_keys:
            if mk not in aggregated["per_model"]:
                continue
            a = aggregated["per_model"][mk]
            f.write(f"| **{a['display_name']}** | {a['params']:,} | {a['n_seeds']} | "
                    f"{a['icbhi_mean']:.4f} ± {a['icbhi_std']:.4f} | "
                    f"{a['se_mean']:.4f} ± {a['se_std']:.4f} | "
                    f"{a['sp_mean']:.4f} ± {a['sp_std']:.4f} | "
                    f"{a['acc_mean']*100:.2f}% ± {a['acc_std']*100:.2f}% |\n")

        f.write("\n## Per-class sensitivity\n\n")
        f.write("| Model | Normal | Crackle | Wheeze | Both |\n")
        f.write("|---|---|---|---|---|\n")
        for mk in model_keys:
            if mk not in aggregated["per_model"]:
                continue
            a = aggregated["per_model"][mk]
            pc = {p["class"]: p for p in a["per_class_means"]}
            f.write(f"| {a['display_name']} | ")
            for cls in ["Normal", "Crackle", "Wheeze", "Both"]:
                p = pc.get(cls, {})
                f.write(f"{p.get('se_mean', 0):.3f} ± {p.get('se_std', 0):.3f} | ")
            f.write("\n")

        f.write("\n## Statistical tests\n\n")
        for test_key, test in aggregated.get("stats_tests", {}).items():
            f.write(f"### {test['name_a']} vs {test['name_b']}\n\n")
            f.write(f"- Seeds: {test['seeds']}\n")
            f.write(f"- ICBHI {test['name_a']}: {test['values_a']}\n")
            f.write(f"- ICBHI {test['name_b']}: {test['values_b']}\n")
            f.write(f"- Mean diff (b - a): {test['diff_mean']:+.4f}\n")
            pt = test.get("paired_t", {})
            if "p_value" in pt:
                f.write(f"- **Paired t-test:** t={pt['t']:.3f}, "
                        f"p={pt['p_value']:.4f} "
                        f"({'significant' if pt['p_value'] < 0.05 else 'NOT significant'} at α=0.05)\n")
            wx = test.get("wilcoxon", {})
            if "p_value" in wx:
                f.write(f"- **Wilcoxon signed-rank:** W={wx['W']:.3f}, "
                        f"p={wx['p_value']:.4f}\n")
            f.write("\n")

    print(f"  Markdown:          {md_path}")

    try:
        fig_path = _make_figure(aggregated, model_keys)
        print(f"  Figure:            {fig_path}")
    except Exception as e:
        print(f"  [WARN] Could not generate figure: {e}")


def _make_figure(aggregated: dict, model_keys: list) -> Path:
    """Bar chart with error bars (std across seeds) in 3 panels: ICBHI, Se, Sp."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models_present = [mk for mk in model_keys if mk in aggregated["per_model"]]
    if not models_present:
        raise ValueError("No models to plot")

    names = [aggregated["per_model"][mk]["display_name"] for mk in models_present]
    colors = [MODELS[mk]["color"] for mk in models_present]

    icbhi_m = [aggregated["per_model"][mk]["icbhi_mean"] for mk in models_present]
    icbhi_s = [aggregated["per_model"][mk]["icbhi_std"] for mk in models_present]
    se_m = [aggregated["per_model"][mk]["se_mean"] for mk in models_present]
    se_s = [aggregated["per_model"][mk]["se_std"] for mk in models_present]
    sp_m = [aggregated["per_model"][mk]["sp_mean"] for mk in models_present]
    sp_s = [aggregated["per_model"][mk]["sp_std"] for mk in models_present]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(names))

    for ax, means, stds, title, ylabel, ylim_top in [
        (axes[0], icbhi_m, icbhi_s, "ICBHI Score", "ICBHI", 0.7),
        (axes[1], se_m, se_s, "Sensitivity (macro)", "Se", 0.5),
        (axes[2], sp_m, sp_s, "Specificity (macro)", "Sp", 1.0),
    ]:
        bars = ax.bar(x, means, yerr=stds, capsize=8,
                      color=colors, edgecolor="black", linewidth=0.8,
                      error_kw={"linewidth": 1.5, "ecolor": "black"})
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + s + 0.02 * ylim_top,
                    f"{m:.3f}\n±{s:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim([0, ylim_top])

    n_seeds = aggregated["per_model"][models_present[0]]["n_seeds"]
    plt.suptitle(f"Multi-seed comparison (n={n_seeds} seeds, test set)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "multiseed_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-seed training and evaluation of the CNN and CfC models"
    )
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 7],
                        help="List of seeds. Default: 42 123 7")
    # The hybrid_t* variants are opt-in through --models; the default keeps the three
    # base models.
    _default_models = ["cnn", "transposed", "hybrid"]
    parser.add_argument("--models", type=str, nargs="+",
                        default=_default_models,
                        choices=list(MODELS.keys()),
                        help=f"Subset of models. Default: {_default_models}. "
                             f"Temporal-stride variants: hybrid_t122 (seq62), "
                             f"hybrid_t112 (seq125), hybrid_t111 (seq251).")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip training and only evaluate existing checkpoints")
    parser.add_argument("--reset-progress", action="store_true",
                        help="Delete progress.json and start from scratch")
    parser.add_argument("--split", type=str, default="strategy2_custom",
                        choices=["strategy1_official", "strategy2_custom"],
                        help="Split strategy. Default strategy2_custom (zero "
                             "leakage). strategy1_official = ICBHI 60/40 split, "
                             "comparable with the literature (minor leakage of 2 patients). "
                             "Official runs use checkpoints with the _official suffix "
                             "and do not overwrite strategy2 runs.")
    return parser.parse_args()


def main():
    args = parse_args()

    # The split affects commands, checkpoint/log names, run keys, progress.json and the
    # evaluation DataLoader.
    global SPLIT, SPLIT_TAG
    SPLIT = args.split
    SPLIT_TAG = _split_tag(args.split)

    print("=" * 70)
    print("Script 14: Multi-Seed Runs")
    print("=" * 70)
    print(f"  Seeds:  {args.seeds}")
    print(f"  Models:  {args.models}")
    print(f"  Split:   {SPLIT}" + (f"  (checkpoint tag: '{SPLIT_TAG}')" if SPLIT_TAG else ""))
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  Output:   {MULTISEED_DIR}")
    print("=" * 70)

    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print(f"  progress.json reset.")

    progress = load_progress()

    if not args.evaluate_only:
        run_training_phase(args.seeds, args.models, progress)
    else:
        print("\n[--evaluate-only mode: skipping training]")

    results_per_run = run_evaluation_phase(args.seeds, args.models)

    if not results_per_run:
        print("\n[ERROR] No results to aggregate. Are checkpoints missing?")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"PHASE 3: STATISTICAL AGGREGATION")
    print(f"{'='*70}")
    aggregated = aggregate_results(results_per_run, args.models)

    print("\nSummary per model:")
    for mk in args.models:
        if mk not in aggregated["per_model"]:
            continue
        a = aggregated["per_model"][mk]
        print(f"  {a['display_name']:25s} "
              f"ICBHI = {a['icbhi_mean']:.4f} ± {a['icbhi_std']:.4f} "
              f"(n={a['n_seeds']})")

    if aggregated.get("stats_tests"):
        print("\nStatistical tests:")
        for tk, t in aggregated["stats_tests"].items():
            pt = t.get("paired_t", {})
            if "p_value" in pt:
                sig = "*" if pt["p_value"] < 0.05 else "ns"
                print(f"  {t['name_a']} vs {t['name_b']}: "
                      f"diff={t['diff_mean']:+.4f}, "
                      f"paired t p={pt['p_value']:.4f} [{sig}]")

    print(f"\n{'='*70}")
    print(f"PHASE 4: SAVING RESULTS")
    print(f"{'='*70}")
    write_outputs(results_per_run, aggregated, args.models)

    print(f"\n{'='*70}")
    print(f"ALL DONE")
    print(f"{'='*70}")
    print(f"  Results in: {MULTISEED_DIR}")
    print(f"  Figure in:  {FIGURES_DIR}/multiseed_comparison.png")
    print(f"  Summary table: multiseed_summary.md")


if __name__ == "__main__":
    main()
