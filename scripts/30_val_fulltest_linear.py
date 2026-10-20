#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 30: Validation-chosen operating points, complete official test set, linear baseline.

Three inference-only stages (no retraining), selected with --stage:
  val       Infers the 30 checkpoints (6 configurations x 5 seeds) on validation, chooses
            decision thresholds there (Se = Sp, Youden, Sp >= 0.70, Sp >= 0.80) and
            applies them to the stored test probabilities, so the deployment operating
            point is fixed without looking at the test set.
  fulltest  Reintegrates the official-test recordings that the script 03 quality control
            excluded for clipping > 10 %, passes them through the same chain (03 without
            the exclusion rule, 05, 06) and re-infers the 30 checkpoints on the complete
            official test set. Reports post-QC vs complete, the AKG C417L subset, and
            worst/best-case bounds (all reintegrated cycles wrong / right).
  linear    Multinomial logistic regression and linear SVM (one-vs-rest) on per-band
            statistics of the same 128-band mel (mean, SD, temporal max; 384 dims), with
            C chosen on validation. Bounds how much of the discrimination is reached
            without time-frequency modelling. Patient-level bootstrap CIs, per-device AUC.
  all       The three stages in that order.

Reintegrated recordings use the script 03 preprocessor with the exclusion threshold set
to 100 %, so they receive the same clipping correction, 50-2000 Hz band-pass and RMS
normalisation as moderately clipped recordings; segmentation (05) and mel extraction (06)
are the original functions. The complete test set is written as a parallel split whose
first 2733 rows are the post-QC test; reproducing the stored post-QC argmax on those rows
is checked as an integrity control. Thresholds are chosen per seed on the validation ROC
and applied to the same seed on test, with the decision rule of script 29:
y = Normal if P(abnormal) < tau, else 1 + argmax P(1:4).

Inputs:  models/checkpoints/, data/splits/strategy1_official/,
         results/metrics/full_reeval/arrays/, data/processed/excluded_files.csv,
         data/processed/preprocessing_config.json, data/raw/ (ICBHI WAV/TXT files and
         ICBHI_challenge_train_test.txt)
Outputs: results/metrics/reanalysis/val_thresholds.{csv,json},
         fulltest_results.{csv,json}, fulltest_added_cycles.csv,
         linear_baseline_sel{icbhi,auc}.{csv,json}, linear_*_test_pabn.npy,
         script30_summary.md
         results/metrics/full_reeval/arrays/{run}_val_probs.npy, {run}_val_labels_preds.npz
         results/metrics/full_reeval/arrays_fulltest/{run}_probs.npy, {run}_labels_preds.npz
         data/splits/strategy1_official_fulltest/{train,val,test}_cycles.csv
         script30_log.txt (repository root)
Usage:
    python 30_val_fulltest_linear.py --stage {val,fulltest,linear,all}
        [--no_gpu] [--skip_panns] [--select {icbhi,auc}] [--n_boot 2000] [--boot_seed 2026]
"""

import sys
import json
import shutil
import argparse
import logging
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR_FIXED = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent

if BASE_DIR_FIXED.exists():
    BASE_DIR = BASE_DIR_FIXED
elif (SCRIPTS_DIR.parent / "models").exists():
    BASE_DIR = SCRIPTS_DIR.parent
else:
    BASE_DIR = SCRIPTS_DIR

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SEGMENTS_DIR = DATA_DIR / "segments"
MEL_DIR = DATA_DIR / "features" / "mel_spectrograms"
SPLITS_DIR = DATA_DIR / "splits"
SPLIT_OFFICIAL = SPLITS_DIR / "strategy1_official"
SPLIT_FULLTEST = SPLITS_DIR / "strategy1_official_fulltest"
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
METRICS_DIR = BASE_DIR / "results" / "metrics"
ARRAYS_DIR = METRICS_DIR / "full_reeval" / "arrays"
ARRAYS_FULL_DIR = METRICS_DIR / "full_reeval" / "arrays_fulltest"
OUT_DIR = METRICS_DIR / "reanalysis"

EXCLUDED_CSV = PROCESSED_DIR / "excluded_files.csv"
PREPROC_CONFIG = PROCESSED_DIR / "preprocessing_config.json"
OFFICIAL_SPLIT_TXT = RAW_DIR / "ICBHI_challenge_train_test.txt"
# Original ICBHI database (WAV + annotation TXT); same path as script 05
ICBHI_DIR = BASE_DIR / "data" / "raw"

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}
SEEDS = [42, 123, 7, 99, 314]

# kind: how the model is rebuilt
#   cnn / cfc  -> load_checkpoint_model from script 18
#   mnv3       -> build_model from script 21 (config stored in the checkpoint)
#   panns      -> Transfer_Cnn6 from script 22 + raw_audio_dataset
MODELS = {
    "CNN":        {"display": "CNN-RespiratoryNet (FP32)", "kind": "cnn",
                   "ckpt": "cnn_official_seed{S}.pth", "prefix": "cnn"},
    "MNv3_IN":    {"display": "MobileNetV3-Small (ImageNet)", "kind": "mnv3",
                   "ckpt": "tl_mobilenet_v3_small_pretrained_official_seed{S}.pth",
                   "prefix": "tl_mobilenet_v3_small_pretrained_official"},
    "MNv3_SC":    {"display": "MobileNetV3-Small (scratch)", "kind": "mnv3",
                   "ckpt": "tl_mobilenet_v3_small_official_seed{S}.pth",
                   "prefix": "tl_mobilenet_v3_small_official"},
    "PANNs":      {"display": "PANNs CNN6 (AudioSet)", "kind": "panns",
                   "ckpt": "panns_cnn6_official_seed{S}.pth",
                   "prefix": "panns_cnn6_official"},
    "PANNs_tune": {"display": "PANNs CNN6 (AudioSet, tune)", "kind": "panns",
                   "ckpt": "panns_cnn6_tune_official_seed{S}.pth",
                   "prefix": "panns_cnn6_tune_official"},
    "CfC":        {"display": "CfC-Hybrid", "kind": "cfc",
                   "ckpt": "hybrid_official_seed{S}.pth", "prefix": "hybrid"},
}
MEL_KINDS = {"cnn", "cfc", "mnv3"}

# (rule key, description written to val_thresholds.csv)
THRESHOLD_RULES = [
    ("se_eq_sp", "Se = Sp on validation"),
    ("youden",   "Max. Youden on validation"),
    ("sp_0.70",  "Sp >= 0.70 on validation"),
    ("sp_0.80",  "Sp >= 0.80 on validation"),
]


logger = logging.getLogger(__name__)


def setup_logging():
    log_file = BASE_DIR / "script30_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, mode="a", encoding="utf-8")],
        force=True,
    )
    return log_file


def import_script(name, filename):
    path = SCRIPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Metrics: same definitions as scripts 18 and 29

def confusion_from_preds(y_true, y_pred, n_classes=4):
    idx = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def icbhi_official(y_true, y_pred):
    cm = confusion_from_preds(y_true, y_pred).astype(float)
    sp = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0.0
    abn = cm[1:, :].sum()
    se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / abn if abn > 0 else 0.0
    return (se + sp) / 2.0, float(se), float(sp)


def auc_from_scores(y_bin, scores):
    pos = y_bin == 1
    n_pos = int(pos.sum()); n_neg = int(len(y_bin) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = stats.rankdata(scores)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_se_sp(y_true, y_pred):
    yt = y_true != 0; yp = y_pred != 0
    se = float((yt & yp).sum() / yt.sum()) if yt.sum() else 0.0
    sp = float((~yt & ~yp).sum() / (~yt).sum()) if (~yt).sum() else 0.0
    return se, sp


def all_metrics(y_true, y_pred, p_abn):
    s, se, sp = icbhi_official(y_true, y_pred)
    bse, bsp = binary_se_sp(y_true, y_pred)
    return {"icbhi": s, "se_off": se, "sp_off": sp, "bin_se": bse, "bin_sp": bsp,
            "auc": auc_from_scores((y_true != 0).astype(int), p_abn)}


def p_abnormal(probs32):
    """P(abnormal) = 1 - P(Normal), computed in float32 as in script 18 so that ties match script 29."""
    return (np.float32(1.0) - np.asarray(probs32, dtype=np.float32)[:, 0]).astype(np.float64)


def preds_at_threshold(probs, p_abn, tau):
    sub = 1 + np.argmax(np.asarray(probs)[:, 1:], axis=1)
    return np.where(p_abn >= tau, sub, 0)


def find_threshold(y_bin, p_abn, mode):
    taus = np.unique(np.concatenate([p_abn, [1.0 + 1e-9]]))
    pos_s = np.sort(p_abn[y_bin == 1]); neg_s = np.sort(p_abn[y_bin == 0])
    se = 1 - np.searchsorted(pos_s, taus, side="left") / max(len(pos_s), 1)
    sp = np.searchsorted(neg_s, taus, side="left") / max(len(neg_s), 1)
    if mode == "se_eq_sp":
        i = int(np.argmin(np.abs(se - sp)))
    elif mode == "youden":
        i = int(np.argmax(se + sp - 1))
    elif mode.startswith("sp_"):
        t = float(mode.split("_")[1]); ok = np.where(sp >= t)[0]
        i = int(ok[np.argmax(se[ok])]) if len(ok) else int(np.argmax(sp))
    elif mode.startswith("se_"):
        t = float(mode.split("_")[1]); ok = np.where(se >= t)[0]
        i = int(ok[np.argmax(sp[ok])]) if len(ok) else int(np.argmax(se))
    else:
        raise ValueError(mode)
    return float(taus[i])


def mean_sd(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else 0.0)


def fmt(mu, sd=None, nd=3):
    if mu is None or (isinstance(mu, float) and np.isnan(mu)):
        return "n/a"
    return f"{mu:.{nd}f}" if sd is None or np.isnan(sd) else f"{mu:.{nd}f} ± {sd:.{nd}f}"


class Inferencer:
    """
    Lazily imports the project modules (scripts 18/21/22) and builds the DataLoaders once
    per stage, so the loaders are not rebuilt for each of the 30 checkpoints.
    """

    def __init__(self, device, strategy, skip_panns=False):
        import torch
        self.torch = torch
        self.device = device
        self.strategy = strategy
        self.skip_panns = skip_panns
        self.s18 = import_script("script18", "18_full_reeval_multiseed.py")
        self.s09 = import_script("script09", "09_cnn_training.py")
        self.s11a = import_script("script11a", "11a_cfc_mel_v7.py")
        self.s21 = import_script("script21", "21_transfer_learning.py")
        self.s08 = import_script("script08", "08_dataset_dataloader.py")
        self._mel_loaders = None
        self._raw_loaders = None
        self.s22 = None; self.pann = None; self.rad = None

    def mel_loaders(self):
        if self._mel_loaders is None:
            self._mel_loaders = self.s08.create_dataloaders(
                strategy=self.strategy, class_mode="4class", augment_preset="none",
                batch_size=32, use_weighted_sampling=False)
        return self._mel_loaders

    def raw_loaders(self):
        if self._raw_loaders is None:
            self.s22 = import_script("script22", "22_panns_transfer.py")
            self.pann = self.s22._import_module("panns_cnn6", "panns_cnn6.py")
            self.rad = self.s22._import_module("raw_audio_dataset", "raw_audio_dataset.py")
            self._raw_loaders = self.rad.create_raw_audio_dataloaders(
                splits_dir=SPLITS_DIR, segments_dir=SEGMENTS_DIR,
                strategy=self.strategy, batch_size=16, use_weighted_sampling=False)
        return self._raw_loaders

    def build(self, kind, ckpt_path):
        torch = self.torch
        if kind in ("cnn", "cfc"):
            model, ck, cfg = self.s18.load_checkpoint_model(
                ckpt_path, kind, self.s09, self.s11a, self.device)
            return model
        if kind == "mnv3":
            ck = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            mcfg = {**self.s21.TRAINING_CONFIG, **ck.get("config", {})}
            model = self.s21.build_model(mcfg, self.device)
            model.load_state_dict(ck["model_state_dict"]); model.eval()
            return model
        if kind == "panns":
            self.raw_loaders()
            ck = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            mcfg = ck.get("config", {})
            model = self.pann.Transfer_Cnn6(classes_num=mcfg.get("classes_num", 4),
                                            freeze_base=False).to(self.device)
            model.load_state_dict(ck["model_state_dict"]); model.eval()
            return model
        raise ValueError(kind)

    def run(self, model, loader):
        """Return (y_true, y_pred, float32 probs), iterating the loader in order (no shuffling)."""
        torch = self.torch
        ys, ps = [], []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    xk = next((k for k in ("x", "input", "waveform", "audio", "spectrogram", "mel")
                               if k in batch), None)
                    yk = next((k for k in ("y", "label", "labels", "target")
                               if k in batch), None)
                    if xk is None or yk is None:
                        raise KeyError(f"Batch dict with keys {list(batch.keys())}: "
                                       f"input/label key not recognised. Adjust Inferencer.run.")
                    x, y = batch[xk], batch[yk]
                else:
                    x, y = batch[0], batch[1]
                x = x.to(self.device)
                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                if isinstance(logits, dict):        # some PANNs wrappers return a dict
                    logits = logits.get("clipwise_output", next(iter(logits.values())))
                ps.append(torch.softmax(logits.float(), dim=1).cpu().numpy().astype(np.float32))
                ys.append(np.asarray(y))
        probs = np.concatenate(ps); y_true = np.concatenate(ys).astype(int)
        return y_true, np.argmax(probs, axis=1), probs

    def infer(self, key, seed, split):
        spec = MODELS[key]
        if spec["kind"] == "panns" and self.skip_panns:
            return None
        ckpt = CHECKPOINTS_DIR / spec["ckpt"].format(S=seed)
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model = self.build(spec["kind"], ckpt)
        loaders = self.raw_loaders() if spec["kind"] == "panns" else self.mel_loaders()
        lk = f"{split}_loader"
        if lk not in loaders:
            raise KeyError(f"Loader dict has no '{lk}'; keys: {list(loaders.keys())}")
        loader = loaders[lk]
        return self.run(model, loader)


def stage_val(args):
    """Choose thresholds on validation per seed and apply them to the stored test probabilities."""
    import torch
    device = torch.device("cpu") if args.no_gpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[VAL] device = {device}")
    inf = Inferencer(device, "strategy1_official", skip_panns=args.skip_panns)
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(SPLIT_OFFICIAL / "test_cycles.csv")
    y_test = df_test["cycle_class"].map(CLASS_MAP).values.astype(int)
    df_val = pd.read_csv(SPLIT_OFFICIAL / "val_cycles.csv")
    y_val_csv = df_val["cycle_class"].map(CLASS_MAP).values.astype(int)

    rows, per_seed_val = [], {}
    for key, spec in MODELS.items():
        if spec["kind"] == "panns" and args.skip_panns:
            logger.info(f"  {key}: skipped (--skip_panns)"); continue
        per_rule = {m: {k: [] for k in ["tau", "val_se", "val_sp", "icbhi", "se_off",
                                       "sp_off", "bin_se", "bin_sp"]} for m, _ in THRESHOLD_RULES}
        val_metrics = {k: [] for k in ["icbhi", "auc"]}
        for seed in SEEDS:
            out = inf.infer(key, seed, "val")
            yv, pv, probs_v = out
            if len(yv) != len(y_val_csv) or not np.array_equal(yv, y_val_csv):
                raise ValueError(f"{key} seed {seed}: val not aligned with val_cycles.csv")
            stem = f"{spec['prefix']}_seed{seed}"
            np.save(ARRAYS_DIR / f"{stem}_val_probs.npy", probs_v)
            np.savez(ARRAYS_DIR / f"{stem}_val_labels_preds.npz", y_true=yv, y_pred=pv)
            pabn_v = p_abnormal(probs_v)
            mv = all_metrics(yv, pv, pabn_v)
            val_metrics["icbhi"].append(mv["icbhi"]); val_metrics["auc"].append(mv["auc"])

            # Test: previously stored arrays
            probs_t = np.load(ARRAYS_DIR / f"{stem}_probs.npy")
            pabn_t = p_abnormal(probs_t)
            for mode, _ in THRESHOLD_RULES:
                tau = find_threshold((yv != 0).astype(int), pabn_v, mode)
                # Se/Sp that this tau yields on val (sanity check)
                yv_pred = preds_at_threshold(probs_v, pabn_v, tau)
                vse, vsp = binary_se_sp(yv, yv_pred)
                yt_pred = preds_at_threshold(probs_t, pabn_t, tau)
                mt = all_metrics(y_test, yt_pred, pabn_t)
                d = per_rule[mode]
                d["tau"].append(tau); d["val_se"].append(vse); d["val_sp"].append(vsp)
                for k in ["icbhi", "se_off", "sp_off", "bin_se", "bin_sp"]:
                    d[k].append(mt[k])
        vi, va = mean_sd(val_metrics["icbhi"]), mean_sd(val_metrics["auc"])
        per_seed_val[key] = {"val_icbhi": val_metrics["icbhi"], "val_auc": val_metrics["auc"]}
        logger.info(f"  {key:<10} VAL ICBHI {fmt(*vi)}  AUC {fmt(*va)}")
        for mode, desc in THRESHOLD_RULES:
            d = per_rule[mode]
            row = {"model": key, "rule": mode, "description": desc}
            for k, v in d.items():
                row[f"{k}_mean"], row[f"{k}_sd"] = mean_sd(v)
            rows.append(row)
            logger.info(f"     {mode:<9} tau={row['tau_mean']:.3f}±{row['tau_sd']:.3f} "
                        f"val Se/Sp {row['val_se_mean']:.2f}/{row['val_sp_mean']:.2f} → "
                        f"TEST bin Se/Sp {row['bin_se_mean']:.3f}/{row['bin_sp_mean']:.3f}  "
                        f"ICBHI {row['icbhi_mean']:.3f}±{row['icbhi_sd']:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "val_thresholds.csv", index=False)
    with open(OUT_DIR / "val_thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "rules": rows, "val_metrics_per_seed": per_seed_val}, f, indent=2)
    return rows


def read_official_split():
    split = {}
    with open(OFFICIAL_SPLIT_TXT, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                split[parts[0].strip()] = parts[1].strip()
    return split


def locate_recording(stem):
    """WAV and TXT of a recording: data/raw first, then ICBHI_DIR."""
    wav = txt = None
    for d in (RAW_DIR, ICBHI_DIR):
        if wav is None and (d / f"{stem}.wav").exists():
            wav = d / f"{stem}.wav"
        if txt is None and (d / f"{stem}.txt").exists():
            txt = d / f"{stem}.txt"
    return wav, txt


def reintegrate_excluded_test():
    """
    Preprocess (03 without exclusion), segment (05) and extract mel (06) for the excluded
    recordings that belong to the official test partition. Returns a DataFrame with the
    columns of test_cycles.csv.
    """
    s03 = import_script("script03", "03_preprocessing_pipeline.py")
    s05 = import_script("script05", "05_segment_respiratory_cycles.py")
    s06 = import_script("script06", "06_extract_mel_spectrograms.py")
    import soundfile as sf

    if not EXCLUDED_CSV.exists():
        raise FileNotFoundError(f"Not found: {EXCLUDED_CSV} (written by script 03)")
    exc = pd.read_csv(EXCLUDED_CSV)
    official = read_official_split()
    exc["stem"] = exc["filename"].astype(str).str.replace(".wav", "", regex=False)
    exc["official_split"] = exc["stem"].map(official)
    logger.info(f"  Excluded by QC: {len(exc)}  "
                f"(official test: {int((exc['official_split'] == 'test').sum())}, "
                f"official train: {int((exc['official_split'] == 'train').sum())})")
    exc_test = exc[exc["official_split"] == "test"].copy()
    for _, r in exc_test.iterrows():
        logger.info(f"    {r['stem']}: {r.get('exclusion_reason', '')}")

    # Same preprocessor as script 03, with the exclusion rule disabled
    pcfg = json.load(open(PREPROC_CONFIG)) if PREPROC_CONFIG.exists() else {}
    pre = s03.RespiratoryAudioPreprocessor(
        target_sr=pcfg.get("target_sr", 16000),
        bandpass_lowcut=pcfg.get("bandpass_lowcut", 50),
        bandpass_highcut=pcfg.get("bandpass_highcut", 2000),
        clipping_correction_threshold=pcfg.get("clipping_correction_threshold", 1.0),
        clipping_exclude_threshold=100.0,          # only difference from script 03
        normalize_rms_target=pcfg.get("normalize_rms_target", 0.1),
    )
    seg_cfg = s05.SEGMENTATION_CONFIG
    mel_cfg = s06.FEATURE_CONFIG
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True); MEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in exc_test.iterrows():
        stem = r["stem"]
        wav, txt = locate_recording(stem)
        if wav is None or txt is None:
            raise FileNotFoundError(f"{stem}: WAV={wav} TXT={txt}")
        meta = s05.parse_filename(wav.name)
        audio, pmeta = pre.process_single_file(str(wav))
        if audio is None:
            raise RuntimeError(f"{stem}: excluded by the preprocessor despite the 100 % threshold: {pmeta}")
        logger.info(f"    {stem}: original clipping {pmeta.get('original_clipping_pct', float('nan')):.2f} % → "
                    f"{'; '.join(pmeta.get('transformations_applied', []))}")
        sr = pre.target_sr
        cycles = s05.parse_annotation_file(txt)
        audio_dur = len(audio) / sr
        for ci, cinfo in enumerate(cycles):
            if cinfo["end"] > audio_dur + 0.1:
                cinfo["end"] = audio_dur
            cyc_audio, dur, padded, trunc = s05.extract_cycle_audio(
                audio, sr, cinfo["start"], cinfo["end"], seg_cfg)
            cycle_filename = f"{meta['patient_id']}_{meta['recording_idx']}_{meta['chest_location']}_cycle{ci:04d}.wav"
            cycle_id = f"{meta['patient_id']}_{meta['recording_idx']}_{meta['chest_location']}_c{ci:04d}"
            sf.write(SEGMENTS_DIR / cycle_filename, cyc_audio, sr, subtype=seg_cfg["output_subtype"])
            mel, mmeta = s06.compute_mel_spectrogram(SEGMENTS_DIR / cycle_filename, mel_cfg)
            mel_filename = Path(cycle_filename).stem + "_mel.npy"
            np.save(MEL_DIR / mel_filename, mel)
            cls = s05.CYCLE_CLASSES.get((cinfo["crackles"], cinfo["wheezes"]), "Unknown")
            rows.append({
                "cycle_id": cycle_id, "cycle_filename": cycle_filename,
                "patient_id": meta["patient_id"], "recording_idx": meta["recording_idx"],
                "filename_original": wav.name, "cycle_index": ci,
                "start_time": cinfo["start"], "end_time": cinfo["end"], "duration": dur,
                "duration_padded": seg_cfg["max_cycle_duration_sec"],
                "was_padded": padded, "was_truncated": trunc,
                "has_crackles": cinfo["crackles"], "has_wheezes": cinfo["wheezes"],
                "cycle_class": cls, "chest_location": meta["chest_location"],
                "chest_location_full": s05.CHEST_LOCATION_MAP.get(meta["chest_location"], ""),
                "acquisition_mode": meta["acq_mode"], "recording_equipment": meta["equipment"],
                "recording_equipment_full": s05.EQUIPMENT_MAP.get(meta["equipment"], ""),
                "sample_rate": sr, "total_samples": len(cyc_audio), "mel_filename": mel_filename,
                "mel_shape_rows": mmeta["n_mel_bands"], "mel_shape_cols": mmeta["n_time_frames"],
                "raw_db_min": mmeta["raw_db_min"], "raw_db_max": mmeta["raw_db_max"],
                "raw_db_mean": mmeta["raw_db_mean"], "raw_db_std": mmeta["raw_db_std"],
                "filename_base": stem, "split": "test", "reintegrated": True,
                "qc_exclusion_reason": r.get("exclusion_reason", ""),
            })
    added = pd.DataFrame(rows)
    logger.info(f"  Reintegrated cycles: {len(added)}  "
                + ", ".join(f"{c}={int((added['cycle_class'] == c).sum())}" for c in CLASS_NAMES))
    return added


def build_fulltest_split(added):
    """Write strategy1_official_fulltest: train/val copied, test = post-QC test followed by the added cycles."""
    SPLIT_FULLTEST.mkdir(parents=True, exist_ok=True)
    for name in ("train_cycles.csv", "val_cycles.csv"):
        shutil.copy2(SPLIT_OFFICIAL / name, SPLIT_FULLTEST / name)
    test = pd.read_csv(SPLIT_OFFICIAL / "test_cycles.csv")
    test["reintegrated"] = False
    dup = set(test["cycle_id"]) & set(added["cycle_id"])
    if dup:
        raise ValueError(f"Duplicate cycle_id between test and reintegrated cycles: {sorted(dup)[:5]}")
    full = pd.concat([test, added[[c for c in added.columns if c in test.columns or c == "reintegrated"]]],
                     ignore_index=True)
    full.to_csv(SPLIT_FULLTEST / "test_cycles.csv", index=False)
    logger.info(f"  Fulltest split: test {len(test)} + {len(added)} = {len(full)} cycles, "
                f"{full['patient_id'].nunique()} patients")
    return full


def stage_fulltest(args):
    """Re-infer all checkpoints on the complete official test set and compare with post-QC."""
    import torch
    device = torch.device("cpu") if args.no_gpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[FULLTEST] device = {device}")

    added = reintegrate_excluded_test()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    added.to_csv(OUT_DIR / "fulltest_added_cycles.csv", index=False)
    full = build_fulltest_split(added)
    n_old = int((~full["reintegrated"]).sum())
    y_full = full["cycle_class"].map(CLASS_MAP).values.astype(int)
    is_new = full["reintegrated"].values.astype(bool)
    equip = full["recording_equipment"].values

    inf = Inferencer(device, "strategy1_official_fulltest", skip_panns=args.skip_panns)
    ARRAYS_FULL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, spec in MODELS.items():
        if spec["kind"] == "panns" and args.skip_panns:
            logger.info(f"  {key}: skipped (--skip_panns)"); continue
        acc = {k: [] for k in ["icbhi_post", "auc_post", "icbhi_full", "auc_full",
                               "se_full", "sp_full", "icbhi_worst", "icbhi_best",
                               "new_acc", "icbhi_akg_post", "icbhi_akg_full",
                               "auc_akg_post", "auc_akg_full", "max_abs_diff_old"]}
        for seed in SEEDS:
            y, yp, probs = inf.infer(key, seed, "test")
            if len(y) != len(y_full) or not np.array_equal(y, y_full):
                raise ValueError(f"{key} seed {seed}: fulltest test set not aligned with the CSV")
            stem = f"{spec['prefix']}_seed{seed}"
            np.save(ARRAYS_FULL_DIR / f"{stem}_probs.npy", probs)
            np.savez(ARRAYS_FULL_DIR / f"{stem}_labels_preds.npz", y_true=y, y_pred=yp)

            # Integrity: the first n_old rows must reproduce the post-QC array
            old = np.load(ARRAYS_DIR / f"{stem}_probs.npy")
            diff = float(np.max(np.abs(old.astype(np.float64) - probs[:n_old].astype(np.float64))))
            same_argmax = bool(np.array_equal(np.argmax(old, 1), yp[:n_old]))
            acc["max_abs_diff_old"].append(diff)
            if not same_argmax:
                logger.warning(f"    {stem}: argmax differs from the post-QC array in "
                               f"{int((np.argmax(old, 1) != yp[:n_old]).sum())} rows (Δmax={diff:.2e})")

            pabn = p_abnormal(probs)
            m_post = all_metrics(y_full[:n_old], yp[:n_old], pabn[:n_old])
            m_full = all_metrics(y_full, yp, pabn)
            acc["icbhi_post"].append(m_post["icbhi"]); acc["auc_post"].append(m_post["auc"])
            acc["icbhi_full"].append(m_full["icbhi"]); acc["auc_full"].append(m_full["auc"])
            acc["se_full"].append(m_full["se_off"]); acc["sp_full"].append(m_full["sp_off"])
            acc["new_acc"].append(float((yp[is_new] == y_full[is_new]).mean()) if is_new.any() else np.nan)
            # Bounds: all reintegrated cycles wrong / all right
            yp_worst = yp.copy(); yp_worst[is_new] = np.where(y_full[is_new] == 0, 1, 0)
            yp_best = yp.copy(); yp_best[is_new] = y_full[is_new]
            acc["icbhi_worst"].append(icbhi_official(y_full, yp_worst)[0])
            acc["icbhi_best"].append(icbhi_official(y_full, yp_best)[0])
            # AKG subset (the device of the excluded recordings)
            akg = equip == "AKGC417L"
            akg_old = akg.copy(); akg_old[n_old:] = False
            mo = all_metrics(y_full[akg_old], yp[akg_old], pabn[akg_old])
            mf = all_metrics(y_full[akg], yp[akg], pabn[akg])
            acc["icbhi_akg_post"].append(mo["icbhi"]); acc["icbhi_akg_full"].append(mf["icbhi"])
            acc["auc_akg_post"].append(mo["auc"]); acc["auc_akg_full"].append(mf["auc"])
        row = {"model": key, "n_test_post_qc": n_old, "n_test_full": int(len(y_full)),
               "n_reintegrated": int(is_new.sum())}
        for k, v in acc.items():
            row[f"{k}_mean"], row[f"{k}_sd"] = mean_sd(v)
        rows.append(row)
        logger.info(f"  {key:<10} ICBHI post-QC {fmt(row['icbhi_post_mean'], row['icbhi_post_sd'])} → "
                    f"complete {fmt(row['icbhi_full_mean'], row['icbhi_full_sd'])}  "
                    f"[worst {row['icbhi_worst_mean']:.3f}, best {row['icbhi_best_mean']:.3f}]  "
                    f"AUC {fmt(row['auc_post_mean'], row['auc_post_sd'])} → {fmt(row['auc_full_mean'], row['auc_full_sd'])}  "
                    f"acc reintegrated {row['new_acc_mean']:.2f}  Δmax vs post-QC {row['max_abs_diff_old_mean']:.1e}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "fulltest_results.csv", index=False)
    with open(OUT_DIR / "fulltest_results.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "reintegrated_recordings": sorted(added["filename_base"].unique().tolist()),
                   "n_reintegrated_cycles": int(len(added)),
                   "class_counts_reintegrated": {c: int((added["cycle_class"] == c).sum()) for c in CLASS_NAMES},
                   "results": rows}, f, indent=2)
    return rows, added


def mel_features(df, desc):
    """(N, 384) features: per-band mean, SD and temporal max of the mel (already in [0, 1] from script 06)."""
    X = np.empty((len(df), 384), dtype=np.float32)
    for i, fn in enumerate(df["mel_filename"].values):
        m = np.load(MEL_DIR / fn).astype(np.float32)          # (128, T)
        X[i, :128] = m.mean(1); X[i, 128:256] = m.std(1); X[i, 256:] = m.max(1)
        if (i + 1) % 1000 == 0:
            logger.info(f"    {desc}: {i + 1}/{len(df)}")
    return X


def stage_linear(args):
    """Logistic regression and linear SVM on mel statistics; C chosen on validation by --select."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler

    tr = pd.read_csv(SPLIT_OFFICIAL / "train_cycles.csv")
    va = pd.read_csv(SPLIT_OFFICIAL / "val_cycles.csv")
    te = pd.read_csv(SPLIT_OFFICIAL / "test_cycles.csv")
    ytr, yva, yte = (d["cycle_class"].map(CLASS_MAP).values.astype(int) for d in (tr, va, te))
    logger.info(f"[LINEAR] train {len(tr)} / val {len(va)} / test {len(te)}; extracting features")
    Xtr, Xva, Xte = mel_features(tr, "train"), mel_features(va, "val"), mel_features(te, "test")
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    C_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    results, rows = {}, []

    def score_abn(model, X):
        """P(abnormal) from predict_proba when available, else from a softmax of decision_function."""
        if hasattr(model, "predict_proba"):
            return p_abnormal(model.predict_proba(X).astype(np.float32))
        d = model.decision_function(X)
        e = np.exp(d - d.max(1, keepdims=True)); e /= e.sum(1, keepdims=True)
        return p_abnormal(e.astype(np.float32))

    for name, make in [("logreg", lambda C: LogisticRegression(C=C, class_weight="balanced",
                                                                 max_iter=3000)),
                       ("linsvm", lambda C: LinearSVC(C=C, class_weight="balanced", max_iter=20000))]:
        best = None
        for C in C_grid:
            m = make(C).fit(Xtr, ytr)
            mv = all_metrics(yva, m.predict(Xva), score_abn(m, Xva))
            logger.info(f"  {name} C={C:<6} val ICBHI {mv['icbhi']:.3f}  AUC {mv['auc']:.3f}")
            crit = mv["icbhi"] if args.select == "icbhi" else mv["auc"]
            if best is None or crit > best[3]:
                best = (C, mv, m, crit)
        C, mv, m, _ = best
        yp = m.predict(Xte); pabn = score_abn(m, Xte)
        mt = all_metrics(yte, yp, pabn)
        row = {"model": name, "select": args.select, "C": C, "val_icbhi": mv["icbhi"],
               "val_auc": mv["auc"], **{f"test_{k}": v for k, v in mt.items()}}
        # Patient-level bootstrap
        rng = np.random.default_rng(args.boot_seed)
        pats = te["patient_id"].values; uniq = np.unique(pats)
        idx_by = {p: np.where(pats == p)[0] for p in uniq}
        bi, ba = [], []
        for _ in range(args.n_boot):
            idx = np.concatenate([idx_by[p] for p in rng.choice(uniq, len(uniq), replace=True)])
            mb = all_metrics(yte[idx], yp[idx], pabn[idx]); bi.append(mb["icbhi"]); ba.append(mb["auc"])
        row["test_icbhi_ci_lo"], row["test_icbhi_ci_hi"] = np.nanpercentile(bi, 2.5), np.nanpercentile(bi, 97.5)
        row["test_auc_ci_lo"], row["test_auc_ci_hi"] = np.nanpercentile(ba, 2.5), np.nanpercentile(ba, 97.5)
        for eq, sub in te.groupby("recording_equipment"):
            i = sub.index.values
            me = all_metrics(yte[i], yp[i], pabn[i])
            row[f"{eq}_icbhi"], row[f"{eq}_auc"] = me["icbhi"], me["auc"]
        rows.append(row)
        results[name] = {"C": C, "test_pred": yp.tolist()}
        logger.info(f"  {name} (C={C}) TEST ICBHI {mt['icbhi']:.3f} "
                    f"[{row['test_icbhi_ci_lo']:.3f}, {row['test_icbhi_ci_hi']:.3f}]  "
                    f"Se {mt['se_off']:.3f} / Sp {mt['sp_off']:.3f}  "
                    f"AUC {mt['auc']:.3f} [{row['test_auc_ci_lo']:.3f}, {row['test_auc_ci_hi']:.3f}]  "
                    + "  ".join(f"{eq} AUC {row[f'{eq}_auc']:.3f}" for eq in te['recording_equipment'].unique()))
        np.save(OUT_DIR / f"linear_{name}_sel{args.select}_test_pabn.npy", pabn.astype(np.float32))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"linear_baseline_sel{args.select}.csv", index=False)
    with open(OUT_DIR / f"linear_baseline_sel{args.select}.json", "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "features": "mel 128 bands: temporal mean, sd, max (384 dims), StandardScaler(train)",
                   "C_grid": C_grid, "selection": f"{args.select} on validation", "rows": rows}, f, indent=2)
    return rows


def write_summary(val_rows, full_rows, lin_rows):
    """Append the Markdown tables of the stages that ran to script30_summary.md."""
    L = ["# Script 30: validation thresholds, complete test set and linear baseline\n",
         f"_{datetime.now().isoformat(timespec='seconds')}_\n"]
    if val_rows:
        L += ["\n## Threshold chosen on VALIDATION, applied to TEST (mean ± SD, 5 seeds)\n",
              "| Model | Rule | tau | Se/Sp val | Binary Se test | Binary Sp test | ICBHI test | Official Se | Official Sp |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in val_rows:
            L.append(f"| {r['model']} | {r['rule']} | {fmt(r['tau_mean'], r['tau_sd'])} "
                     f"| {r['val_se_mean']:.2f}/{r['val_sp_mean']:.2f} "
                     f"| {fmt(r['bin_se_mean'], r['bin_se_sd'])} | {fmt(r['bin_sp_mean'], r['bin_sp_sd'])} "
                     f"| {fmt(r['icbhi_mean'], r['icbhi_sd'])} | {fmt(r['se_off_mean'], r['se_off_sd'])} "
                     f"| {fmt(r['sp_off_mean'], r['sp_off_sd'])} |")
    if full_rows:
        L += ["\n## Official test set: post-QC vs COMPLETE (excluded recordings reintegrated)\n",
              "| Model | ICBHI post-QC | ICBHI complete | Worst case | Best case | AUC post-QC | AUC complete | AKG ICBHI post→full | AKG AUC post→full |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in full_rows:
            L.append(f"| {r['model']} | {fmt(r['icbhi_post_mean'], r['icbhi_post_sd'])} "
                     f"| {fmt(r['icbhi_full_mean'], r['icbhi_full_sd'])} | {r['icbhi_worst_mean']:.3f} | {r['icbhi_best_mean']:.3f} "
                     f"| {fmt(r['auc_post_mean'], r['auc_post_sd'])} | {fmt(r['auc_full_mean'], r['auc_full_sd'])} "
                     f"| {r['icbhi_akg_post_mean']:.3f} → {r['icbhi_akg_full_mean']:.3f} "
                     f"| {r['auc_akg_post_mean']:.3f} → {r['auc_akg_full_mean']:.3f} |")
    if lin_rows:
        L += [f"\n## Linear baseline on mel statistics (post-QC test, patient-level 95 % CI; "
              f"C selected by {lin_rows[0]['select']} on val)\n",
              "| Model | C | ICBHI val | ICBHI test [CI] | Official Se | Official Sp | AUC test [CI] |",
              "|---|---|---|---|---|---|---|"]
        for r in lin_rows:
            L.append(f"| {r['model']} | {r['C']} | {r['val_icbhi']:.3f} "
                     f"| {r['test_icbhi']:.3f} [{r['test_icbhi_ci_lo']:.3f}, {r['test_icbhi_ci_hi']:.3f}] "
                     f"| {r['test_se_off']:.3f} | {r['test_sp_off']:.3f} "
                     f"| {r['test_auc']:.3f} [{r['test_auc_ci_lo']:.3f}, {r['test_auc_ci_hi']:.3f}] |")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if (OUT_DIR / "script30_summary.md").exists() else "w"
    with open(OUT_DIR / "script30_summary.md", mode, encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Script 30: val / fulltest / linear")
    p.add_argument("--stage", choices=["val", "fulltest", "linear", "all"], required=True)
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--skip_panns", action="store_true")
    p.add_argument("--select", choices=["icbhi", "auc"], default="icbhi",
                   help="Criterion for selecting C on validation (linear stage); run both.")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--boot_seed", type=int, default=2026)
    return p.parse_args()


def main():
    args = parse_args()
    log_file = setup_logging()
    logger.info("=" * 78)
    logger.info(f"SCRIPT 30, stage: {args.stage}")
    logger.info("=" * 78)
    logger.info(f"BASE_DIR: {BASE_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_rows = full_rows = lin_rows = None
    if args.stage in ("val", "all"):
        val_rows = stage_val(args)
    if args.stage in ("fulltest", "all"):
        full_rows, _ = stage_fulltest(args)
    if args.stage in ("linear", "all"):
        lin_rows = stage_linear(args)
    write_summary(val_rows, full_rows, lin_rows)
    logger.info(f"Summary: {OUT_DIR / 'script30_summary.md'}   Log: {log_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
