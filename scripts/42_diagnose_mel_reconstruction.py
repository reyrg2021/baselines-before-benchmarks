#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 42: Diagnosis of the log-mel reconstruction used by the noise-robustness experiment.

The clean condition of script 40 did not reproduce the reference test results of
script 18 (ICBHI 0.528 and AUC 0.724 instead of 0.460 and 0.572, with seed SD rising
from 0.017 to 0.154). Three causes are separated: (A) faulty metrics or checkpoint
loading; (B) a different input tensor, because script 40 re-applies the bandpass filter
and RMS normalization to segments that were already conditioned; (C) precomputed .npy
features that do not correspond to the WAV segments. The same test set and checkpoints
are evaluated with three inputs: A = precomputed .npy from script 06 (must reproduce
the reference), B = mel from the segment WAV without filter/RMS (must match A),
C = mel from the WAV with filter and RMS (script 40 clean condition). An element-wise
A-vs-B and A-vs-C comparison on a sample of cycles detects cause C directly.

Inputs:  data/splits/strategy1_official/test_cycles.csv, data/segments/,
         data/features/mel_spectrograms/, models/checkpoints/cnn_official_seed{S}.pth
Outputs: results/metrics/noise_robustness/mel_diagnosis.csv,
         results/metrics/noise_robustness/mel_diagnosis.json, mel_diagnosis_log.txt
Usage:
    python scripts/42_diagnose_mel_reconstruction.py [--n_compare 100]
"""

import sys
import json
import argparse
import importlib.util
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy import signal
from scipy.stats import rankdata

import torch


SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR_FIXED = Path(__file__).resolve().parent.parent

if BASE_DIR_FIXED.exists():
    BASE_DIR = BASE_DIR_FIXED
elif (SCRIPTS_DIR.parent / "data").exists():
    BASE_DIR = SCRIPTS_DIR.parent
else:
    BASE_DIR = SCRIPTS_DIR

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_MAP = {"Normal": 0, "Crackle": 1, "Wheeze": 2, "Both": 3}
SEEDS = [42, 123, 7, 99, 314]

AUDIO_CONFIG = {
    "target_sr": 16000,
    "bandpass_lowcut": 50,
    "bandpass_highcut": 2000,
    "bandpass_order": 5,
    "rms_target": 0.1,
}

MEL_CONFIG = {
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "fmin": 50,
    "fmax": 2000,
    "power": 2.0,
    "top_db": 80.0,
}

# Reference CNN FP32 test results (script 18) and matching tolerance
OFFICIAL_ICBHI = 0.460
OFFICIAL_AUC = 0.572
TOLERANCE = 0.010


logger = logging.getLogger(__name__)


def setup_logging(base_dir):
    log_file = base_dir / "mel_diagnosis_log.txt"
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False   # avoids duplicated log lines

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return log_file


def section(title_text):
    """Log a section heading."""
    logger.info("")
    logger.info("=" * 74)
    logger.info(f"  {title_text}")
    logger.info("=" * 74)


def design_bandpass(sr):
    """Butterworth bandpass coefficients with the preprocessing parameters."""
    nyq = 0.5 * sr
    return signal.butter(
        AUDIO_CONFIG["bandpass_order"],
        [AUDIO_CONFIG["bandpass_lowcut"] / nyq, AUDIO_CONFIG["bandpass_highcut"] / nyq],
        btype="band",
    )


def build_mel(audio, sr):
    """Exact replica of compute_mel_spectrogram from script 06."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=MEL_CONFIG["n_mels"], n_fft=MEL_CONFIG["n_fft"],
        hop_length=MEL_CONFIG["hop_length"],
        fmin=MEL_CONFIG["fmin"], fmax=MEL_CONFIG["fmax"],
        power=MEL_CONFIG["power"],
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=MEL_CONFIG["top_db"])
    mn, mx = mel_db.min(), mel_db.max()
    if mx - mn > 1e-6:
        mel_db = (mel_db - mn) / (mx - mn)
    else:
        mel_db = np.zeros_like(mel_db)
    return mel_db.astype(np.float32)


def mel_variant_A(mel_path):
    """Variant A: array precomputed by script 06 (the input behind the reference results)."""
    return np.load(mel_path).astype(np.float32)


def mel_variant_B(wav_path):
    """Variant B: literal reconstruction WAV -> mel, without filter or RMS normalization."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return build_mel(audio.astype(np.float64), sr)


def mel_variant_C(wav_path, b, a):
    """Variant C: script 40 clean condition, WAV -> bandpass -> RMS -> mel."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = signal.filtfilt(b, a, audio.astype(np.float64))
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-10:
        audio = audio * (AUDIO_CONFIG["rms_target"] / rms)
    return build_mel(audio, sr)


def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_models(base_dir, device):
    """Load the five CNN checkpoints using the model class from script 09."""
    script09 = import_module_from_path("script09", SCRIPTS_DIR / "09_cnn_training.py")
    ckpt_dir = base_dir / "models" / "checkpoints"
    model_set = {}
    for seed in SEEDS:
        p = ckpt_dir / f"cnn_official_seed{seed}.pth"
        if not p.exists():
            logger.warning(f"  Missing {p.name}")
            continue
        ck = torch.load(str(p), map_location=device, weights_only=False)
        cfg = ck.get("config", {})
        m = script09.RespiratoryNet(
            n_classes=cfg.get("n_classes", 4),
            input_channels=cfg.get("input_channels", 1),
            base_filters=cfg.get("base_filters", 16),
            n_conv_blocks=cfg.get("n_conv_blocks", 4),
            fc_hidden_dim=cfg.get("fc_hidden_dim", 128),
            dropout_rate=cfg.get("dropout_rate", 0.5),
        ).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        model_set[seed] = m
    logger.info(f"  {len(model_set)} checkpoints loaded")
    return model_set


def icbhi_official(y_true, y_pred):
    """Official ICBHI score; returns (score, se, sp)."""
    es_n = (y_true == 0)
    sp = float((y_pred[es_n] == 0).sum()) / max(int(es_n.sum()), 1)
    se = float((y_pred[~es_n] == y_true[~es_n]).sum()) / max(int((~es_n).sum()), 1)
    return (se + sp) / 2.0, se, sp


def binary_auc(y_true, p_abn):
    """Binary Normal-vs-Abnormal AUC via the Mann-Whitney rank formula."""
    y_bin = (y_true != 0).astype(int)
    n_pos, n_neg = int(y_bin.sum()), len(y_bin) - int(y_bin.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(p_abn)
    return float((r[y_bin == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def run_inference(model, X, device, bs=64):
    """Softmax probabilities for X in batches."""
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.softmax(model(X[i:i + bs].to(device)), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def parse_args():
    p = argparse.ArgumentParser(description="Script 42 - Diagnosis of the log-mel reconstruction")
    p.add_argument("--base_dir", type=str, default=None)
    p.add_argument("--n_compare", type=int, default=100,
                   help="Number of cycles for the element-wise comparison")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir) if args.base_dir else BASE_DIR
    log_file = setup_logging(base_dir)
    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("=" * 74)
    logger.info("  SCRIPT 42 - DIAGNOSIS OF THE LOG-MEL RECONSTRUCTION")
    logger.info("=" * 74)
    logger.info(f"  Date:     {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"  BASE_DIR: {base_dir}")
    logger.info(f"  Device:   {device}")
    logger.info(f"  Log:      {log_file}")

    split_csv = base_dir / "data" / "splits" / "strategy1_official" / "test_cycles.csv"
    segments_dir = base_dir / "data" / "segments"
    mel_dir = base_dir / "data" / "features" / "mel_spectrograms"
    out_dir = base_dir / "results" / "metrics" / "noise_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in [split_csv, segments_dir, mel_dir]:
        if not r.exists():
            logger.error(f"  Not found: {r}")
            return 1

    test_df = pd.read_csv(split_csv)
    y_true_full = test_df["cycle_class"].map(CLASS_MAP).to_numpy()
    logger.info(f"  Test cycles: {len(test_df)}")

    b, a = design_bandpass(AUDIO_CONFIG["target_sr"])

    # WAV and .npy paths per cycle
    path_map = []
    for row_data in test_df.itertuples(index=False):
        wav = segments_dir / row_data.cycle_filename
        if hasattr(row_data, "mel_filename") and isinstance(row_data.mel_filename, str):
            npy = mel_dir / row_data.mel_filename
        else:
            npy = mel_dir / (Path(row_data.cycle_filename).stem + "_mel.npy")
        path_map.append((wav, npy))

    missing_wav = sum(1 for w, _ in path_map if not w.exists())
    missing_npy = sum(1 for _, n in path_map if not n.exists())
    logger.info(f"  Missing WAV: {missing_wav}   missing .npy: {missing_npy}")

    if missing_npy > 0:
        logger.error("  Some .npy files are missing. The variant A control would be partial.")
        return 1

    # Part 1: element-wise comparison on an evenly spaced sample of cycles
    section("PART 1 - ELEMENT-WISE COMPARISON (A vs B, A vs C)")

    n_cmp = min(args.n_compare, len(path_map))
    idx_cmp = np.linspace(0, len(path_map) - 1, n_cmp).astype(int)

    dif_AB, dif_AC, corr_AB, corr_AC = [], [], [], []

    for i in idx_cmp:
        wav, npy = path_map[i]
        if not wav.exists():
            continue
        A = mel_variant_A(npy)
        B = mel_variant_B(wav)
        C = mel_variant_C(wav, b, a)

        if A.shape != B.shape:
            logger.error(f"  Shape mismatch at index {i}: A {A.shape} vs B {B.shape}")
            continue

        dif_AB.append(float(np.abs(A - B).max()))
        dif_AC.append(float(np.abs(A - C).max()))
        corr_AB.append(float(np.corrcoef(A.ravel(), B.ravel())[0, 1]))
        corr_AC.append(float(np.corrcoef(A.ravel(), C.ravel())[0, 1]))

    dif_AB, dif_AC = np.array(dif_AB), np.array(dif_AC)
    corr_AB, corr_AC = np.array(corr_AB), np.array(corr_AC)

    logger.info(f"  Cycles compared: {len(dif_AB)}")
    logger.info("")
    logger.info("  A (.npy) vs B (WAV, no filter)")
    logger.info(f"    Max absolute difference: mean {dif_AB.mean():.6f}, "
                f"worst {dif_AB.max():.6f}")
    logger.info(f"    Correlation:             mean {corr_AB.mean():.6f}, "
                f"worst {corr_AB.min():.6f}")
    logger.info("")
    logger.info("  A (.npy) vs C (WAV with filter and RMS, as in script 40)")
    logger.info(f"    Max absolute difference: mean {dif_AC.mean():.6f}, "
                f"worst {dif_AC.max():.6f}")
    logger.info(f"    Correlation:             mean {corr_AC.mean():.6f}, "
                f"worst {corr_AC.min():.6f}")

    # Part 2: full test-set metrics with each input variant
    section("PART 2 - METRICS ON THE FULL TEST SET")

    model_set = load_models(base_dir, device)
    if not model_set:
        logger.error("  No checkpoints. Aborted.")
        return 1

    results_data, row_list = {}, []

    for name_text, constructor in [
        ("A_npy_precomputed", lambda w, n: mel_variant_A(n)),
        ("B_wav_no_filter",   lambda w, n: mel_variant_B(w)),
        ("C_wav_with_filter",   lambda w, n: mel_variant_C(w, b, a)),
    ]:
        logger.info("")
        logger.info(f"  Variant {name_text} ...")

        mels, valid_vals = [], []
        for i, (wav, npy) in enumerate(path_map):
            if not wav.exists():
                continue
            mels.append(constructor(wav, npy))
            valid_vals.append(i)

        X = torch.from_numpy(np.stack(mels)).unsqueeze(1)
        y_true = y_true_full[np.array(valid_vals)]

        ic, au = [], []
        for seed, model in model_set.items():
            probs = run_inference(model, X, device)
            y_pred = np.argmax(probs, axis=1)
            icbhi, se, sp = icbhi_official(y_true, y_pred)
            auc = binary_auc(y_true, 1.0 - probs[:, 0])
            ic.append(icbhi)
            au.append(auc)
            row_list.append({"variant": name_text, "seed": seed, "icbhi": icbhi,
                          "se": se, "sp": sp, "auc_bin": auc})
            logger.info(f"    seed {seed:>3d}: ICBHI {icbhi:.4f}  AUC {auc:.4f}")

        m_ic, s_ic = float(np.mean(ic)), float(np.std(ic, ddof=1))
        m_au, s_au = float(np.mean(au)), float(np.std(au, ddof=1))
        results_data[name_text] = {"icbhi_mean": m_ic, "icbhi_sd": s_ic,
                              "auc_mean": m_au, "auc_sd": s_au}

        d_ic, d_au = abs(m_ic - OFFICIAL_ICBHI), abs(m_au - OFFICIAL_AUC)
        mark = "MATCH" if (d_ic <= TOLERANCE and d_au <= TOLERANCE) else "DIFFERS"
        logger.info(f"    MEAN: ICBHI {m_ic:.4f} +/- {s_ic:.4f}   "
                    f"AUC {m_au:.4f} +/- {s_au:.4f}   -> {mark}")

    # Part 3: verdict
    section("PART 3 - VERDICT")

    A = results_data["A_npy_precomputed"]
    B = results_data["B_wav_no_filter"]

    A_ok = (abs(A["icbhi_mean"] - OFFICIAL_ICBHI) <= TOLERANCE and
            abs(A["auc_mean"] - OFFICIAL_AUC) <= TOLERANCE)
    B_ok = (abs(B["icbhi_mean"] - OFFICIAL_ICBHI) <= TOLERANCE and
            abs(B["auc_mean"] - OFFICIAL_AUC) <= TOLERANCE)
    AB_equal = len(dif_AB) > 0 and dif_AB.max() < 0.01

    logger.info(f"  Reference (script 18): ICBHI {OFFICIAL_ICBHI:.3f}  AUC {OFFICIAL_AUC:.3f}")
    logger.info("")
    logger.info(f"  A reproduces the reference ..... {'YES' if A_ok else 'NO'}")
    logger.info(f"  B reproduces the reference ..... {'YES' if B_ok else 'NO'}")
    logger.info(f"  A and B numerically equal ...... {'YES' if AB_equal else 'NO'}")
    logger.info("")

    if not A_ok:
        logger.error("  CAUSE A: the precomputed .npy, which is the exact input behind the")
        logger.error("  reference results, does NOT reproduce them with this code. The fault")
        logger.error("  lies in the metrics or checkpoint loading of these scripts, not in")
        logger.error("  the front end. The noise experiment is suspended until resolved.")
        logger.error("  The reference results are NOT affected.")
    elif A_ok and B_ok and AB_equal:
        logger.info("  CAUSE B confirmed. The metrics are correct and the reconstruction")
        logger.info("  from the WAV is faithful. Script 40 failed because it re-applied")
        logger.info("  the filter and RMS normalization to segments that were already")
        logger.info("  conditioned. Fix the noise injection and rerun.")
    elif A_ok and not AB_equal:
        logger.error("  Possible CAUSE C. The metrics are correct, but the mel")
        logger.error("  reconstructed from the WAV does NOT match the precomputed .npy.")
        logger.error("  Check whether the .npy files come from these same segments or from")
        logger.error("  an earlier version of the preprocessing. This goes beyond the")
        logger.error("  noise experiment and must be examined separately.")
    else:
        logger.warning("  Unexpected combination. Inspect Part 2 row by row.")

    pd.DataFrame(row_list).to_csv(out_dir / "mel_diagnosis.csv", index=False,
                               float_format="%.6f")
    with open(out_dir / "mel_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "reference": {"icbhi": OFFICIAL_ICBHI, "auc": OFFICIAL_AUC},
            "results": results_data,
            "elementwise_comparison": {
                "n": int(len(dif_AB)),
                "A_vs_B_maxdiff_mean": float(dif_AB.mean()) if len(dif_AB) else None,
                "A_vs_B_maxdiff_worst": float(dif_AB.max()) if len(dif_AB) else None,
                "A_vs_B_corr_mean": float(corr_AB.mean()) if len(corr_AB) else None,
                "A_vs_C_maxdiff_mean": float(dif_AC.mean()) if len(dif_AC) else None,
                "A_vs_C_corr_mean": float(corr_AC.mean()) if len(corr_AC) else None,
            },
            "verdict": {"A_ok": bool(A_ok), "B_ok": bool(B_ok),
                          "AB_equal": bool(AB_equal)},
        }, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("  mel_diagnosis.csv and mel_diagnosis.json written")
    logger.info("=" * 74)
    logger.info("  END OF SCRIPT 42")
    logger.info("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
