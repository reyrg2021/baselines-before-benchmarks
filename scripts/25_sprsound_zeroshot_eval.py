# -*- coding: utf-8 -*-
"""
Script 25: Zero-shot cross-dataset evaluation of the ICBHI CNN on SPRSound.

Evaluates the five CNN-RespiratoryNet checkpoints trained on ICBHI 2017, without any
retraining, on the pediatric SPRSound database (different population, device and
class distribution). Per-cycle [0,1] normalization means no dataset statistics leak
between corpora. The primary metric is binary Normal-vs-Abnormal (the actual triage
decision); the 4-class ICBHI score is secondary because SPRSound has only ~34 "Both"
events (0.4%). Labels are mapped with the CAS/DAS convention: Wheeze, Rhonchi and
Stridor -> Wheeze; Fine/Coarse Crackle -> Crackle; Wheeze+Crackle -> Both. Preprocessing
replicates scripts 03/05/06 (16 kHz, 50-2000 Hz Butterworth bandpass, 8 s pad/truncate,
128-band log-mel, per-cycle [0,1] normalization, 1x128x251 tensors).

Inputs:  data/SPRSound/BioCAS2022/ (paired <name>.wav / <name>.json with event_annotation),
         models/checkpoints/cnn_official_seed{S}.pth
Outputs: scripts/25_sprsound_zeroshot_log.txt, scripts/sprsound_zeroshot_results.json,
         scripts/sprsound_cnn_seed{S}_probs.npy, scripts/sprsound_cnn_seed{S}_labels_preds.npz
Usage:
    python scripts/25_sprsound_zeroshot_eval.py
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder with SPRSound .wav/.json pairs (train and test together: all held out for the model)
SPRSOUND_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "SPRSound", "BioCAS2022")

CHECKPOINT_DIR = os.path.join(os.path.dirname(BASE_DIR), "models", "checkpoints")
CHECKPOINT_PATTERN = "cnn_official_seed{seed}.pth"

SEEDS = [42, 123, 7, 99, 314]

# Mel/segmentation parameters identical to scripts 03/05/06
MEL_CONFIG = {
    "target_sr": 16000,
    "bandpass_lowcut": 50,
    "bandpass_highcut": 2000,
    "bandpass_order": 5,
    "max_cycle_duration_sec": 8.0,
    "min_cycle_duration_sec": 0.5,   # shorter events are discarded, as in ICBHI
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "fmin": 50,
    "fmax": 2000,
    "power": 2.0,
    "top_db": 80.0,
}

# ICBHI class order, identical to scripts 07/08
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
NORMAL_IDX = 0

# SPRSound event type (lower-cased) -> ICBHI class index
SPRSOUND_TO_ICBHI = {
    "normal": 0,
    "wheeze": 2,
    "rhonchi": 2,          # CAS -> Wheeze
    "stridor": 2,          # CAS -> Wheeze
    "fine crackle": 1,     # DAS -> Crackle
    "coarse crackle": 1,   # DAS -> Crackle
    "wheeze+crackle": 3,   # Both
    "wheeze & crackle": 3, # name variant
    "wheeze and crackle": 3,
}

LOG_PATH = os.path.join(BASE_DIR, "25_sprsound_zeroshot_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sprsound_zeroshot")


def butter_bandpass_filter(audio, sr, lowcut, highcut, order):
    """Zero-phase Butterworth bandpass (filtfilt), as in script 03."""
    from scipy import signal
    nyquist = 0.5 * sr
    b, a = signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return signal.filtfilt(b, a, audio)


def fix_length(cycle_audio, sr, cfg):
    """Zero-pad at the end or truncate to 8 s, as in script 05."""
    max_samples = int(cfg["max_cycle_duration_sec"] * sr)
    if len(cycle_audio) < max_samples:
        pad = np.zeros(max_samples - len(cycle_audio), dtype=cycle_audio.dtype)
        cycle_audio = np.concatenate([cycle_audio, pad])
    elif len(cycle_audio) > max_samples:
        cycle_audio = cycle_audio[:max_samples]
    return cycle_audio


def extract_logmel(cycle_audio, sr, cfg):
    """Log-mel spectrogram with per-cycle [0,1] normalization, as in script 06."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=cycle_audio, sr=sr,
        n_mels=cfg["n_mels"], n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
        fmin=cfg["fmin"], fmax=cfg["fmax"], power=cfg["power"],
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=cfg["top_db"])
    mn, mx = mel_db.min(), mel_db.max()
    if mx - mn > 1e-6:
        mel_norm = (mel_db - mn) / (mx - mn)
    else:
        mel_norm = np.zeros_like(mel_db)
    return mel_norm.astype(np.float32)


# Model definition identical to script 09 so that state_dict keys match
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class RespiratoryNet(nn.Module):
    def __init__(self, n_classes=4, input_channels=1, base_filters=16,
                 n_conv_blocks=4, fc_hidden_dim=128, dropout_rate=0.5):
        super().__init__()
        conv_blocks, in_ch = [], input_channels
        for i in range(n_conv_blocks):
            out_ch = base_filters * (2 ** i)
            conv_blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*conv_blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 4 * 4, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_hidden_dim, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)


def load_cnn_checkpoint(path, device):
    """Load a CNN checkpoint (same pattern as script 18)."""
    ck = torch.load(str(path), map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = RespiratoryNet(
        n_classes=cfg.get("n_classes", 4),
        input_channels=cfg.get("input_channels", 1),
        base_filters=cfg.get("base_filters", 16),
        n_conv_blocks=cfg.get("n_conv_blocks", 4),
        fc_hidden_dim=cfg.get("fc_hidden_dim", 128),
        dropout_rate=cfg.get("dropout_rate", 0.5),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def build_sprsound_tensors(sprsound_dir, cfg):
    """
    Extract every annotated SPRSound event, apply the ICBHI pipeline and map its label.

    Returns (X, y4): X is (N, 1, 128, 251) float32 and y4 holds ICBHI class indices (0-3).
    """
    import librosa
    wavs = sorted(glob.glob(os.path.join(sprsound_dir, "**", "*.wav"), recursive=True))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found in {sprsound_dir}")

    # SPRSound stores .wav and .json in sibling folders (e.g. train2022_wav/, train2022_json/),
    # so JSON files are indexed by base name over the whole tree.
    json_index = {
        os.path.splitext(os.path.basename(jp))[0]: jp
        for jp in glob.glob(os.path.join(sprsound_dir, "**", "*.json"), recursive=True)
    }
    logger.info(f"WAVs found: {len(wavs)} | JSONs indexed: {len(json_index)}")

    X, y4 = [], []
    n_events, n_skipped_short, n_skipped_label, n_files_ok = 0, 0, 0, 0
    sr = cfg["target_sr"]
    min_samples = int(cfg["min_cycle_duration_sec"] * sr)

    for wav_path in wavs:
        json_path = json_index.get(os.path.splitext(os.path.basename(wav_path))[0])
        if json_path is None:
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            events = ann.get("event_annotation", [])
            if not events:
                continue
            # Whole recording is resampled to 16 kHz and bandpass-filtered before cropping events
            audio, _ = librosa.load(wav_path, sr=sr, mono=True)
            audio = butter_bandpass_filter(
                audio, sr, cfg["bandpass_lowcut"], cfg["bandpass_highcut"], cfg["bandpass_order"]
            ).astype(np.float32)
        except Exception as e:
            logger.warning(f"Error in {os.path.basename(wav_path)}: {e}")
            continue

        for ev in events:
            etype = str(ev.get("type", "")).strip().lower()
            if etype not in SPRSOUND_TO_ICBHI:
                n_skipped_label += 1
                continue
            start = int(round(float(ev["start"]) / 1000.0 * sr))
            end = int(round(float(ev["end"]) / 1000.0 * sr))
            start, end = max(0, start), min(len(audio), end)
            seg = audio[start:end]
            if len(seg) < min_samples:
                n_skipped_short += 1
                continue
            seg = fix_length(seg, sr, cfg)
            mel = extract_logmel(seg, sr, cfg)         # (128, 251)
            X.append(mel[np.newaxis, :, :])            # (1, 128, 251)
            y4.append(SPRSOUND_TO_ICBHI[etype])
            n_events += 1
        n_files_ok += 1

    logger.info(f"Files processed: {n_files_ok} | events used: {n_events} | "
                f"discarded (short): {n_skipped_short} | (unknown label): {n_skipped_label}")
    X = np.stack(X).astype(np.float32)
    y4 = np.array(y4, dtype=np.int64)
    for c, name in enumerate(CLASS_NAMES):
        logger.info(f"    {name}: {int((y4 == c).sum())} ({100*(y4==c).mean():.1f}%)")
    return X, y4


def icbhi_official_4class(y_true, y_pred):
    """Official ICBHI score: Sp = recall(Normal); Se = (C->C + W->W + B->B) / all abnormal."""
    n_total = np.sum(y_true == NORMAL_IDX)
    sp = np.sum((y_true == NORMAL_IDX) & (y_pred == NORMAL_IDX)) / n_total if n_total else 0.0
    a_total = np.sum(y_true != NORMAL_IDX)
    se = np.sum((y_true != NORMAL_IDX) & (y_pred == y_true)) / a_total if a_total else 0.0
    return (se + sp) / 2.0, se, sp


def binary_metrics(y_true, y_pred, probs):
    """Normal (0) vs Abnormal ({1,2,3}); the AUC uses P(abnormal) = 1 - P(Normal)."""
    from sklearn.metrics import roc_auc_score
    yt, yp = (y_true != 0).astype(int), (y_pred != 0).astype(int)
    p_abn = 1.0 - probs[:, NORMAL_IDX]
    tp = int(np.sum((yt == 1) & (yp == 1))); fn = int(np.sum((yt == 1) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1))); tn = int(np.sum((yt == 0) & (yp == 0)))
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(yt, p_abn) if len(np.unique(yt)) > 1 else float("nan")
    return {"se": se, "sp": sp, "auc": float(auc)}


@torch.no_grad()
def infer(model, X, device, batch_size=256):
    """Return softmax probabilities (N, 4) and argmax predictions (N,)."""
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        p = F.softmax(model(xb), dim=1).cpu().numpy()
        probs.append(p)
    probs = np.concatenate(probs, axis=0)
    return probs, probs.argmax(axis=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("ZERO-SHOT CROSS-DATASET EVALUATION: CNN(ICBHI) -> SPRSound")
    logger.info(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S} | Device: {device}")
    logger.info("=" * 70)

    # The SPRSound tensor is built once and shared by all seeds
    logger.info("Building SPRSound tensors (pipeline identical to ICBHI)...")
    X, y4 = build_sprsound_tensors(SPRSOUND_DIR, MEL_CONFIG)
    logger.info(f"Final tensor: X={X.shape}, y={y4.shape}")

    results = {"seeds": SEEDS, "n_events": int(len(y4)), "per_seed": {}, "device": device}
    icbhi_list, se_list, sp_list, bin_se, bin_sp, bin_auc = [], [], [], [], [], []

    for seed in SEEDS:
        ckpt = os.path.join(CHECKPOINT_DIR, CHECKPOINT_PATTERN.format(seed=seed))
        if not os.path.exists(ckpt):
            logger.error(f"Checkpoint not found: {ckpt}")
            continue
        model = load_cnn_checkpoint(ckpt, device)
        probs, preds = infer(model, X, device)

        icbhi, se, sp = icbhi_official_4class(y4, preds)
        b = binary_metrics(y4, preds, probs)
        icbhi_list.append(icbhi); se_list.append(se); sp_list.append(sp)
        bin_se.append(b["se"]); bin_sp.append(b["sp"]); bin_auc.append(b["auc"])

        results["per_seed"][seed] = {
            "icbhi_official": float(icbhi), "se_official": float(se), "sp_official": float(sp),
            "binary": b,
        }
        logger.info(f"seed {seed:>3}: ICBHI={icbhi:.4f} (Se={se:.3f} Sp={sp:.3f}) | "
                    f"BIN Se={b['se']:.3f} Sp={b['sp']:.3f} AUC={b['auc']:.3f}")

        # Per-seed arrays for ROC curves and operating-point analysis
        np.save(os.path.join(BASE_DIR, f"sprsound_cnn_seed{seed}_probs.npy"), probs)
        np.savez(os.path.join(BASE_DIR, f"sprsound_cnn_seed{seed}_labels_preds.npz"),
                 y_true=y4, y_pred=preds)

    def ms(v):  # mean and std
        return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else None

    results["aggregated"] = {
        "icbhi_official": ms(icbhi_list), "se_official": ms(se_list), "sp_official": ms(sp_list),
        "binary_se": ms(bin_se), "binary_sp": ms(bin_sp), "binary_auc": ms(bin_auc),
    }

    logger.info("-" * 70)
    logger.info("SUMMARY (mean ± SD over %d seeds):" % len(icbhi_list))
    logger.info(f"  Official ICBHI 4-class : {np.mean(icbhi_list):.3f} ± {np.std(icbhi_list):.3f}  (secondary)")
    logger.info(f"  BINARY   Se            : {np.mean(bin_se):.3f} ± {np.std(bin_se):.3f}")
    logger.info(f"  BINARY   Sp            : {np.mean(bin_sp):.3f} ± {np.std(bin_sp):.3f}")
    logger.info(f"  BINARY   AUC           : {np.mean(bin_auc):.3f} ± {np.std(bin_auc):.3f}  (PRIMARY)")
    logger.info("  Reference ICBHI->ICBHI: binary AUC 0.572")

    out = os.path.join(BASE_DIR, "sprsound_zeroshot_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()
