# -*- coding: utf-8 -*-
"""
Script 27: Zero-shot ICBHI -> SPRSound evaluation on inter_test (paired comparison).

Script 25 evaluated zero-shot transfer on all of SPRSound (8518 events), while Script 26
trained in-domain and evaluated on inter_test (1271 events, patient-disjoint). For the
zero-shot vs in-domain comparison to be paired, both figures must come from the same
test set, so this script evaluates the ICBHI checkpoints zero-shot on exactly the
inter_test used by Script 26. No training; inference only.

Inputs:  data/SPRSound/BioCAS2022/test2022_wav/,
         data/SPRSound/BioCAS2022/test2022_json/inter_test_json/,
         models/checkpoints/cnn_official_seed{seed}.pth
Outputs: sprsound_zeroshot_intertest_results.json, 27_sprsound_zeroshot_intertest_log.txt
Usage:
    python 27_sprsound_zeroshot_intertest.py
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration (identical to Script 26 for inter_test)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPR = os.path.join(os.path.dirname(BASE_DIR), "data", "SPRSound", "BioCAS2022")
TEST_WAV_DIR = os.path.join(SPR, "test2022_wav")
TEST_JSON_DIR = os.path.join(SPR, "test2022_json", "inter_test_json")

ICBHI_CKPT_DIR = os.path.join(os.path.dirname(BASE_DIR), "models", "checkpoints")
ICBHI_CKPT_PATTERN = "cnn_official_seed{seed}.pth"
SEEDS = [42, 123, 7, 99, 314]

MEL_CONFIG = {
    "target_sr": 16000, "bandpass_lowcut": 50, "bandpass_highcut": 2000, "bandpass_order": 5,
    "max_cycle_duration_sec": 8.0, "min_cycle_duration_sec": 0.5,
    "n_mels": 128, "n_fft": 2048, "hop_length": 512, "fmin": 50, "fmax": 2000,
    "power": 2.0, "top_db": 80.0,
}
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
NORMAL_IDX = 0
SPRSOUND_TO_ICBHI = {
    "normal": 0, "wheeze": 2, "rhonchi": 2, "stridor": 2,
    "fine crackle": 1, "coarse crackle": 1,
    "wheeze+crackle": 3, "wheeze & crackle": 3, "wheeze and crackle": 3,
}

LOG_PATH = os.path.join(BASE_DIR, "27_sprsound_zeroshot_intertest_log.txt")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("zeroshot_intertest")


# Feature pipeline (exact replica of Scripts 03/05/06; identical to 25/26)
def butter_bandpass_filter(audio, sr, lowcut, highcut, order):
    from scipy import signal
    nyq = 0.5 * sr
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, audio)


def fix_length(seg, sr, cfg):
    n = int(cfg["max_cycle_duration_sec"] * sr)
    if len(seg) < n:
        seg = np.concatenate([seg, np.zeros(n - len(seg), dtype=seg.dtype)])
    elif len(seg) > n:
        seg = seg[:n]
    return seg


def extract_logmel(seg, sr, cfg):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=seg, sr=sr, n_mels=cfg["n_mels"], n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
        fmin=cfg["fmin"], fmax=cfg["fmax"], power=cfg["power"])
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=cfg["top_db"])
    mn, mx = mel_db.min(), mel_db.max()
    return ((mel_db - mn) / (mx - mn) if mx - mn > 1e-6 else np.zeros_like(mel_db)).astype(np.float32)


def build_sprsound(wav_dir, json_dir, cfg):
    """Build inter_test from separate wav/json folders. Returns (X, y4)."""
    import librosa
    wavs = sorted(glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True))
    json_index = {os.path.splitext(os.path.basename(jp))[0]: jp
                  for jp in glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)}
    sr = cfg["target_sr"]; min_samples = int(cfg["min_cycle_duration_sec"] * sr)
    X, y4 = [], []
    for wav_path in wavs:
        jp = json_index.get(os.path.splitext(os.path.basename(wav_path))[0])
        if jp is None:
            continue
        try:
            events = json.load(open(jp, "r", encoding="utf-8")).get("event_annotation", [])
            if not events:
                continue
            audio, _ = librosa.load(wav_path, sr=sr, mono=True)
            audio = butter_bandpass_filter(audio, sr, cfg["bandpass_lowcut"],
                                           cfg["bandpass_highcut"], cfg["bandpass_order"]).astype(np.float32)
        except Exception as e:
            logger.warning(f"Error in {os.path.basename(wav_path)}: {e}")
            continue
        for ev in events:
            etype = str(ev.get("type", "")).strip().lower()
            if etype not in SPRSOUND_TO_ICBHI:
                continue
            s = max(0, int(round(float(ev["start"]) / 1000.0 * sr)))
            e = min(len(audio), int(round(float(ev["end"]) / 1000.0 * sr)))
            seg = audio[s:e]
            if len(seg) < min_samples:
                continue
            X.append(extract_logmel(fix_length(seg, sr, cfg), sr, cfg)[np.newaxis, :, :])
            y4.append(SPRSOUND_TO_ICBHI[etype])
    X = np.stack(X).astype(np.float32); y4 = np.array(y4, np.int64)
    logger.info(f"inter_test: {len(y4)} events")
    for c, name in enumerate(CLASS_NAMES):
        logger.info(f"    {name}: {int((y4 == c).sum())} ({100*(y4==c).mean():.1f}%)")
    return X, y4


# Model (identical to Script 09) and loading (identical to Scripts 18/25)
class ConvBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, padding=1, bias=False); self.bn1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(oc)
        self.relu = nn.ReLU(inplace=True); self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class RespiratoryNet(nn.Module):
    def __init__(self, n_classes=4, input_channels=1, base_filters=16,
                 n_conv_blocks=4, fc_hidden_dim=128, dropout_rate=0.5):
        super().__init__()
        blocks, ic = [], input_channels
        for i in range(n_conv_blocks):
            oc = base_filters * (2 ** i); blocks.append(ConvBlock(ic, oc)); ic = oc
        self.features = nn.Sequential(*blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(ic * 4 * 4, fc_hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate), nn.Linear(fc_hidden_dim, n_classes))

    def forward(self, x):
        return self.classifier(self.adaptive_pool(self.features(x)))


def load_cnn_checkpoint(path, device):
    ck = torch.load(str(path), map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = RespiratoryNet(
        n_classes=cfg.get("n_classes", 4), input_channels=cfg.get("input_channels", 1),
        base_filters=cfg.get("base_filters", 16), n_conv_blocks=cfg.get("n_conv_blocks", 4),
        fc_hidden_dim=cfg.get("fc_hidden_dim", 128), dropout_rate=cfg.get("dropout_rate", 0.5)).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    return model


@torch.no_grad()
def evaluate(model, X, device, batch_size=256):
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        probs.append(F.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    return probs, probs.argmax(axis=1)


def icbhi_official_4class(y_true, y_pred):
    n = np.sum(y_true == NORMAL_IDX)
    sp = np.sum((y_true == NORMAL_IDX) & (y_pred == NORMAL_IDX)) / n if n else 0.0
    a = np.sum(y_true != NORMAL_IDX)
    se = np.sum((y_true != NORMAL_IDX) & (y_pred == y_true)) / a if a else 0.0
    return (se + sp) / 2.0, se, sp


def binary_metrics(y_true, y_pred, probs):
    from sklearn.metrics import roc_auc_score
    yt, yp = (y_true != 0).astype(int), (y_pred != 0).astype(int)
    p_abn = 1.0 - probs[:, NORMAL_IDX]
    tp = int(np.sum((yt == 1) & (yp == 1))); fn = int(np.sum((yt == 1) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1))); tn = int(np.sum((yt == 0) & (yp == 0)))
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(yt, p_abn) if len(np.unique(yt)) > 1 else float("nan")
    return {"se": se, "sp": sp, "auc": float(auc)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info("ZERO-SHOT ICBHI→SPRSound ON inter_test (paired comparison with Script 26)")
    logger.info(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S} | Device: {device}")
    logger.info("=" * 70)

    X, y4 = build_sprsound(TEST_WAV_DIR, TEST_JSON_DIR, MEL_CONFIG)

    results = {"seeds": SEEDS, "n_test": int(len(y4)), "device": device, "per_seed": {}}
    icbhi_l, se_l, sp_l, bse, bsp, bauc = [], [], [], [], [], []
    for seed in SEEDS:
        ckpt = os.path.join(ICBHI_CKPT_DIR, ICBHI_CKPT_PATTERN.format(seed=seed))
        if not os.path.exists(ckpt):
            logger.error(f"Checkpoint not found: {ckpt}"); continue
        model = load_cnn_checkpoint(ckpt, device)
        probs, preds = evaluate(model, X, device)
        icbhi, se, sp = icbhi_official_4class(y4, preds)
        b = binary_metrics(y4, preds, probs)
        icbhi_l.append(icbhi); se_l.append(se); sp_l.append(sp)
        bse.append(b["se"]); bsp.append(b["sp"]); bauc.append(b["auc"])
        results["per_seed"][seed] = {"icbhi_official": float(icbhi), "se_official": float(se),
                                     "sp_official": float(sp), "binary": b}
        logger.info(f"seed {seed:>3}: ICBHI={icbhi:.4f} | BIN Se={b['se']:.3f} Sp={b['sp']:.3f} AUC={b['auc']:.3f}")

    def ms(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else None
    results["aggregated"] = {"icbhi_official": ms(icbhi_l), "se_official": ms(se_l), "sp_official": ms(sp_l),
                             "binary_se": ms(bse), "binary_sp": ms(bsp), "binary_auc": ms(bauc)}

    logger.info("-" * 70)
    logger.info(f"ZERO-SHOT on inter_test (n={len(y4)}, {len(bauc)} seeds):")
    logger.info(f"  Binary AUC  = {np.mean(bauc):.3f} ± {np.std(bauc):.3f}")
    logger.info(f"  Binary Se   = {np.mean(bse):.3f} ± {np.std(bse):.3f}")
    logger.info(f"  Binary Sp   = {np.mean(bsp):.3f} ± {np.std(bsp):.3f}")
    logger.info(f"  ICBHI 4-cl. = {np.mean(icbhi_l):.3f} ± {np.std(icbhi_l):.3f}")
    logger.info("  (compare with the Script 26 in-domain result on the SAME inter_test)")

    out = os.path.join(BASE_DIR, "sprsound_zeroshot_intertest_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()
