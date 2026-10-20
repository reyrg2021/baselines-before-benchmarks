# -*- coding: utf-8 -*-
"""
Script 26: In-domain training of the CNN on SPRSound (counterpart to the zero-shot test).

Script 25 showed that the CNN trained on ICBHI does not transfer zero-shot to
SPRSound (chance-level binary AUC). Here the same CNN-RespiratoryNet is trained
on SPRSound to tell domain shift apart from an architectural limitation: a
binary AUC clearly above chance in-domain points to domain shift.
Two modes (both run by default):
  - from_scratch:   random initialization, trained only on SPRSound.
  - finetune_icbhi: initialized from the ICBHI checkpoint of the same seed.
Training protocol identical to the ICBHI CNN (Adam, lr 3e-4, weight decay 1e-4,
batch 32, up to 80 epochs, early stopping on val_loss with patience 15,
ReduceLROnPlateau, class-balanced sampling, cross-entropy, seeds
{42, 123, 7, 99, 314}) and feature pipeline identical to Scripts 03/05/06.
Patient-independent splits: train2022 is split 85/15 by patient into train/val
(val used only for early stopping); the test set is inter_test, whose patients
are disjoint from train. intra_test is not used because it shares patients with
train.

Inputs:  data/SPRSound/BioCAS2022/{train2022_wav, train2022_json, test2022_wav,
         test2022_json/inter_test_json}, models/checkpoints/cnn_official_seed{seed}.pth
Outputs: sprsound_indomain_results.json, spr_indomain_{mode}_seed{seed}_probs.npy,
         spr_indomain_{mode}_seed{seed}_labels_preds.npz, 26_sprsound_indomain_log.txt
         (all in scripts/)
Usage:
    python scripts/26_sprsound_indomain_train.py
"""

import os
import sys
import json
import glob
import logging
import random
from datetime import datetime

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPR = os.path.join(os.path.dirname(BASE_DIR), "data", "SPRSound", "BioCAS2022")

TRAIN_WAV_DIR = os.path.join(SPR, "train2022_wav")
TRAIN_JSON_DIR = os.path.join(SPR, "train2022_json")
TEST_WAV_DIR = os.path.join(SPR, "test2022_wav")
TEST_JSON_DIR = os.path.join(SPR, "test2022_json", "inter_test_json")  # patient-independent

# Used only in finetune_icbhi mode
ICBHI_CKPT_DIR = os.path.join(os.path.dirname(BASE_DIR), "models", "checkpoints")
ICBHI_CKPT_PATTERN = "cnn_official_seed{seed}.pth"

MODES = ["from_scratch", "finetune_icbhi"]
SEEDS = [42, 123, 7, 99, 314]
SPLIT_SEED = 42          # fixes the patient-level train/val partition (same for all seeds)
VAL_PATIENT_FRAC = 0.15

# Hyperparameters of the ICBHI CNN
HP = {
    "lr": 3e-4, "weight_decay": 1e-4, "batch_size": 32, "max_epochs": 80,
    "patience": 15, "plateau_factor": 0.5, "plateau_patience": 5,
}

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

LOG_PATH = os.path.join(BASE_DIR, "26_sprsound_indomain_log.txt")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sprsound_indomain")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


# Feature pipeline (exact replica of Scripts 03/05/06)
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


def patient_id_from_name(fname):
    """The patient number is the first '_'-separated field of the file name."""
    return os.path.basename(fname).split("_")[0]


def build_sprsound(wav_dir, json_dir, cfg, tag=""):
    """Return (X[N,1,128,251], y4[N], patients[N]), one entry per annotated event."""
    import librosa
    wavs = sorted(glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True))
    json_index = {os.path.splitext(os.path.basename(jp))[0]: jp
                  for jp in glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)}
    sr = cfg["target_sr"]
    min_samples = int(cfg["min_cycle_duration_sec"] * sr)
    X, y4, pats = [], [], []
    n_ev, n_short, n_unk = 0, 0, 0

    for wav_path in wavs:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        jp = json_index.get(base)
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
            logger.warning(f"Error in {base}: {e}")
            continue
        pid = patient_id_from_name(base)
        for ev in events:
            etype = str(ev.get("type", "")).strip().lower()
            if etype not in SPRSOUND_TO_ICBHI:
                n_unk += 1
                continue
            s = max(0, int(round(float(ev["start"]) / 1000.0 * sr)))
            e = min(len(audio), int(round(float(ev["end"]) / 1000.0 * sr)))
            seg = audio[s:e]
            if len(seg) < min_samples:
                n_short += 1
                continue
            mel = extract_logmel(fix_length(seg, sr, cfg), sr, cfg)
            X.append(mel[np.newaxis, :, :]); y4.append(SPRSOUND_TO_ICBHI[etype]); pats.append(pid)
            n_ev += 1

    X = np.stack(X).astype(np.float32); y4 = np.array(y4, np.int64); pats = np.array(pats)
    logger.info(f"[{tag}] events={n_ev} (short discarded={n_short}, unknown label={n_unk}) "
                f"| patients={len(set(pats))}")
    for c, name in enumerate(CLASS_NAMES):
        logger.info(f"    [{tag}] {name}: {int((y4 == c).sum())} ({100*(y4==c).mean():.1f}%)")
    return X, y4, pats


# Model (same structure as Script 09)
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


def init_from_icbhi(model, seed, device):
    """Load the weights of the ICBHI checkpoint of the same seed (finetune mode)."""
    path = os.path.join(ICBHI_CKPT_DIR, ICBHI_CKPT_PATTERN.format(seed=seed))
    ck = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    return model


# Splits, sampler and training
def split_train_val_by_patient(patients, val_frac, seed):
    """Train/val partition disjoint by patient (no patient in both)."""
    uniq = sorted(set(patients.tolist()))
    rng = random.Random(seed); rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_pat = set(uniq[:n_val])
    val_idx = np.array([i for i, p in enumerate(patients) if p in val_pat], dtype=np.int64)
    tr_idx = np.array([i for i, p in enumerate(patients) if p not in val_pat], dtype=np.int64)
    assert not (set(patients[tr_idx]) & set(patients[val_idx])), "LEAKAGE: patient in both train and val"
    return tr_idx, val_idx


def make_balanced_loader(X, y, batch_size):
    """DataLoader with WeightedRandomSampler (class-balanced sampling)."""
    counts = np.bincount(y, minlength=len(CLASS_NAMES)).astype(np.float64)
    w_per_class = 1.0 / np.maximum(counts, 1)
    weights = w_per_class[y]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(y), replacement=True)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler)


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, X, device, batch_size=256):
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        probs.append(F.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    return probs, probs.argmax(axis=1)


def train_one(model, Xtr, ytr, Xval, yval, device):
    """Train with early stopping on val_loss; return the model with the best weights."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=HP["plateau_factor"],
                                                       patience=HP["plateau_patience"])
    crit = nn.CrossEntropyLoss()
    loader = make_balanced_loader(Xtr, ytr, HP["batch_size"])
    Xval_t = torch.from_numpy(Xval).to(device); yval_t = torch.from_numpy(yval).to(device)

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(HP["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for i in range(0, len(Xval), 256):
                vb = Xval_t[i:i + 256]; tb = yval_t[i:i + 256]
                vloss += crit(model(vb), tb).item() * len(vb)
            vloss /= len(Xval)
        sched.step(vloss)
        if vloss < best_val - 1e-5:
            best_val, best_state, patience = vloss, {k: v.detach().cpu().clone()
                                                     for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= HP["patience"]:
                logger.info(f"      early stopping at epoch {epoch} (val_loss={vloss:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# Metrics (same definitions as Scripts 18/25)
def icbhi_official_4class(y_true, y_pred):
    """Official ICBHI score: Se counts an abnormal cycle as detected only if its exact class is predicted."""
    n = np.sum(y_true == NORMAL_IDX)
    sp = np.sum((y_true == NORMAL_IDX) & (y_pred == NORMAL_IDX)) / n if n else 0.0
    a = np.sum(y_true != NORMAL_IDX)
    se = np.sum((y_true != NORMAL_IDX) & (y_pred == y_true)) / a if a else 0.0
    return (se + sp) / 2.0, se, sp


def binary_metrics(y_true, y_pred, probs):
    """Normal vs abnormal Se, Sp and AUC (score = 1 - P(Normal))."""
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
    logger.info("IN-DOMAIN TRAINING ON SPRSound (counterpart to zero-shot)")
    logger.info(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S} | Device: {device} | Modes: {MODES}")
    logger.info("=" * 70)

    logger.info("Building train2022 and inter_test (same pipeline as ICBHI)...")
    Xtr_all, ytr_all, pats_all = build_sprsound(TRAIN_WAV_DIR, TRAIN_JSON_DIR, MEL_CONFIG, tag="train2022")
    Xte, yte, pats_te = build_sprsound(TEST_WAV_DIR, TEST_JSON_DIR, MEL_CONFIG, tag="inter_test")

    # Leakage check: test patients must not appear in train
    leak = set(pats_all.tolist()) & set(pats_te.tolist())
    logger.info(f"Patients train={len(set(pats_all))}, test={len(set(pats_te))}, "
                f"intersection={len(leak)} (must be 0)")
    if leak:
        logger.warning(f"Warning: {len(leak)} patients in both train AND test: {sorted(leak)[:10]}")

    tr_idx, val_idx = split_train_val_by_patient(pats_all, VAL_PATIENT_FRAC, SPLIT_SEED)
    Xtr, ytr = Xtr_all[tr_idx], ytr_all[tr_idx]
    Xval, yval = Xtr_all[val_idx], ytr_all[val_idx]
    logger.info(f"Train: {len(ytr)} events / {len(set(pats_all[tr_idx]))} pat | "
                f"Val: {len(yval)} / {len(set(pats_all[val_idx]))} pat | Test: {len(yte)}")

    results = {"modes": MODES, "seeds": SEEDS, "n_test": int(len(yte)),
               "device": device, "per_mode": {}}

    for mode in MODES:
        logger.info("#" * 70)
        logger.info(f"MODE: {mode}")
        icbhi_l, se_l, sp_l, bse, bsp, bauc = [], [], [], [], [], []
        per_seed = {}
        for seed in SEEDS:
            set_all_seeds(seed)
            model = RespiratoryNet()
            if mode == "finetune_icbhi":
                try:
                    model = init_from_icbhi(model, seed, device)
                except Exception as e:
                    logger.error(f"Could not load ICBHI checkpoint for seed {seed}: {e}")
                    continue
            logger.info(f"  [{mode}] seed {seed}: training...")
            model, best_val = train_one(model, Xtr, ytr, Xval, yval, device)

            probs, preds = evaluate(model, Xte, device)
            icbhi, se, sp = icbhi_official_4class(yte, preds)
            b = binary_metrics(yte, preds, probs)
            icbhi_l.append(icbhi); se_l.append(se); sp_l.append(sp)
            bse.append(b["se"]); bsp.append(b["sp"]); bauc.append(b["auc"])
            per_seed[seed] = {"icbhi_official": float(icbhi), "se_official": float(se),
                              "sp_official": float(sp), "binary": b, "val_loss": float(best_val)}
            logger.info(f"  [{mode}] seed {seed}: ICBHI={icbhi:.4f} | "
                        f"BIN Se={b['se']:.3f} Sp={b['sp']:.3f} AUC={b['auc']:.3f}")
            np.save(os.path.join(BASE_DIR, f"spr_indomain_{mode}_seed{seed}_probs.npy"), probs)
            np.savez(os.path.join(BASE_DIR, f"spr_indomain_{mode}_seed{seed}_labels_preds.npz"),
                     y_true=yte, y_pred=preds)

        def ms(v):
            return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else None
        results["per_mode"][mode] = {
            "per_seed": per_seed,
            "aggregated": {"icbhi_official": ms(icbhi_l), "se_official": ms(se_l),
                           "sp_official": ms(sp_l), "binary_se": ms(bse),
                           "binary_sp": ms(bsp), "binary_auc": ms(bauc)},
        }
        logger.info(f"  >>> [{mode}] SUMMARY: ICBHI={np.mean(icbhi_l):.3f}±{np.std(icbhi_l):.3f} | "
                    f"binary AUC={np.mean(bauc):.3f}±{np.std(bauc):.3f}")

    logger.info("=" * 70)
    logger.info("FINAL COMPARISON (binary AUC):")
    logger.info("  zero-shot (scripts 25 and 27)  : see sprsound_zeroshot_intertest_results.json")
    for mode in MODES:
        agg = results["per_mode"].get(mode, {}).get("aggregated", {}).get("binary_auc")
        if agg:
            logger.info(f"  in-domain {mode:<16}: {agg['mean']:.3f} ± {agg['std']:.3f}")

    out = os.path.join(BASE_DIR, "sprsound_indomain_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {out}")


if __name__ == "__main__":
    main()
