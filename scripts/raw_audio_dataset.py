# -*- coding: utf-8 -*-
"""
Raw-audio DataLoaders for PANNs CNN6 (used by Script 22).

Script 08 serves precomputed 16 kHz mel spectrograms (1, 128, 251), but PANNs CNN6
consumes raw 32 kHz audio and computes its own 64-band log-mel front-end with its
AudioSet pretraining parameters. This module therefore serves the waveform of each
cycle WAV produced by Script 05, resampled 16 kHz -> 32 kHz (respiratory sounds lie
below 2 kHz, so upsampling neither adds nor removes diagnostic content) and fixed to
TARGET_SECONDS * 32000 samples. Split, classes, label mapping and the optional
WeightedRandomSampler replicate Script 08. __getitem__ returns (waveform float32 (L,), label).

Inputs:  data/splits/<strategy>/{train,val,test}_cycles.csv, data/segments/*.wav
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

# Classes (identical to Script 08)
CLASS_NAMES_4 = ["Normal", "Crackle", "Wheeze", "Both"]

# PANNs audio parameters (must match the pretrained checkpoint)
PANNS_SR = 32000
TARGET_SECONDS = 8.0                      # Script 05 cycles are padded to 8 s
TARGET_LENGTH = int(TARGET_SECONDS * PANNS_SR)   # 256000 samples


def _resolve_columns(df):
    """Find the file-name column and the label column of a split CSV."""
    fname_col = None
    for c in ["cycle_filename", "filename", "file", "wav", "cycle_file"]:
        if c in df.columns:
            fname_col = c
            break
    if fname_col is None:
        fname_col = df.columns[0]

    # 'cycle_class' is the canonical label column (Script 08) and takes priority.
    label_col = None
    for c in ["cycle_class", "label_4class", "label_4", "label", "class_4", "class", "y"]:
        if c in df.columns:
            label_col = c
            break
    return fname_col, label_col


# Same class map as DATASET_CONFIG['class_map_4'] in Script 08.
CLASS_MAP_4 = {"Normal": 0, "Crackle": 1, "Wheeze": 2, "Both": 3}


def _label_to_index(value):
    """Map a label (cycle_class text or int) to 0..3 with the Script 08 map, so labels stay aligned."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value).strip()
    if s in CLASS_MAP_4:
        return CLASS_MAP_4[s]
    # Tolerate case and plural variants
    low = s.lower()
    alt = {"normal": 0, "0": 0, "crackle": 1, "crackles": 1, "1": 1,
           "wheeze": 2, "wheezes": 2, "2": 2, "both": 3, "3": 3}
    if low in alt:
        return alt[low]
    raise ValueError(f"Unrecognized label in class column: {value!r} "
                     f"(expected one of {list(CLASS_MAP_4)})")


class RawAudioCycleDataset(Dataset):
    """
    Raw waveform of each respiratory cycle, for PANNs.

    split_csv_path : split CSV (train/val/test) from Script 07.
    segments_dir   : folder with the cycle WAVs from Script 05.
    target_sr      : output sample rate (32000 for PANNs).
    target_length  : fixed length in samples (pad/crop).
    normalize_peak : if True, scale each waveform to peak |1.0| (levels are
                     homogenized; the PANNs bn0 layer handles the rest).
    """

    def __init__(self, split_csv_path, segments_dir, target_sr=PANNS_SR,
                 target_length=TARGET_LENGTH, normalize_peak=True):
        self.segments_dir = Path(segments_dir)
        self.target_sr = target_sr
        self.target_length = target_length
        self.normalize_peak = normalize_peak

        df = pd.read_csv(split_csv_path)
        self.fname_col, self.label_col = _resolve_columns(df)
        if self.label_col is None:
            raise ValueError(
                f"No label column found in {split_csv_path}. "
                f"Columns: {list(df.columns)}"
            )

        self.samples = []
        n_missing = 0
        for _, row in df.iterrows():
            fname = str(row[self.fname_col])
            label = _label_to_index(row[self.label_col])
            wav_path = self.segments_dir / fname
            if not wav_path.exists():
                n_missing += 1
                continue
            self.samples.append((wav_path, label))

        if n_missing > 0:
            logger.warning(f"  [RawAudio] {n_missing} .wav files not found in {self.segments_dir} "
                           f"(out of {len(df)} rows)")

        # Per-class counts (for the sampler and class weights)
        self.labels = [lab for _, lab in self.samples]
        self.class_counts = [int(np.sum(np.array(self.labels) == c)) for c in range(4)]
        logger.info(f"  [RawAudio] {len(self.samples)} cycles | per-class counts: {self.class_counts}")

    def __len__(self):
        return len(self.samples)

    def _load_waveform(self, wav_path):
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:                       # stereo -> mono
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != self.target_sr:                 # 16k -> 32k
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        elif len(audio) > self.target_length:
            audio = audio[:self.target_length]
        if self.normalize_peak:
            peak = np.max(np.abs(audio))
            if peak > 1e-8:
                audio = audio / peak
        return audio

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        audio = self._load_waveform(wav_path)
        return torch.from_numpy(audio).float(), int(label)


def create_raw_audio_dataloaders(splits_dir, segments_dir, strategy="strategy1_official",
                                 batch_size=32, num_workers=0,
                                 use_weighted_sampling=True, target_sr=PANNS_SR,
                                 target_length=TARGET_LENGTH):
    """
    Build raw-audio train/val/test DataLoaders for PANNs (same signature style as
    create_dataloaders in Script 08).

    splits_dir   : folder containing <strategy>/ with train/val/test_cycles.csv.
    segments_dir : folder with the cycle WAVs (data/segments).
    """
    splits_dir = Path(splits_dir)
    strat_dir = splits_dir / strategy
    csv_train = strat_dir / "train_cycles.csv"
    csv_val = strat_dir / "val_cycles.csv"
    csv_test = strat_dir / "test_cycles.csv"

    logger.info("=" * 70)
    logger.info("BUILDING RAW-AUDIO DATALOADERS (PANNs)")
    logger.info("=" * 70)
    logger.info(f"  Strategy:          {strategy}")
    logger.info(f"  Segments (.wav):   {segments_dir}")
    logger.info(f"  Sample rate:       {target_sr} Hz  (resampled from 16 kHz)")
    logger.info(f"  Fixed length:      {target_length} samples "
                f"({target_length/target_sr:.1f} s)")
    logger.info(f"  Batch size:        {batch_size}")
    logger.info(f"  Weighted sampling: {use_weighted_sampling}")

    logger.info("\n[TRAIN]")
    train_ds = RawAudioCycleDataset(csv_train, segments_dir, target_sr, target_length)
    logger.info("[VAL]")
    val_ds = RawAudioCycleDataset(csv_val, segments_dir, target_sr, target_length)
    logger.info("[TEST]")
    test_ds = RawAudioCycleDataset(csv_test, segments_dir, target_sr, target_length)

    # WeightedRandomSampler (same criterion as Script 08)
    train_sampler = None
    shuffle = True
    if use_weighted_sampling:
        counts = np.array(train_ds.class_counts, dtype=np.float64)
        per_class_w = 1.0 / np.maximum(counts, 1)
        sample_w = np.array([per_class_w[lab] for lab in train_ds.labels], dtype=np.float64)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w), replacement=True)
        shuffle = False
        logger.info(f"  [Sampler] WeightedRandomSampler enabled: {len(sample_w)} samples/epoch")

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                              shuffle=shuffle, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)

    # Class weights (reported only; available for a weighted CE)
    counts = np.array(train_ds.class_counts, dtype=np.float64)
    class_weights = {c: float(counts.sum() / max(counts[c] * 4, 1)) for c in range(4)}

    logger.info("\n  Train: %d | Val: %d | Test: %d" %
                (len(train_ds), len(val_ds), len(test_ds)))
    logger.info("=" * 70)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": CLASS_NAMES_4,
        "class_weights": class_weights,
        "class_counts": train_ds.class_counts,
        "target_length": target_length,
        "target_sr": target_sr,
    }
