#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 08: PyTorch Dataset and DataLoaders for the Mel-spectrogram classifiers.

Maps each respiratory cycle of a split CSV to its Mel-spectrogram .npy file and serves
(1, n_mels, time_steps) tensors with labels for 4-class (Normal/Crackle/Wheeze/Both) or
binary (Normal vs abnormal) classification. Class imbalance is handled with a
WeightedRandomSampler plus class weights for the loss. Optional on-the-fly augmentation
presets (train set only) range from none to aggressive; the clean baseline uses none,
because spectrogram transforms can erase or fabricate adventitious sounds.

The module is imported by the training scripts (create_dataloaders,
create_kfold_dataloaders, RespiratoryDataset, AUGMENTATION_REGISTRY). Run directly, it
performs an integrity test of the data pipeline.

Inputs:  data/splits/<strategy>/{train,val,test}_cycles.csv,
         data/features/mel_spectrograms/*.npy
Outputs (integrity test only): results/metrics/dataset_config.json,
         results/figures/augmentation_comparison.png,
         results/figures/augmentation_presets_comparison.png, dataset_dataloader_log.txt
Usage:
    python 08_dataset_dataloader.py
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("=" * 70)
    print("WARNING: PyTorch is not installed.")
    print("Install with: pip install torch torchvision")
    print("The script will run in verification mode without PyTorch.")
    print("=" * 70)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dataset_dataloader_log.txt', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_DIR = BASE_DIR / "data" / "features"
MEL_DIR = FEATURES_DIR / "mel_spectrograms"
SPLITS_DIR = BASE_DIR / "data" / "splits"

FEATURE_METADATA_CSV = FEATURES_DIR / "feature_metadata.csv"

FIGURES_DIR = BASE_DIR / "results" / "figures"
RESULTS_DIR = BASE_DIR / "results" / "metrics"


DATASET_CONFIG = {
    # 4-class classification (default)
    "class_map_4": {
        "Normal":  0,
        "Crackle": 1,
        "Wheeze":  2,
        "Both":    3,
    },
    "class_names_4": ["Normal", "Crackle", "Wheeze", "Both"],

    # Binary classification for screening: Normal (0) vs abnormal (1)
    "class_map_2": {
        "Normal":  0,
        "Crackle": 1,
        "Wheeze":  1,
        "Both":    1,
    },
    "class_names_2": ["Normal", "Abnormal"],

    # Per-sample min-max normalization of the dB spectrograms to [0, 1]
    "normalize_spectrograms": True,

    # Script 06 pads/truncates every cycle to a fixed duration, so all
    # spectrograms share the same shape
    "expected_n_mels": 128,

    "batch_size": 32,
    "num_workers": 0,      # 0 = load in the main process (safest on Windows)
    "pin_memory": True,

    # Balances classes without modifying the data
    "use_weighted_sampling": True,
}


# Augmentation transforms. Each one takes a numpy spectrogram (n_mels, time_steps) and
# returns a modified copy; they are applied sequentially and only to the training set.
# Defaults are deliberately more conservative than SpecAugment for speech.


class TimeMasking:
    """
    Time masking (SpecAugment, Park et al., 2019): zero a random block of consecutive frames.

    Forces the model not to rely on a single instant of the cycle. A large mask could
    remove isolated crackles, so the mask is limited to 15 % of the width
    (~0.3-1.2 s on a 2-8 s cycle).
    """

    def __init__(self, max_mask_pct=0.15, num_masks=1, p=0.5):
        """
        max_mask_pct: maximum masked fraction of the time axis (0.2-0.4 in SpecAugment for speech).
        num_masks: number of masked blocks. p: probability of applying the transform.
        """
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.p = p
        self.name = "TimeMasking"

    def __call__(self, spectrogram):
        """Apply time masking; returns an array with the same shape."""
        if np.random.random() > self.p:
            return spectrogram

        spec = spectrogram.copy()
        n_mels, time_steps = spec.shape
        max_mask_width = int(time_steps * self.max_mask_pct)

        if max_mask_width < 1:
            return spec

        for _ in range(self.num_masks):
            mask_width = np.random.randint(1, max_mask_width + 1)
            mask_start = np.random.randint(0, max(1, time_steps - mask_width))
            spec[:, mask_start:mask_start + mask_width] = 0.0

        return spec

    def __repr__(self):
        return (f"TimeMasking(max_mask_pct={self.max_mask_pct}, "
                f"num_masks={self.num_masks}, p={self.p})")


class FrequencyMasking:
    """
    Frequency masking (SpecAugment, Park et al., 2019): zero a random block of Mel bands.

    The riskiest transform for respiratory audio: wheezes have fundamentals around
    100-1000 Hz and a mask over that band can turn a pathological example into one that
    looks normal. Limited to 10 % of the bands (~13 of 128) and p=0.3.
    """

    def __init__(self, max_mask_pct=0.10, num_masks=1, p=0.3):
        """
        max_mask_pct: maximum masked fraction of Mel bands (0.15-0.27 in SpecAugment).
        num_masks: number of masked blocks. p: probability of applying the transform.
        """
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.p = p
        self.name = "FrequencyMasking"

    def __call__(self, spectrogram):
        if np.random.random() > self.p:
            return spectrogram

        spec = spectrogram.copy()
        n_mels, time_steps = spec.shape
        max_mask_height = int(n_mels * self.max_mask_pct)

        if max_mask_height < 1:
            return spec

        for _ in range(self.num_masks):
            mask_height = np.random.randint(1, max_mask_height + 1)
            mask_start = np.random.randint(0, max(1, n_mels - mask_height))
            spec[mask_start:mask_start + mask_height, :] = 0.0

        return spec

    def __repr__(self):
        return (f"FrequencyMasking(max_mask_pct={self.max_mask_pct}, "
                f"num_masks={self.num_masks}, p={self.p})")


class GaussianNoise:
    """
    Additive Gaussian noise on the spectrogram.

    Simulates ambient noise and the higher noise floor of a low-cost MEMS microphone.
    Excessive noise can mask fine crackles or mimic crackle-like artefacts, so the std is
    kept small (0.001-0.01 on a [0, 1] spectrogram).
    """

    def __init__(self, std_range=(0.001, 0.01), p=0.5):
        """std_range: (min, max) noise std, sampled uniformly. p: probability of applying the transform."""
        self.std_range = std_range
        self.p = p
        self.name = "GaussianNoise"

    def __call__(self, spectrogram):
        if np.random.random() > self.p:
            return spectrogram

        spec = spectrogram.copy()
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        noise = np.random.normal(0, std, spec.shape).astype(spec.dtype)
        spec = spec + noise

        # A Mel spectrogram has no negative values
        spec = np.clip(spec, 0.0, None)

        return spec

    def __repr__(self):
        return f"GaussianNoise(std_range={self.std_range}, p={self.p})"


class TimeShift:
    """
    Circular shift along the time axis.

    Cycle segmentation does not always start at the same respiratory phase, so the model
    should be invariant to the starting point. Lowest-risk transform: it neither removes
    information nor creates artefacts. Limited to 20 % of the width.
    """

    def __init__(self, max_shift_pct=0.20, p=0.5):
        """max_shift_pct: maximum shift as a fraction of the width, in either direction. p: probability."""
        self.max_shift_pct = max_shift_pct
        self.p = p
        self.name = "TimeShift"

    def __call__(self, spectrogram):
        if np.random.random() > self.p:
            return spectrogram

        spec = spectrogram.copy()
        n_mels, time_steps = spec.shape
        max_shift = int(time_steps * self.max_shift_pct)

        if max_shift < 1:
            return spec

        shift = np.random.randint(-max_shift, max_shift + 1)
        spec = np.roll(spec, shift, axis=1)

        return spec

    def __repr__(self):
        return f"TimeShift(max_shift_pct={self.max_shift_pct}, p={self.p})"


class GainPerturbation:
    """
    Multiply the whole spectrogram by a random gain.

    Models amplitude variation from stethoscope pressure, chest-wall thickness or AGC.
    A uniform gain preserves all spectral and temporal relations; range limited to ±20 %.
    """

    def __init__(self, gain_range=(0.8, 1.2), p=0.5):
        """gain_range: (min, max) multiplicative factor. p: probability of applying the transform."""
        self.gain_range = gain_range
        self.p = p
        self.name = "GainPerturbation"

    def __call__(self, spectrogram):
        if np.random.random() > self.p:
            return spectrogram

        spec = spectrogram.copy()
        gain = np.random.uniform(self.gain_range[0], self.gain_range[1])
        spec = spec * gain

        spec = np.clip(spec, 0.0, None)

        return spec

    def __repr__(self):
        return f"GainPerturbation(gain_range={self.gain_range}, p={self.p})"


class RandomCrop:
    """
    Keep a random contiguous crop of the time axis and zero-pad it back on the right.

    A crop that misses the adventitious event would train with a wrong label; keeping
    85 % of the width makes this unlikely, since events tend to repeat and wheezes are
    sustained.
    """

    def __init__(self, crop_pct=0.85, p=0.3):
        """crop_pct: fraction of the width kept. p: probability of applying the transform."""
        self.crop_pct = crop_pct
        self.p = p
        self.name = "RandomCrop"

    def __call__(self, spectrogram):
        if np.random.random() > self.p:
            return spectrogram

        spec = spectrogram.copy()
        n_mels, time_steps = spec.shape
        crop_width = int(time_steps * self.crop_pct)

        if crop_width >= time_steps or crop_width < 1:
            return spec

        max_start = time_steps - crop_width
        start = np.random.randint(0, max_start + 1)
        cropped = spec[:, start:start + crop_width]

        result = np.zeros_like(spec)
        result[:, :crop_width] = cropped

        return result

    def __repr__(self):
        return f"RandomCrop(crop_pct={self.crop_pct}, p={self.p})"


# Augmentation presets of increasing aggressiveness.
AUGMENTATION_REGISTRY = {
    "none": {
        "description": (
            "No augmentation. Clean baseline. "
            "Spectrograms are used as extracted. "
            "Recommended as the first experiment."
        ),
        "transforms": [],
        "risk_level": "None",
        "expected_impact": "Reference baseline",
    },

    "conservative": {
        "description": (
            "Conservative augmentation suited to medical audio. "
            "Low-risk transforms only: TimeShift + GainPerturbation. "
            "They neither create artefacts nor destroy diagnostic information. "
            "Recommended as the second experiment."
        ),
        "transforms": [
            TimeShift(max_shift_pct=0.15, p=0.5),
            GainPerturbation(gain_range=(0.85, 1.15), p=0.5),
        ],
        "risk_level": "Low",
        "expected_impact": "Marginal robustness gain, minimal risk",
    },

    "moderate": {
        "description": (
            "Moderate SpecAugment-style augmentation. "
            "Adds conservative TimeMasking + mild GaussianNoise. "
            "Transforms used in RespireNet (Gairola et al., 2021). "
            "Third experiment, for comparison."
        ),
        "transforms": [
            TimeShift(max_shift_pct=0.15, p=0.5),
            GainPerturbation(gain_range=(0.85, 1.15), p=0.5),
            TimeMasking(max_mask_pct=0.10, num_masks=1, p=0.4),
            GaussianNoise(std_range=(0.001, 0.008), p=0.4),
        ],
        "risk_level": "Moderate",
        "expected_impact": "Possible generalization gain, controlled risk",
    },

    "aggressive": {
        "description": (
            "Aggressive augmentation, for experimentation only. "
            "Includes FrequencyMasking, which can destroy diagnostic "
            "information of wheezes. Not intended for the main "
            "result. Used to show whether excessive augmentation "
            "degrades performance on medical audio "
            "(informative negative result)."
        ),
        "transforms": [
            TimeShift(max_shift_pct=0.20, p=0.5),
            GainPerturbation(gain_range=(0.8, 1.2), p=0.5),
            TimeMasking(max_mask_pct=0.15, num_masks=2, p=0.5),
            FrequencyMasking(max_mask_pct=0.10, num_masks=1, p=0.3),
            GaussianNoise(std_range=(0.002, 0.015), p=0.5),
            RandomCrop(crop_pct=0.85, p=0.3),
        ],
        "risk_level": "High",
        "expected_impact": "Possible degradation; evidence of augmentation limits",
    },
}


class RespiratoryDataset(Dataset):
    """
    PyTorch Dataset of respiratory cycles.

    Each item is loaded from its .npy file, min-max normalized to [0, 1] (optional),
    augmented (training only) and returned as a (1, n_mels, time_steps) float tensor
    with an integer label.
    """

    def __init__(
        self,
        split_csv_path,
        mel_dir,
        class_mode="4class",
        normalize=True,
        augment=False,
        augment_preset="conservative",
        custom_transforms=None,
    ):
        """
        split_csv_path: split CSV with columns cycle_id, cycle_class and (optionally) mel_filename.
        mel_dir: directory of the Mel-spectrogram .npy files.
        class_mode: "4class" (Normal/Crackle/Wheeze/Both) or "binary" (Normal vs abnormal).
        normalize: min-max normalize each spectrogram to [0, 1].
        augment: apply augmentation (training set only).
        augment_preset: key of AUGMENTATION_REGISTRY.
        custom_transforms: list of transforms that overrides augment_preset.
        Rows whose .npy file is missing or whose class is unknown are skipped.
        """
        self.mel_dir = Path(mel_dir)
        self.normalize = normalize
        self.augment = augment
        self.class_mode = class_mode

        if class_mode == "binary":
            self.class_map = DATASET_CONFIG["class_map_2"]
            self.class_names = DATASET_CONFIG["class_names_2"]
            self.n_classes = 2
        else:
            self.class_map = DATASET_CONFIG["class_map_4"]
            self.class_names = DATASET_CONFIG["class_names_4"]
            self.n_classes = 4

        df = pd.read_csv(split_csv_path)

        self.samples = []  # (npy_path, label_int)
        self.labels = []   # labels only, for weighted sampling
        self.cycle_ids = []

        skipped = 0
        for _, row in df.iterrows():
            if "mel_filename" in row and pd.notna(row["mel_filename"]):
                mel_filename = row["mel_filename"]
            else:
                mel_filename = f"{row['cycle_id']}_mel.npy"

            mel_path = self.mel_dir / mel_filename

            if not mel_path.exists():
                skipped += 1
                continue

            class_name = row["cycle_class"]
            if class_name not in self.class_map:
                skipped += 1
                continue

            label = self.class_map[class_name]

            self.samples.append((mel_path, label))
            self.labels.append(label)
            self.cycle_ids.append(row["cycle_id"])

        self.labels = np.array(self.labels)

        if skipped > 0:
            logger.warning(
                f"  Dataset: {skipped} samples skipped "
                f"(.npy file not found or invalid class)"
            )

        if augment and custom_transforms is not None:
            self.transforms = custom_transforms
        elif augment and augment_preset in AUGMENTATION_REGISTRY:
            self.transforms = AUGMENTATION_REGISTRY[augment_preset]["transforms"]
        else:
            self.transforms = []

        self._compute_stats()

        logger.info(
            f"  Dataset created: {len(self.samples)} samples, "
            f"{self.n_classes} classes ({class_mode}), "
            f"augment={'Yes (' + augment_preset + ')' if augment else 'No'}"
        )

    def _compute_stats(self):
        """Compute class counts and inverse-frequency class weights, and log the distribution."""
        self.class_counts = Counter(self.labels.tolist())
        self.class_weights = {}
        total = len(self.labels)

        for cls_idx in range(self.n_classes):
            count = self.class_counts.get(cls_idx, 0)
            # Inverse frequency: rare classes are sampled more often
            self.class_weights[cls_idx] = total / max(count * self.n_classes, 1)

        for cls_idx in range(self.n_classes):
            count = self.class_counts.get(cls_idx, 0)
            pct = 100 * count / max(total, 1)
            weight = self.class_weights[cls_idx]
            logger.info(
                f"    Class {cls_idx} ({self.class_names[cls_idx]:>8s}): "
                f"{count:5d} ({pct:5.1f}%), weight: {weight:.3f}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return (spectrogram tensor of shape (1, n_mels, time_steps), int label)."""
        mel_path, label = self.samples[idx]

        spectrogram = np.load(mel_path).astype(np.float32)

        if self.normalize:
            spec_min = spectrogram.min()
            spec_max = spectrogram.max()
            if spec_max - spec_min > 1e-6:
                spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
            else:
                spectrogram = np.zeros_like(spectrogram)

        if self.augment and self.transforms:
            for transform in self.transforms:
                spectrogram = transform(spectrogram)

        # Add the channel dimension: (n_mels, time_steps) -> (1, n_mels, time_steps)
        if TORCH_AVAILABLE:
            spectrogram_tensor = torch.from_numpy(spectrogram).unsqueeze(0)
            return spectrogram_tensor, label
        else:
            return spectrogram[np.newaxis, :, :], label

    def get_sample_weights(self):
        """
        Per-sample weights (the weight of each sample's class) for WeightedRandomSampler.

        Preferred over static oversampling: no duplicated data, and a different
        resampled set in every epoch.
        """
        weights = np.array([
            self.class_weights[label] for label in self.labels
        ])
        return weights


def create_dataloaders(
    strategy="strategy2_custom",
    class_mode="4class",
    augment_preset="none",
    batch_size=None,
    num_workers=None,
    use_weighted_sampling=None,
    pin_memory=None,
):
    """
    Build train/val/test datasets and DataLoaders for a split strategy.

    strategy: "strategy1_official" (official ICBHI split), "strategy2_custom"
        (StratifiedGroupKFold, no patient leakage) or "strategy3_kfold" (use
        create_kfold_dataloaders). class_mode: "4class" or "binary".
    augment_preset: AUGMENTATION_REGISTRY key, applied to train only.
    batch_size, num_workers, use_weighted_sampling, pin_memory: None uses DATASET_CONFIG.
    Returns a dict with the three loaders and datasets, class_names, n_classes,
    class_weights (tensor for the loss, normalized to sum n_classes), input_shape and
    config (record of the configuration used).
    """

    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required to create DataLoaders. "
            "Install with: pip install torch torchvision"
        )

    bs = batch_size or DATASET_CONFIG["batch_size"]
    nw = num_workers if num_workers is not None else DATASET_CONFIG["num_workers"]
    ws = use_weighted_sampling if use_weighted_sampling is not None else DATASET_CONFIG["use_weighted_sampling"]
    pm = pin_memory if pin_memory is not None else DATASET_CONFIG["pin_memory"]

    logger.info("=" * 70)
    logger.info("CREATING DATALOADERS")
    logger.info("=" * 70)
    logger.info(f"  Split strategy:      {strategy}")
    logger.info(f"  Class mode:          {class_mode}")
    logger.info(f"  Augmentation:        {augment_preset}")
    logger.info(f"  Batch size:          {bs}")
    logger.info(f"  Num workers:         {nw}")
    logger.info(f"  Weighted sampling:   {ws}")

    split_dir = SPLITS_DIR / strategy

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Run Script 07 first."
        )

    train_csv = split_dir / "train_cycles.csv"
    val_csv = split_dir / "val_cycles.csv"
    test_csv = split_dir / "test_cycles.csv"

    for csv_path, name in [(train_csv, "train"), (val_csv, "val"), (test_csv, "test")]:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV for {name} not found: {csv_path}"
            )

    if not MEL_DIR.exists():
        raise FileNotFoundError(
            f"Mel-spectrogram directory not found: {MEL_DIR}\n"
            f"Run Script 06 first."
        )

    logger.info(f"\n[TRAIN] Creating training dataset...")
    use_augment = augment_preset != "none"
    train_dataset = RespiratoryDataset(
        split_csv_path=train_csv,
        mel_dir=MEL_DIR,
        class_mode=class_mode,
        normalize=True,
        augment=use_augment,
        augment_preset=augment_preset,
    )

    logger.info(f"\n[VAL] Creating validation dataset...")
    val_dataset = RespiratoryDataset(
        split_csv_path=val_csv,
        mel_dir=MEL_DIR,
        class_mode=class_mode,
        normalize=True,
        augment=False,
    )

    logger.info(f"\n[TEST] Creating test dataset...")
    test_dataset = RespiratoryDataset(
        split_csv_path=test_csv,
        mel_dir=MEL_DIR,
        class_mode=class_mode,
        normalize=True,
        augment=False,
    )

    train_sampler = None
    train_shuffle = True

    if ws:
        logger.info("\n[SAMPLER] Configuring WeightedRandomSampler for train...")
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False  # the sampler already randomizes the order
        logger.info(f"  Weighted sampling enabled: {len(train_dataset)} samples/epoch")

    logger.info("\n[DATALOADERS] Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=pm,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm,
        drop_last=False,
    )

    # Class weights for nn.CrossEntropyLoss(weight=...), complementary to the sampler
    total_train = len(train_dataset)
    n_classes = train_dataset.n_classes
    class_weights_for_loss = torch.zeros(n_classes)

    for cls_idx in range(n_classes):
        count = train_dataset.class_counts.get(cls_idx, 0)
        if count > 0:
            class_weights_for_loss[cls_idx] = total_train / (n_classes * count)
        else:
            class_weights_for_loss[cls_idx] = 1.0

    # Normalize to sum n_classes so the effective learning rate stays stable
    class_weights_for_loss = class_weights_for_loss / class_weights_for_loss.sum() * n_classes

    logger.info(f"\n  Class weights for the loss:")
    for cls_idx in range(n_classes):
        logger.info(
            f"    {train_dataset.class_names[cls_idx]:>8s}: "
            f"{class_weights_for_loss[cls_idx]:.4f}"
        )

    logger.info(f"\n[CHECK] Loading one training batch...")
    try:
        batch_specs, batch_labels = next(iter(train_loader))
        logger.info(f"  Batch shape: {batch_specs.shape}")
        logger.info(f"  Labels shape: {batch_labels.shape}")
        logger.info(f"  Spec dtype: {batch_specs.dtype}")
        logger.info(f"  Spec range: [{batch_specs.min():.4f}, {batch_specs.max():.4f}]")
        logger.info(f"  Labels in batch: {Counter(batch_labels.numpy().tolist())}")
    except Exception as e:
        logger.error(f"  Error loading batch: {e}")
        raise

    logger.info("\n" + "=" * 70)
    logger.info("DATALOADERS CREATED")
    logger.info("=" * 70)
    logger.info(f"  Train:  {len(train_dataset):5d} samples -> {len(train_loader):4d} batches")
    logger.info(f"  Val:    {len(val_dataset):5d} samples -> {len(val_loader):4d} batches")
    logger.info(f"  Test:   {len(test_dataset):5d} samples -> {len(test_loader):4d} batches")
    logger.info(f"  Input shape: {batch_specs.shape[1:]} (C, H, W)")
    logger.info("=" * 70)

    config_record = {
        "strategy": strategy,
        "class_mode": class_mode,
        "n_classes": n_classes,
        "class_names": train_dataset.class_names,
        "augment_preset": augment_preset,
        "batch_size": bs,
        "num_workers": nw,
        "use_weighted_sampling": ws,
        "pin_memory": pm,
        "normalize": True,
        "input_shape": list(batch_specs.shape[1:]),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "class_weights_for_loss": class_weights_for_loss.tolist(),
        "augmentation_transforms": [
            str(t) for t in (train_dataset.transforms if use_augment else [])
        ],
    }

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "class_names": train_dataset.class_names,
        "n_classes": n_classes,
        "class_weights": class_weights_for_loss,
        "input_shape": tuple(batch_specs.shape[1:]),
        "config": config_record,
    }


def create_kfold_dataloaders(
    fold_number,
    class_mode="4class",
    augment_preset="none",
    batch_size=None,
    num_workers=None,
    use_weighted_sampling=None,
):
    """
    Build DataLoaders for one fold (1-5) of strategy3_kfold.

    Other arguments as in create_dataloaders; returns the same dict plus fold_number
    (without config).
    """

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")

    bs = batch_size or DATASET_CONFIG["batch_size"]
    nw = num_workers if num_workers is not None else DATASET_CONFIG["num_workers"]
    ws = use_weighted_sampling if use_weighted_sampling is not None else DATASET_CONFIG["use_weighted_sampling"]
    pm = DATASET_CONFIG["pin_memory"]

    fold_dir = SPLITS_DIR / "strategy3_kfold"

    train_csv = fold_dir / f"fold_{fold_number}_train.csv"
    val_csv = fold_dir / f"fold_{fold_number}_val.csv"
    test_csv = fold_dir / f"fold_{fold_number}_test.csv"

    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Not found: {csv_path}")

    logger.info(f"\n{'='*70}")
    logger.info(f"CREATING DATALOADERS: K-Fold, Fold {fold_number}")
    logger.info(f"{'='*70}")

    use_augment = augment_preset != "none"

    train_dataset = RespiratoryDataset(
        train_csv, MEL_DIR, class_mode=class_mode,
        augment=use_augment, augment_preset=augment_preset,
    )
    val_dataset = RespiratoryDataset(
        val_csv, MEL_DIR, class_mode=class_mode, augment=False,
    )
    test_dataset = RespiratoryDataset(
        test_csv, MEL_DIR, class_mode=class_mode, augment=False,
    )

    train_sampler = None
    train_shuffle = True
    if ws:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=train_shuffle,
        sampler=train_sampler, num_workers=nw, pin_memory=pm, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pm,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=pm,
    )

    total_train = len(train_dataset)
    n_classes = train_dataset.n_classes
    class_weights_for_loss = torch.zeros(n_classes)
    for cls_idx in range(n_classes):
        count = train_dataset.class_counts.get(cls_idx, 0)
        class_weights_for_loss[cls_idx] = total_train / (n_classes * max(count, 1))
    class_weights_for_loss = class_weights_for_loss / class_weights_for_loss.sum() * n_classes

    batch_specs, batch_labels = next(iter(train_loader))

    logger.info(f"  Train: {len(train_dataset)} -> {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_dataset)} -> {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_dataset)} -> {len(test_loader)} batches")
    logger.info(f"  Input shape: {batch_specs.shape[1:]}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "class_names": train_dataset.class_names,
        "n_classes": n_classes,
        "class_weights": class_weights_for_loss,
        "input_shape": tuple(batch_specs.shape[1:]),
        "fold_number": fold_number,
    }


def run_integrity_test():
    """
    Integrity test of the data pipeline.

    Checks the .npy files and their shapes and value ranges, builds the strategy2_custom
    DataLoaders without augmentation, applies every augmentation preset, saves two
    comparison figures and writes results/metrics/dataset_config.json. Returns True on success.
    """
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRITY TEST: data pipeline")
    logger.info("=" * 70)
    start_time = time.time()

    logger.info("\n[TEST 1] Checking .npy files...")

    if not MEL_DIR.exists():
        logger.error(f"  Directory not found: {MEL_DIR}")
        return False

    npy_files = list(MEL_DIR.glob("*.npy"))
    logger.info(f"  Found: {len(npy_files)} .npy files")

    if len(npy_files) == 0:
        logger.error("  No .npy files. Run Script 06 first.")
        return False

    logger.info("\n[TEST 2] Checking spectrogram shapes...")

    shapes = []
    sample_indices = np.random.choice(
        len(npy_files), min(100, len(npy_files)), replace=False
    )

    for idx in sample_indices:
        spec = np.load(npy_files[idx])
        shapes.append(spec.shape)

    unique_shapes = set(shapes)
    logger.info(f"  Unique shapes (out of {len(sample_indices)} samples): {unique_shapes}")

    if len(unique_shapes) == 1:
        logger.info(f"  All samples have the same shape: {shapes[0]}")
    else:
        logger.warning(f"  WARNING: variable shapes: {unique_shapes}")
        logger.warning("  This can break batching in the DataLoader.")
        logger.warning("  Check that Script 06 applied padding/truncation correctly.")

    logger.info("\n[TEST 3] Checking value ranges...")

    test_spec = np.load(npy_files[0])
    logger.info(f"  Example: {npy_files[0].name}")
    logger.info(f"  Shape: {test_spec.shape}")
    logger.info(f"  dtype: {test_spec.dtype}")
    logger.info(f"  Range: [{test_spec.min():.4f}, {test_spec.max():.4f}]")
    logger.info(f"  Mean: {test_spec.mean():.4f}, Std: {test_spec.std():.4f}")

    logger.info("\n[TEST 4] Creating DataLoaders (baseline, no augmentation)...")

    if TORCH_AVAILABLE:
        try:
            result = create_dataloaders(
                strategy="strategy2_custom",
                class_mode="4class",
                augment_preset="none",
                batch_size=16,
            )

            train_loader = result["train_loader"]
            val_loader = result["val_loader"]
            test_loader = result["test_loader"]

            logger.info(f"\n  DataLoaders created")
            logger.info(f"  Input shape: {result['input_shape']}")

            logger.info(f"\n  Iterating 3 training batches...")
            for i, (specs, labels) in enumerate(train_loader):
                if i >= 3:
                    break
                logger.info(
                    f"    Batch {i+1}: specs={specs.shape}, "
                    f"labels={labels.shape}, "
                    f"range=[{specs.min():.3f}, {specs.max():.3f}]"
                )

        except Exception as e:
            logger.error(f"  Error creating DataLoaders: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        logger.warning("  PyTorch not available; skipping the DataLoader test")

    logger.info("\n[TEST 5] Testing augmentation transforms...")

    test_spec = np.load(npy_files[0]).astype(np.float32)
    test_spec = (test_spec - test_spec.min()) / (test_spec.max() - test_spec.min() + 1e-8)

    for preset_name, preset_config in AUGMENTATION_REGISTRY.items():
        transforms = preset_config["transforms"]
        if not transforms:
            logger.info(f"  Preset '{preset_name}': no transforms (baseline)")
            continue

        logger.info(f"\n  Preset '{preset_name}' ({preset_config['risk_level']}):")
        for t in transforms:
            augmented = t(test_spec)
            diff = np.abs(augmented - test_spec).mean()
            logger.info(
                f"    {t.name:20s}: mean_diff={diff:.6f}, "
                f"shape={augmented.shape}"
            )

    logger.info("\n[TEST 6] Generating augmentation figures...")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Data Augmentation for Respiratory Sounds: Technique Comparison\n"
            "Mel-Spectrograms",
            fontsize=14, fontweight='bold'
        )

        axes[0, 0].imshow(test_spec, aspect='auto', origin='lower', cmap='magma')
        axes[0, 0].set_title("Original (No Augmentation)", fontsize=11)
        axes[0, 0].set_ylabel("Mel band")

        techniques = [
            ("Time Masking", TimeMasking(max_mask_pct=0.15, p=1.0)),
            ("Frequency Masking", FrequencyMasking(max_mask_pct=0.10, p=1.0)),
            ("Gaussian Noise", GaussianNoise(std_range=(0.005, 0.01), p=1.0)),
            ("Time Shift", TimeShift(max_shift_pct=0.20, p=1.0)),
            ("Gain Perturbation", GainPerturbation(gain_range=(0.7, 1.3), p=1.0)),
        ]

        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        np.random.seed(42)  # reproducible figure

        for (name, transform), (r, c) in zip(techniques, positions):
            augmented = transform(test_spec)
            axes[r, c].imshow(augmented, aspect='auto', origin='lower', cmap='magma')
            axes[r, c].set_title(f"{name}", fontsize=11)

            # Relative change with respect to the original spectrogram
            diff_pct = 100 * np.abs(augmented - test_spec).sum() / (test_spec.sum() + 1e-8)
            axes[r, c].text(
                0.02, 0.98, f"Δ={diff_pct:.1f}%",
                transform=axes[r, c].transAxes,
                verticalalignment='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        for ax in axes.flat:
            ax.set_xlabel("Time frame")

        plt.tight_layout()
        fig_path = FIGURES_DIR / "augmentation_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Figure saved: {fig_path}")

        fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
        fig2.suptitle(
            "Augmentation Presets: From Conservative to Aggressive\n"
            "Mel-Spectrograms",
            fontsize=13, fontweight='bold'
        )

        np.random.seed(42)

        for ax, (preset_name, preset_config) in zip(axes2, AUGMENTATION_REGISTRY.items()):
            spec_aug = test_spec.copy()
            for t in preset_config["transforms"]:
                spec_aug = t(spec_aug)

            ax.imshow(spec_aug, aspect='auto', origin='lower', cmap='magma')
            ax.set_title(
                f"{preset_name.capitalize()}\n"
                f"Risk: {preset_config['risk_level']}",
                fontsize=10
            )
            ax.set_xlabel("Time frame")
            ax.set_ylabel("Mel band")

        plt.tight_layout()
        fig2_path = FIGURES_DIR / "augmentation_presets_comparison.png"
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Figure saved: {fig2_path}")

    except Exception as e:
        logger.warning(f"  Error generating figures: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRITY TEST COMPLETED")
    logger.info(f"Time: {elapsed:.1f} seconds")
    logger.info("=" * 70)

    config_output = {
        "script": "08_dataset_dataloader.py",
        "timestamp": datetime.now().isoformat(),
        "n_npy_files": len(npy_files),
        "shapes_found": [str(s) for s in unique_shapes],
        "augmentation_presets": {
            name: {
                "description": cfg["description"],
                "risk_level": cfg["risk_level"],
                "n_transforms": len(cfg["transforms"]),
                "transforms": [str(t) for t in cfg["transforms"]],
            }
            for name, cfg in AUGMENTATION_REGISTRY.items()
        },
        "dataset_config": DATASET_CONFIG,
    }

    config_path = RESULTS_DIR / "dataset_config.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config_output, f, indent=2, default=str)
    logger.info(f"Configuration saved: {config_path}")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Script 08: Dataset and DataLoader")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    print()
    print("Running as an INTEGRITY TEST.")
    print("For normal use, import it from the training script:")
    print()
    print("    from script_08_dataset_dataloader import create_dataloaders")
    print("    loaders = create_dataloaders(augment_preset='none')")
    print()
    print("=" * 70 + "\n")

    success = run_integrity_test()

    if success:
        print("\n" + "=" * 70)
        print("INTEGRITY TEST PASSED: data pipeline working")
        print()
        print("NEXT STEPS:")
        print("  1. Run Script 09 with augment_preset='none' (baseline)")
        print("  2. Run Script 09 with augment_preset='conservative'")
        print("  3. Compare the results")
        print()
        print("RECOMMENDED EXPERIMENT PLAN:")
        print("  Exp 1: Baseline (no augmentation)          -> main result")
        print("  Exp 2: Conservative augmentation           -> robustness")
        print("  Exp 3: Moderate augmentation               -> exploration")
        print("  Exp 4: Aggressive augmentation             -> negative result")
        print()
        print("Together, the four runs characterize the impact")
        print("of data augmentation")
        print("on respiratory audio classification.")
        print("=" * 70)
    else:
        print("\nTEST FAILED: check the errors above")
