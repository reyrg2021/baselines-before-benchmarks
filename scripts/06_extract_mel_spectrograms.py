#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 06: Mel-spectrogram extraction for every segmented respiratory cycle.

Converts each cycle WAV into a log-Mel spectrogram normalized to [0, 1], the direct
input of the CNN. The Mel scale gives finer resolution at low frequencies, where
crackles and wheezes lie. Parameters: n_mels=128; n_fft=2048 (128 ms at 16 kHz) and
hop_length=512 (32 ms, 75% overlap), a compromise between resolving brief crackles and
separating nearby wheeze tones; fmin/fmax = 50/2000 Hz, matching the preprocessing
band-pass filter.

Inputs:  data/segments/processed_cycles.csv, data/segments/*.wav
Outputs: data/features/mel_spectrograms/*_mel.npy, data/features/feature_metadata.csv,
         data/features/feature_extraction_config.json,
         results/metrics/feature_extraction_report.txt,
         results/figures/mel_spectrogram_examples.png
Usage:
    python 06_extract_mel_spectrograms.py
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# Silence librosa warnings about PySoundFile
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('feature_extraction_log.txt', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
SEGMENTS_DIR = BASE_DIR / "data" / "segments"
FEATURES_DIR = BASE_DIR / "data" / "features" / "mel_spectrograms"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

# Input files
CYCLES_CSV = SEGMENTS_DIR / "processed_cycles.csv"

# Output files
FEATURE_METADATA_CSV = BASE_DIR / "data" / "features" / "feature_metadata.csv"
REPORT_FILE = RESULTS_DIR / "feature_extraction_report.txt"
CONFIG_FILE = BASE_DIR / "data" / "features" / "feature_extraction_config.json"


FEATURE_CONFIG = {
    # Mel-spectrogram parameters
    "n_mels": 128,           # Number of Mel bands
    "n_fft": 2048,           # FFT window size (128 ms @ 16 kHz)
    "hop_length": 512,       # Hop between windows (32 ms @ 16 kHz)
    "fmin": 50,              # Minimum frequency (Hz), matches the band-pass filter
    "fmax": 2000,            # Maximum frequency (Hz), matches the band-pass filter
    "power": 2.0,            # Exponent for the power spectrogram

    # Log (dB) scale compresses the dynamic range and makes faint crackles visible
    "use_log_scale": True,
    "ref_power": 1.0,        # dB reference
    "top_db": 80.0,          # Maximum dynamic range in dB

    # Per-cycle [0, 1] normalization removes residual loudness differences between
    # cycles, so the CNN learns spectral patterns rather than absolute amplitude
    "normalize_per_cycle": True,

    # Expected sample rate
    "target_sr": 16000,

    # Parallel processing
    "n_workers": None,  # None = all available cores minus one
}


def compute_mel_spectrogram(audio_path, config=FEATURE_CONFIG):
    """
    Compute the normalized log-Mel spectrogram of one cycle WAV.

    Returns (mel_spec, metadata): a float32 array of shape (n_mels, time_frames) scaled to
    [0, 1], and a dict with its shape and pre-normalization dB statistics. An 8 s cycle at
    16 kHz with hop=512 yields 128000 // 512 + 1 = 251 frames, i.e. shape (128, 251).
    """
    audio, sr = sf.read(audio_path)

    # Keep only the first channel if the file is not mono
    if audio.ndim > 1:
        audio = audio[:, 0]

    # STFT + Mel filterbank
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=config["n_mels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        fmin=config["fmin"],
        fmax=config["fmax"],
        power=config["power"],
    )

    if config["use_log_scale"]:
        # 10 * log10(S / max(S)); top_db clips the range and avoids -inf in silent regions
        mel_spec_db = librosa.power_to_db(
            mel_spec,
            ref=np.max,
            top_db=config["top_db"],
        )
    else:
        mel_spec_db = mel_spec

    # Statistics before normalization (stored as metadata)
    raw_min = float(mel_spec_db.min())
    raw_max = float(mel_spec_db.max())
    raw_mean = float(mel_spec_db.mean())
    raw_std = float(mel_spec_db.std())

    if config["normalize_per_cycle"]:
        spec_min = mel_spec_db.min()
        spec_max = mel_spec_db.max()

        if spec_max - spec_min > 1e-6:  # Avoid division by zero
            mel_spec_norm = (mel_spec_db - spec_min) / (spec_max - spec_min)
        else:
            # Essentially silent audio
            mel_spec_norm = np.zeros_like(mel_spec_db)
    else:
        mel_spec_norm = mel_spec_db

    metadata = {
        "shape": mel_spec_norm.shape,
        "n_time_frames": mel_spec_norm.shape[1],
        "n_mel_bands": mel_spec_norm.shape[0],
        "raw_db_min": raw_min,
        "raw_db_max": raw_max,
        "raw_db_mean": raw_mean,
        "raw_db_std": raw_std,
        "norm_min": float(mel_spec_norm.min()),
        "norm_max": float(mel_spec_norm.max()),
    }

    return mel_spec_norm.astype(np.float32), metadata


def process_single_cycle(args):
    """
    Process one cycle and save its spectrogram as .npy (single-argument, picklable wrapper
    for ProcessPoolExecutor).

    args is a tuple (cycle_row_dict, segments_dir, features_dir, config). Returns a dict
    with the extraction metadata, or a dict with a non-empty "error" field on failure.
    """
    cycle_info, segments_dir, features_dir, config = args

    cycle_filename = cycle_info["cycle_filename"]
    audio_path = Path(segments_dir) / cycle_filename

    if not audio_path.exists():
        return {"error": f"Not found: {cycle_filename}", "cycle_id": cycle_info["cycle_id"]}

    try:
        mel_spec, spec_meta = compute_mel_spectrogram(audio_path, config)

        npy_filename = Path(cycle_filename).stem + "_mel.npy"
        npy_path = Path(features_dir) / npy_filename

        np.save(npy_path, mel_spec)

        return {
            "cycle_id": cycle_info["cycle_id"],
            "cycle_filename": cycle_filename,
            "mel_filename": npy_filename,
            "mel_shape_rows": spec_meta["n_mel_bands"],
            "mel_shape_cols": spec_meta["n_time_frames"],
            "raw_db_min": spec_meta["raw_db_min"],
            "raw_db_max": spec_meta["raw_db_max"],
            "raw_db_mean": spec_meta["raw_db_mean"],
            "raw_db_std": spec_meta["raw_db_std"],
            "error": None,
        }

    except Exception as e:
        return {
            "cycle_id": cycle_info["cycle_id"],
            "error": str(e),
        }


def extract_all_features(config=FEATURE_CONFIG):
    """
    Extract spectrograms for all cycles, write the feature metadata, the configuration
    record, an example figure and a statistics report. Returns the merged metadata DataFrame.
    """

    logger.info("=" * 70)
    logger.info("Mel-spectrogram extraction")
    logger.info("=" * 70)
    start_time_total = time.time()

    logger.info("\n[STEP 1] Preparing directories...")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "features").mkdir(parents=True, exist_ok=True)

    logger.info("\n[STEP 2] Loading segmented-cycle metadata...")

    if not CYCLES_CSV.exists():
        logger.error(f"Not found: {CYCLES_CSV}")
        logger.error("Run 05_segment_respiratory_cycles.py first")
        sys.exit(1)

    df_cycles = pd.read_csv(CYCLES_CSV)
    n_cycles = len(df_cycles)
    logger.info(f"  Cycles to process: {n_cycles}")

    logger.info("\n[STEP 3] Computing expected dimensions...")

    # For a cycle of max_duration_sec at target_sr with hop_length
    max_samples = int(8.0 * config["target_sr"])  # 128,000
    expected_frames = max_samples // config["hop_length"] + 1

    logger.info(f"  Expected Mel-spectrogram dimensions:")
    logger.info(f"    Mel bands (rows): {config['n_mels']}")
    logger.info(f"    Time frames (columns): ~{expected_frames}")
    logger.info(f"    Time resolution: {config['hop_length']/config['target_sr']*1000:.1f} ms/frame")
    logger.info(f"    FFT resolution: {config['n_fft']/config['target_sr']*1000:.1f} ms window")
    logger.info(f"    Overlap: {(1 - config['hop_length']/config['n_fft'])*100:.0f}%")
    logger.info(f"    Frequency range: {config['fmin']}-{config['fmax']} Hz")

    memory_per_spec = config["n_mels"] * expected_frames * 4  # float32 = 4 bytes
    total_memory_mb = n_cycles * memory_per_spec / (1024**2)
    logger.info(f"  Estimated total memory: {total_memory_mb:.1f} MB")

    logger.info("\n[STEP 4] Extracting Mel-spectrograms...")

    n_workers = config.get("n_workers")
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    logger.info(f"  Using {n_workers} workers for parallel processing")

    cycle_dicts = df_cycles.to_dict('records')
    process_args = [
        (cd, str(SEGMENTS_DIR), str(FEATURES_DIR), config)
        for cd in cycle_dicts
    ]

    results = []
    errors = []

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_single_cycle, args): i
                for i, args in enumerate(process_args)
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Mel-spectrograms",
                unit="cycle"
            ):
                result = future.result()
                if result and result.get("error"):
                    errors.append(result)
                elif result:
                    results.append(result)
    else:
        # Sequential processing (useful for debugging)
        for args in tqdm(process_args, desc="Mel-spectrograms", unit="cycle"):
            result = process_single_cycle(args)
            if result and result.get("error"):
                errors.append(result)
            elif result:
                results.append(result)

    logger.info(f"  Processed successfully: {len(results)}")
    logger.info(f"  Errors: {len(errors)}")

    if errors:
        logger.warning("  First 5 errors:")
        for err in errors[:5]:
            logger.warning(f"    {err.get('cycle_id', 'unknown')}: {err['error']}")

    logger.info("\n[STEP 5] Building feature metadata...")

    df_features = pd.DataFrame(results)

    # Merge with the original cycle metadata
    df_complete = df_cycles.merge(
        df_features[["cycle_id", "mel_filename", "mel_shape_rows", "mel_shape_cols",
                      "raw_db_min", "raw_db_max", "raw_db_mean", "raw_db_std"]],
        on="cycle_id",
        how="left"
    )

    df_complete.to_csv(FEATURE_METADATA_CSV, index=False)
    logger.info(f"  Feature metadata saved to: {FEATURE_METADATA_CSV}")

    config_record = {
        "script": "06_extract_mel_spectrograms.py",
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in config.items() if k != "n_workers"},
        "results": {
            "cycles_input": n_cycles,
            "cycles_processed": len(results),
            "cycles_errors": len(errors),
            "mel_shape": f"({config['n_mels']}, ~{expected_frames})",
        }
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_record, f, indent=2)

    logger.info("\n[STEP 7] Generating example figure...")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # One example per class
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Example Mel-Spectrograms per Class\n"
            f"(n_mels={config['n_mels']}, fmax={config['fmax']}Hz, hop={config['hop_length']})",
            fontsize=14, fontweight='bold'
        )

        classes = ["Normal", "Crackle", "Wheeze", "Both"]

        for ax, cls_name in zip(axes.flat, classes):
            cls_cycles = df_complete[df_complete["cycle_class"] == cls_name]

            if len(cls_cycles) == 0:
                ax.set_title(f"{cls_name} (no examples)")
                ax.axis('off')
                continue

            example = cls_cycles.iloc[0]
            mel_path = FEATURES_DIR / example["mel_filename"]

            if mel_path.exists():
                mel_spec = np.load(mel_path)

                time_axis = np.arange(mel_spec.shape[1]) * config["hop_length"] / config["target_sr"]

                im = ax.imshow(
                    mel_spec,
                    aspect='auto',
                    origin='lower',
                    cmap='magma',
                    extent=[0, time_axis[-1], config["fmin"], config["fmax"]],
                )
                ax.set_title(f"{cls_name}\n({example['cycle_id']})", fontsize=11)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
                plt.colorbar(im, ax=ax, label="Normalized [0,1]")
            else:
                ax.set_title(f"{cls_name} (file not found)")
                ax.axis('off')

        plt.tight_layout()
        fig_path = BASE_DIR / "results" / "figures" / "mel_spectrogram_examples.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Figure saved: {fig_path}")

    except Exception as e:
        logger.warning(f"  Could not generate figure: {e}")

    logger.info("\n[STEP 8] Generating statistics report...")

    elapsed = time.time() - start_time_total

    shape_cols = df_features["mel_shape_cols"].values if "mel_shape_cols" in df_features else []
    db_means = df_features["raw_db_mean"].values if "raw_db_mean" in df_features else []

    report_lines = [
        "=" * 70,
        "MEL-SPECTROGRAM EXTRACTION REPORT",
        "=" * 70,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run time: {elapsed:.1f} seconds ({elapsed/60:.1f} min)",
        f"Workers used: {n_workers}",
        "",
        "--- RESULTS ---",
        f"Input cycles: {n_cycles}",
        f"Mel-spectrograms generated: {len(results)}",
        f"Errors: {len(errors)}",
        f"Success rate: {100*len(results)/max(n_cycles,1):.1f}%",
        "",
        "--- MEL-SPECTROGRAM DIMENSIONS ---",
        f"Mel bands (rows): {config['n_mels']}",
        f"Time frames (columns): {int(np.median(shape_cols)) if len(shape_cols) > 0 else 'N/A'}",
        f"Column range: [{int(min(shape_cols))}, {int(max(shape_cols))}]" if len(shape_cols) > 0 else "",
        "",
        "--- PARAMETERS ---",
        f"n_fft: {config['n_fft']} ({config['n_fft']/config['target_sr']*1000:.1f} ms)",
        f"hop_length: {config['hop_length']} ({config['hop_length']/config['target_sr']*1000:.1f} ms)",
        f"fmin: {config['fmin']} Hz",
        f"fmax: {config['fmax']} Hz",
        f"Log scale: {'Yes' if config['use_log_scale'] else 'No'}",
        f"Dynamic range: {config['top_db']} dB",
        f"Per-cycle normalization: {'Yes' if config['normalize_per_cycle'] else 'No'}",
        "",
        "--- ENERGY STATISTICS (dB, before normalization) ---",
        f"Global mean: {np.mean(db_means):.1f} dB" if len(db_means) > 0 else "",
        f"Std across cycles: {np.std(db_means):.1f} dB" if len(db_means) > 0 else "",
        "",
        "--- DISK SPACE ---",
        f"Estimated size per spectrogram: {memory_per_spec/1024:.1f} KB",
        f"Estimated total size: {total_memory_mb:.1f} MB",
        "",
        "=" * 70,
        "CONCLUSION: features extracted successfully.",
        "Dataset ready for split creation and CNN training.",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"\n  Report saved to: {REPORT_FILE}")

    logger.info("\n" + "=" * 70)
    logger.info("FEATURE EXTRACTION COMPLETED")
    logger.info(f"   {len(results)} Mel-spectrograms of {config['n_mels']}x~{expected_frames}")
    logger.info(f"   Files in: {FEATURES_DIR}")
    logger.info(f"   Metadata in: {FEATURE_METADATA_CSV}")
    logger.info("=" * 70)

    return df_complete


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Mel-spectrogram extraction")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Segments directory: {SEGMENTS_DIR}")
    print(f"Features directory: {FEATURES_DIR}")
    print("=" * 70 + "\n")

    if not SEGMENTS_DIR.exists():
        print(f"ERROR: directory not found: {SEGMENTS_DIR}")
        print("Run 05_segment_respiratory_cycles.py first")
        sys.exit(1)

    if not CYCLES_CSV.exists():
        print(f"ERROR: file not found: {CYCLES_CSV}")
        sys.exit(1)

    df_result = extract_all_features()

    print("\n--- DataFrame preview ---")
    cols_preview = ["cycle_id", "cycle_class", "mel_filename",
                    "mel_shape_rows", "mel_shape_cols", "raw_db_mean"]
    available_cols = [c for c in cols_preview if c in df_result.columns]
    print(df_result[available_cols].head(10).to_string())
    print(f"\nShape: {df_result.shape}")
