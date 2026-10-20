#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 05: Segmentation of the ICBHI 2017 recordings into respiratory cycles.

Reads each recording's annotation file (start, end, crackles, wheezes), cuts every
annotated cycle from the preprocessed audio, zero-pads or truncates it to a fixed
duration, and saves it as an individual WAV file plus one metadata row per cycle.
Cycle names include the chest location ({patient}_{recording}_{location}_cycle{NNNN})
because recordings of the same patient at different chest locations can share
recording_idx; without the location, those cycles would overwrite each other on disk.

Inputs:  data/processed/*.wav, data/raw/*.txt, data/processed/preprocessing_metadata.csv
Outputs: data/segments/*.wav, data/segments/processed_cycles.csv,
         data/segments/segmentation_config.json,
         results/metrics/segmentation_report.txt, segmentation_log.txt
Usage:
    python 05_segment_respiratory_cycles.py
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('segmentation_log.txt', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SEGMENTS_DIR = BASE_DIR / "data" / "segments"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

ICBHI_DIR = BASE_DIR / "data" / "raw"

PREPROCESSING_METADATA = BASE_DIR / "data" / "processed" / "preprocessing_metadata.csv"
EXCLUDED_FILES = BASE_DIR / "data" / "processed" / "excluded_files.csv"

CYCLES_CSV = SEGMENTS_DIR / "processed_cycles.csv"
REPORT_FILE = RESULTS_DIR / "segmentation_report.txt"
CONFIG_FILE = SEGMENTS_DIR / "segmentation_config.json"


SEGMENTATION_CONFIG = {
    "target_sr": 16000,
    "max_cycle_duration_sec": 8.0,
    "min_cycle_duration_sec": 0.5,
    "padding_strategy": "zero_pad",
    "temporal_margin_sec": 0.0,
    "output_format": "wav",
    "output_subtype": "PCM_16",
}


# ICBHI metadata codes
CHEST_LOCATION_MAP = {
    "Al": "Anterior Left",
    "Ar": "Anterior Right",
    "Pl": "Posterior Left",
    "Pr": "Posterior Right",
    "Ll": "Lateral Left",
    "Lr": "Lateral Right",
    "Tc": "Trachea",
}

EQUIPMENT_MAP = {
    "Meditron":  "Meditron Electronic Stethoscope",
    "LittC2SE":  "Littmann Classic II SE",
    "Litt3200":  "Littmann 3200",
    "AKGC417L":  "AKG C417L Microphone",
}

# (crackles, wheezes) -> class
CYCLE_CLASSES = {
    (0, 0): "Normal",
    (1, 0): "Crackle",
    (0, 1): "Wheeze",
    (1, 1): "Both",
}


def parse_filename(filename):
    """
    Extract metadata from an ICBHI file name.
    Format: {patient_id}_{recording_idx}_{chest_location}_{acq_mode}_{equipment}.wav
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 5:
        logger.warning(f"Unexpected file name format: {filename} ({len(parts)} parts)")
        try:
            patient_id = int(parts[0])
            return {
                "patient_id": patient_id,
                "recording_idx": parts[1] if len(parts) > 1 else "unknown",
                "chest_location": parts[2] if len(parts) > 2 else "unknown",
                "acq_mode": parts[3] if len(parts) > 3 else "unknown",
                "equipment": parts[4] if len(parts) > 4 else "unknown",
            }
        except (ValueError, IndexError):
            return None

    try:
        return {
            "patient_id": int(parts[0]),
            "recording_idx": parts[1],
            "chest_location": parts[2],
            "acq_mode": parts[3],
            "equipment": parts[4],
        }
    except ValueError:
        logger.warning(f"Could not parse patient_id from: {filename}")
        return None


def parse_annotation_file(annotation_path):
    """
    Read and parse an ICBHI annotation file.
    Expected format (tab-separated, no header): start_time  end_time  crackles  wheezes
    """
    cycles = []

    try:
        with open(annotation_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 4:
                    parts = line.split()

                if len(parts) < 4:
                    logger.warning(
                        f"Malformed line {line_num} in {annotation_path}: '{line}'"
                    )
                    continue

                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    crackles = int(parts[2])
                    wheezes = int(parts[3])
                except ValueError as e:
                    logger.warning(
                        f"Error parsing values in {annotation_path}, "
                        f"line {line_num}: {e}"
                    )
                    continue

                if end <= start:
                    logger.warning(
                        f"Cycle with non-positive duration in {annotation_path}, "
                        f"line {line_num}: start={start}, end={end}"
                    )
                    continue

                if crackles not in (0, 1) or wheezes not in (0, 1):
                    logger.warning(
                        f"Labels out of range in {annotation_path}, "
                        f"line {line_num}: crackles={crackles}, wheezes={wheezes}"
                    )
                    crackles = 1 if crackles > 0 else 0
                    wheezes = 1 if wheezes > 0 else 0

                cycles.append({
                    "start": start,
                    "end": end,
                    "crackles": crackles,
                    "wheezes": wheezes,
                })

    except FileNotFoundError:
        logger.error(f"Annotation file not found: {annotation_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading {annotation_path}: {e}")
        return []

    for i in range(1, len(cycles)):
        if cycles[i]["start"] < cycles[i-1]["end"] - 0.001:
            logger.warning(
                f"Possible overlap in {annotation_path}: "
                f"cycle {i} start={cycles[i]['start']:.3f} < "
                f"cycle {i-1} end={cycles[i-1]['end']:.3f}"
            )

    return cycles


def extract_cycle_audio(audio_data, sr, start_time, end_time, config):
    """
    Cut one respiratory cycle from the recording and zero-pad or truncate it to
    max_cycle_duration_sec. Returns (audio, original duration, was_padded, was_truncated).
    """
    margin = config["temporal_margin_sec"]
    max_samples = int(config["max_cycle_duration_sec"] * sr)
    min_samples = int(config["min_cycle_duration_sec"] * sr)

    start_sample = max(0, int((start_time - margin) * sr))
    end_sample = min(len(audio_data), int((end_time + margin) * sr))

    cycle_audio = audio_data[start_sample:end_sample].copy()
    actual_duration = len(cycle_audio) / sr

    was_padded = False
    was_truncated = False

    if len(cycle_audio) < min_samples:
        logger.debug(
            f"Very short cycle ({actual_duration:.2f}s), "
            f"minimum required: {config['min_cycle_duration_sec']}s"
        )

    if len(cycle_audio) < max_samples:
        padding = np.zeros(max_samples - len(cycle_audio), dtype=cycle_audio.dtype)
        cycle_audio = np.concatenate([cycle_audio, padding])
        was_padded = True
    elif len(cycle_audio) > max_samples:
        cycle_audio = cycle_audio[:max_samples]
        was_truncated = True

    return cycle_audio, actual_duration, was_padded, was_truncated


def segment_all_recordings(config=SEGMENTATION_CONFIG):
    """Segment all preprocessed recordings and write the cycle metadata, config and report."""

    logger.info("=" * 70)
    logger.info("Respiratory cycle segmentation")
    logger.info("=" * 70)
    start_time_total = time.time()

    logger.info("\n[STEP 1] Preparing output directories...")
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Segments directory: {SEGMENTS_DIR}")
    logger.info(f"  Reports directory:  {RESULTS_DIR}")

    logger.info("\n[STEP 2] Discovering preprocessed files...")

    processed_files = sorted(PROCESSED_DIR.glob("*.wav"))
    logger.info(f"  WAV files found in processed/: {len(processed_files)}")

    if len(processed_files) == 0:
        logger.error("No preprocessed files found. Check the path.")
        sys.exit(1)

    if PREPROCESSING_METADATA.exists():
        prep_meta = pd.read_csv(PREPROCESSING_METADATA)
        logger.info(f"  Preprocessing metadata loaded: {len(prep_meta)} records")
    else:
        prep_meta = None
        logger.warning("  preprocessing_metadata.csv not found")

    logger.info("\n[STEP 3] Segmenting respiratory cycles...")
    logger.info(f"  Configuration:")
    logger.info(f"    Max cycle duration: {config['max_cycle_duration_sec']}s")
    logger.info(f"    Min cycle duration: {config['min_cycle_duration_sec']}s")
    logger.info(f"    Padding: {config['padding_strategy']}")
    logger.info(f"    Temporal margin: {config['temporal_margin_sec']}s")

    all_cycles = []

    stats = {
        "files_processed": 0,
        "files_skipped_no_annotations": 0,
        "files_skipped_errors": 0,
        "total_cycles": 0,
        "cycles_padded": 0,
        "cycles_truncated": 0,
        "cycles_too_short": 0,
        "cycles_by_class": {"Normal": 0, "Crackle": 0, "Wheeze": 0, "Both": 0},
        "durations": [],
    }

    for processed_file in tqdm(processed_files, desc="Segmenting", unit="file"):

        file_meta = parse_filename(processed_file.name)
        if file_meta is None:
            logger.warning(f"  Skipping {processed_file.name}: unrecognized format")
            stats["files_skipped_errors"] += 1
            continue

        annotation_name = processed_file.stem + ".txt"
        annotation_path = ICBHI_DIR / annotation_name

        if not annotation_path.exists():
            logger.warning(f"  No annotations: {annotation_name}")
            stats["files_skipped_no_annotations"] += 1
            continue

        cycles = parse_annotation_file(annotation_path)
        if len(cycles) == 0:
            logger.warning(f"  No valid cycles in: {annotation_name}")
            stats["files_skipped_no_annotations"] += 1
            continue

        try:
            audio_data, sr = sf.read(processed_file)
        except Exception as e:
            logger.error(f"  Error reading {processed_file.name}: {e}")
            stats["files_skipped_errors"] += 1
            continue

        if sr != config["target_sr"]:
            logger.warning(
                f"  Unexpected sample rate in {processed_file.name}: "
                f"{sr} Hz (expected {config['target_sr']} Hz)"
            )

        audio_duration = len(audio_data) / sr

        for cycle_idx, cycle_info in enumerate(cycles):

            if cycle_info["end"] > audio_duration + 0.1:
                logger.warning(
                    f"  Cycle {cycle_idx} in {processed_file.name} exceeds "
                    f"the audio duration: end={cycle_info['end']:.2f}s > "
                    f"audio={audio_duration:.2f}s"
                )
                cycle_info["end"] = audio_duration

            cycle_duration = cycle_info["end"] - cycle_info["start"]
            if cycle_duration < config["min_cycle_duration_sec"]:
                stats["cycles_too_short"] += 1
                logger.debug(
                    f"  Cycle {cycle_idx} very short ({cycle_duration:.2f}s) "
                    f"in {processed_file.name}"
                )

            cycle_audio, actual_dur, was_padded, was_truncated = extract_cycle_audio(
                audio_data, sr,
                cycle_info["start"], cycle_info["end"],
                config
            )

            # chest_location in the name avoids collisions between recordings of the
            # same patient that share recording_idx
            cycle_filename = (
                f"{file_meta['patient_id']}_"
                f"{file_meta['recording_idx']}_"
                f"{file_meta['chest_location']}_"
                f"cycle{cycle_idx:04d}.wav"
            )
            cycle_path = SEGMENTS_DIR / cycle_filename

            try:
                sf.write(
                    cycle_path,
                    cycle_audio,
                    sr,
                    subtype=config["output_subtype"]
                )
            except Exception as e:
                logger.error(f"  Error saving {cycle_filename}: {e}")
                continue

            cycle_class = CYCLE_CLASSES.get(
                (cycle_info["crackles"], cycle_info["wheezes"]),
                "Unknown"
            )

            # cycle_id also includes chest_location (same collision as the file name)
            cycle_record = {
                "cycle_id": (
                    f"{file_meta['patient_id']}_"
                    f"{file_meta['recording_idx']}_"
                    f"{file_meta['chest_location']}_"
                    f"c{cycle_idx:04d}"
                ),
                "cycle_filename": cycle_filename,
                "patient_id": file_meta["patient_id"],
                "recording_idx": file_meta["recording_idx"],
                "filename_original": processed_file.name,
                "cycle_index": cycle_idx,
                "start_time": cycle_info["start"],
                "end_time": cycle_info["end"],
                "duration": actual_dur,
                "duration_padded": config["max_cycle_duration_sec"],
                "was_padded": was_padded,
                "was_truncated": was_truncated,
                "has_crackles": cycle_info["crackles"],
                "has_wheezes": cycle_info["wheezes"],
                "cycle_class": cycle_class,
                "chest_location": file_meta["chest_location"],
                "chest_location_full": CHEST_LOCATION_MAP.get(
                    file_meta["chest_location"], file_meta["chest_location"]
                ),
                "acquisition_mode": file_meta["acq_mode"],
                "recording_equipment": file_meta["equipment"],
                "recording_equipment_full": EQUIPMENT_MAP.get(
                    file_meta["equipment"], file_meta["equipment"]
                ),
                "sample_rate": sr,
                "total_samples": len(cycle_audio),
            }

            all_cycles.append(cycle_record)

            stats["total_cycles"] += 1
            stats["durations"].append(actual_dur)
            stats["cycles_by_class"][cycle_class] = (
                stats["cycles_by_class"].get(cycle_class, 0) + 1
            )
            if was_padded:
                stats["cycles_padded"] += 1
            if was_truncated:
                stats["cycles_truncated"] += 1

        stats["files_processed"] += 1

    logger.info("\n[STEP 4] Building cycle metadata...")

    if len(all_cycles) == 0:
        logger.error("No cycles were generated. Check the input data.")
        sys.exit(1)

    df_cycles = pd.DataFrame(all_cycles)

    df_cycles = df_cycles.sort_values(
        ["patient_id", "filename_original", "cycle_index"]
    ).reset_index(drop=True)

    # Post-segmentation check for residual name collisions
    n_unique_ids = df_cycles["cycle_id"].nunique()
    n_unique_files = df_cycles["cycle_filename"].nunique()
    if n_unique_ids < len(df_cycles):
        logger.error(
            f"  WARNING: {len(df_cycles) - n_unique_ids} duplicate cycle_ids detected!"
        )
    if n_unique_files < len(df_cycles):
        logger.error(
            f"  WARNING: {len(df_cycles) - n_unique_files} duplicate cycle_filenames!"
        )
    if n_unique_ids == len(df_cycles) and n_unique_files == len(df_cycles):
        logger.info(f"  Check: 0 name collisions. All IDs are unique.")

    df_cycles.to_csv(CYCLES_CSV, index=False)
    logger.info(f"  Metadata saved to: {CYCLES_CSV}")
    logger.info(f"  Total cycles: {len(df_cycles)}")

    # Configuration record for reproducibility
    config_record = {
        "script": "05_segment_respiratory_cycles.py",
        "version": "v2_chest_location_fix",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "paths": {
            "processed_dir": str(PROCESSED_DIR),
            "icbhi_dir": str(ICBHI_DIR),
            "segments_dir": str(SEGMENTS_DIR),
        },
        "results": {
            "files_input": len(processed_files),
            "files_processed": stats["files_processed"],
            "total_cycles": stats["total_cycles"],
            "unique_cycle_ids": n_unique_ids,
            "unique_cycle_filenames": n_unique_files,
        }
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_record, f, indent=2)
    logger.info(f"  Configuration saved to: {CONFIG_FILE}")

    logger.info("\n[STEP 6] Building statistical report...")

    durations = np.array(stats["durations"])
    n_patients = df_cycles["patient_id"].nunique()
    elapsed = time.time() - start_time_total

    report_lines = [
        "=" * 70,
        "RESPIRATORY CYCLE SEGMENTATION REPORT (chest location in cycle names)",
        "=" * 70,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Execution time: {elapsed:.1f} seconds",
        "",
        "--- INPUT FILES ---",
        f"Preprocessed files available: {len(processed_files)}",
        f"Files segmented successfully: {stats['files_processed']}",
        f"Files without annotations: {stats['files_skipped_no_annotations']}",
        f"Files with errors: {stats['files_skipped_errors']}",
        "",
        "--- RESPIRATORY CYCLES ---",
        f"Total cycles extracted: {stats['total_cycles']}",
        f"Unique cycle IDs: {n_unique_ids}",
        f"Unique cycle filenames: {n_unique_files}",
        f"Name collisions: {stats['total_cycles'] - n_unique_files}",
        f"Unique patients: {n_patients}",
        f"Cycles per patient (mean): {stats['total_cycles']/max(n_patients,1):.1f}",
        f"Padded cycles: {stats['cycles_padded']} ({100*stats['cycles_padded']/max(stats['total_cycles'],1):.1f}%)",
        f"Truncated cycles: {stats['cycles_truncated']} ({100*stats['cycles_truncated']/max(stats['total_cycles'],1):.1f}%)",
        f"Very short cycles (<{config['min_cycle_duration_sec']}s): {stats['cycles_too_short']}",
        "",
        "--- CLASS DISTRIBUTION ---",
    ]

    for cls_name, cls_count in sorted(stats["cycles_by_class"].items()):
        pct = 100 * cls_count / max(stats["total_cycles"], 1)
        report_lines.append(f"  {cls_name:10s}: {cls_count:5d} ({pct:5.1f}%)")

    report_lines.extend([
        "",
        "--- DURATION STATISTICS (seconds) ---",
        f"  Mean:     {durations.mean():.2f}",
        f"  Median:   {np.median(durations):.2f}",
        f"  Std:      {durations.std():.2f}",
        f"  Min:      {durations.min():.2f}",
        f"  Max:      {durations.max():.2f}",
        f"  P5:       {np.percentile(durations, 5):.2f}",
        f"  P25:      {np.percentile(durations, 25):.2f}",
        f"  P75:      {np.percentile(durations, 75):.2f}",
        f"  P95:      {np.percentile(durations, 95):.2f}",
        "",
        "--- DISTRIBUTION BY CHEST LOCATION ---",
    ])

    for loc, count in df_cycles["chest_location"].value_counts().items():
        loc_full = CHEST_LOCATION_MAP.get(loc, loc)
        pct = 100 * count / len(df_cycles)
        report_lines.append(f"  {loc} ({loc_full}): {count} ({pct:.1f}%)")

    report_lines.extend([
        "",
        "--- DISTRIBUTION BY EQUIPMENT ---",
    ])

    for equip, count in df_cycles["recording_equipment"].value_counts().items():
        equip_full = EQUIPMENT_MAP.get(equip, equip)
        pct = 100 * count / len(df_cycles)
        report_lines.append(f"  {equip} ({equip_full}): {count} ({pct:.1f}%)")

    report_lines.extend([
        "",
        "--- CONFIGURATION USED ---",
        f"  Target SR: {config['target_sr']} Hz",
        f"  Max duration: {config['max_cycle_duration_sec']}s",
        f"  Min duration: {config['min_cycle_duration_sec']}s",
        f"  Padding: {config['padding_strategy']}",
        f"  Temporal margin: {config['temporal_margin_sec']}s",
        f"  Output format: {config['output_format']} ({config['output_subtype']})",
        "",
        "=" * 70,
        "CONCLUSION: Segmentation completed successfully.",
        f"Dataset ready for Mel-spectrogram extraction.",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"\n  Report saved to: {REPORT_FILE}")

    logger.info("\n" + "=" * 70)
    logger.info("SEGMENTATION COMPLETED")
    logger.info(f"   {stats['total_cycles']} cycles from {stats['files_processed']} files")
    logger.info(f"   {n_patients} unique patients")
    logger.info(f"   Collisions: {stats['total_cycles'] - n_unique_files}")
    logger.info(f"   Files in: {SEGMENTS_DIR}")
    logger.info("=" * 70)

    return df_cycles


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Respiratory cycle segmentation pipeline")
    print("Cycle names include chest_location")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Preprocessed directory: {PROCESSED_DIR}")
    print(f"ICBHI directory: {ICBHI_DIR}")
    print(f"Output directory: {SEGMENTS_DIR}")
    print("=" * 70 + "\n")

    if not PROCESSED_DIR.exists():
        print(f"ERROR: directory not found: {PROCESSED_DIR}")
        print("Run 03_preprocessing_pipeline.py first")
        sys.exit(1)

    if not ICBHI_DIR.exists():
        print(f"ERROR: directory not found: {ICBHI_DIR}")
        print("Check the ICBHI dataset path")
        sys.exit(1)

    df_result = segment_all_recordings()

    print("\n--- Preview of the cycle DataFrame ---")
    print(df_result.head(10).to_string())
    print(f"\nColumns: {list(df_result.columns)}")
    print(f"Shape: {df_result.shape}")
