# -*- coding: utf-8 -*-
"""
Script 28: Supplementary list of the recordings excluded by quality control.

preprocessing_metadata.csv leaves original_clipping_pct empty for the 22 excluded
recordings, because the pipeline discarded them before measuring it. This script
recomputes the clipping percentage on the raw WAV with the same deterministic
detect_clipping() used in script 02 (channel mean, peak normalization, fraction of
samples with |x| > 0.99, times 100), and checks the reported figures: 22 recordings,
21 from the AKG C417L, 3 in the official test set, clipping range 10.0-48.6 %.

Inputs:  data/raw/ (WAV + ICBHI_challenge_train_test.txt),
         data/processed/preprocessing_metadata.csv
Outputs: results/quality_analysis/supplementary_excluded_recordings.csv
         (Recording | Clipping (%) | Device | Official split, sorted by clipping, descending)
Usage:
    python scripts/28_supplementary_excluded_recordings.py
"""

import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_DIR   = os.path.join(BASE_DIR, "data", "raw")
EXCLUDED_CSV = os.path.join(BASE_DIR, "data", "processed", "preprocessing_metadata.csv")
TRAINTEST_TXT = os.path.join(AUDIO_DIR, "ICBHI_challenge_train_test.txt")

OUT_DIR = os.path.join(BASE_DIR, "results", "quality_analysis")
OUT_CSV = os.path.join(OUT_DIR, "supplementary_excluded_recordings.csv")
LOG_TXT = os.path.join(OUT_DIR, "28_supplementary_log.txt")

DEVICES = ("AKGC417L", "Meditron", "LittC2SE", "Litt3200")


def setup_logging():
    """Log to stdout and to LOG_TXT."""
    os.makedirs(OUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.FileHandler(LOG_TXT, mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)
    log.info("Start | %s", datetime.now().isoformat(timespec="seconds"))
    return log


def load_audio_file(filepath):
    """Load a WAV and average channels if stereo (identical to script 02)."""
    audio, sr = sf.read(filepath)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr


def detect_clipping(audio, threshold=0.99):
    """Percentage of saturated samples (identical to script 02)."""
    audio_normalized = audio / (np.max(np.abs(audio)) + 1e-10)
    clipped_samples = np.sum(np.abs(audio_normalized) > threshold)
    return (clipped_samples / len(audio)) * 100


def load_official_split(path, log):
    """Return {name_without_extension: 'train'|'test'} from the official split file."""
    tt = {}
    with open(path, encoding="utf-8") as fh:
        for line_text in fh:
            parts_list = line_text.split()
            if len(parts_list) >= 2:
                tt[parts_list[0]] = parts_list[1]
    log.info("Official split loaded: %d recordings", len(tt))
    return tt


def main():
    log = setup_logging()

    for path_str in (AUDIO_DIR, EXCLUDED_CSV, TRAINTEST_TXT):
        if not os.path.exists(path_str):
            log.error("Not found: %s", path_str)
            log.error("Fix the input paths at the top of the script and run again.")
            sys.exit(1)

    meta = pd.read_csv(EXCLUDED_CSV)
    excluded_recs = meta.loc[meta["excluded"] == True, "filename"].tolist()
    log.info("Excluded recordings in the metadata: %d", len(excluded_recs))

    tt = load_official_split(TRAINTEST_TXT, log)

    # Recompute clipping on the raw WAV
    row_list = []
    for name_text in excluded_recs:
        wav = os.path.join(AUDIO_DIR, name_text)
        if not os.path.exists(wav):
            log.warning("  WAV not found, skipped: %s", name_text)
            continue
        audio, sr = load_audio_file(wav)
        pct = detect_clipping(audio)
        disp = next((d for d in DEVICES if d in name_text), "unknown")
        row_list.append({
            "Recording": name_text,
            "Clipping (%)": round(pct, 2),
            "Device": disp,
            "Official split": tt.get(name_text.replace(".wav", ""), "unknown"),
        })
        log.info("  %-42s %6.2f %%  %-9s %s",
                 name_text, pct, disp, row_list[-1]["Official split"])

    df = pd.DataFrame(row_list).sort_values("Clipping (%)", ascending=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    log.info("Saved: %s", OUT_CSV)

    # Check against the figures reported in the article
    n        = len(df)
    n_akg    = int((df["Device"] == "AKGC417L").sum())
    n_test   = int((df["Official split"] == "test").sum())
    lo, hi   = df["Clipping (%)"].min(), df["Clipping (%)"].max()

    log.info("")
    log.info("=" * 64)
    log.info("CHECK AGAINST THE REPORTED FIGURES")
    log.info("=" * 64)
    checks = [
        ("discarded recordings", n,      22,   n == 22),
        ("from AKG C417L mic", n_akg,  21,   n_akg == 21),
        ("in the official test",      n_test, 3,    n_test == 3),
    ]
    for label_text, observed_value, expected_value, ok in checks:
        log.info("  %-26s obtained=%-4s reported=%-4s %s",
                 label_text, observed_value, expected_value, "OK" if ok else "*** MISMATCH ***")

    range_ok = abs(lo - 10.0) < 0.15 and abs(hi - 48.6) < 0.15
    log.info("  %-26s obtained=%.1f-%.1f  reported=10.0-48.6  %s",
             "clipping range", lo, hi, "OK" if range_ok else "*** MISMATCH ***")

    # Retained recordings must never reach the 10 % threshold
    ret = meta.loc[meta["excluded"] != True, "original_clipping_pct"].dropna()
    if len(ret):
        log.info("  %-26s max=%.2f %% (threshold 10.0) %s",
                 "retained", ret.max(), "OK" if ret.max() < 10.0 else "*** MISMATCH ***")

    if all(c[3] for c in checks) and range_ok:
        log.info("")
        log.info("All figures match. The CSV can be released as")
        log.info("supplementary material.")
    else:
        log.warning("")
        log.warning("There are mismatches. Review them before releasing the CSV.")

    log.info("End | OK")


if __name__ == "__main__":
    main()
