#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
verify_qc_footnote.py: Check the audio quality-control figures reported in the article.

Verifies that a quality-control CSV (regenerated with Script 02) reproduces exactly the
figures stated in the manuscript's quality-control footnote: 22 recordings excluded for
clipping >= 10 %, clipping range 10.0 % to 48.6 % among them, 21 of the 22 from the
AKG C417L microphone, and 3 of the 22 in the official test set with 23 cycles in total.
If a figure differs, the footnote must be corrected, not the CSV.

Inputs:  results/quality_analysis/audio_quality_summary_FULL.csv,
         data/raw/ICBHI_challenge_train_test.txt (official train/test list),
         data/raw/<recording>.txt (cycle annotations); checks needing a missing file are skipped
Outputs: console report only
Usage:
    python verify_qc_footnote.py
    python verify_qc_footnote.py --csv results/quality_analysis/audio_quality_summary_FULL.csv
    python verify_qc_footnote.py --csv ... --threshold 10.0
"""

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR_FIXED = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = BASE_DIR_FIXED if BASE_DIR_FIXED.exists() else SCRIPT_DIR.parent


def pick(df, *names):
    """Return the first column of df whose lowercase name is in names, or None."""
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def main():
    ap = argparse.ArgumentParser(description="Verify the quality-control footnote figures")
    ap.add_argument("--csv", default=str(BASE_DIR / "results" / "quality_analysis" /
                                         "audio_quality_summary_FULL.csv"))
    ap.add_argument("--raw_dir", default=str(BASE_DIR / "data" / "raw"))
    ap.add_argument("--threshold", type=float, default=10.0)
    args = ap.parse_args()

    csv = Path(args.csv)
    if not csv.exists():
        raise SystemExit(f"ERROR: file not found: {csv}")
    df = pd.read_csv(csv)
    c_file = pick(df, "filename", "file", "wav")
    c_clip = pick(df, "clipping_pct", "clipping", "clip_pct")
    if c_file is None or c_clip is None:
        raise SystemExit(f"ERROR: the CSV has no file and clipping columns. It has: {list(df.columns)}")

    print("=" * 74)
    print(f"CSV: {csv}")
    print(f"Recordings analysed: {len(df)}")
    print("=" * 74)

    exc = df[df[c_clip] >= args.threshold].copy()
    exc["stem"] = exc[c_file].astype(str).str.replace(".wav", "", regex=False)
    print(f"\n[1] Excluded with clipping >= {args.threshold:.0f} %: {len(exc)}   (footnote states 22)")
    if len(exc):
        print(f"[2] Clipping range: {exc[c_clip].min():.1f} % to {exc[c_clip].max():.1f} %"
              f"   (footnote states 10.0 % to 48.6 %)")

    # The recording device is the last field of the ICBHI file name
    dev = exc["stem"].str.split("_").str[-1].value_counts()
    print("\n[3] By device   (footnote states 21 from AKG C417L)")
    for k, v in dev.items():
        print(f"      {k:<12} {v}")

    raw = Path(args.raw_dir)
    split_file = raw / "ICBHI_challenge_train_test.txt"
    if not split_file.exists():
        print(f"\n[4] {split_file} not found; skipping the official test-set check.")
        return
    split = {}
    for line in split_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            split[parts[0].strip()] = parts[1].strip().lower()
    in_test = [s for s in exc["stem"] if split.get(s) == "test"]
    print(f"\n[4] Excluded recordings in the official test set: {len(in_test)}   (footnote states 3)")
    for s in in_test:
        print(f"      {s}")

    # Count annotated cycles (non-empty lines) in those recordings
    total = 0
    missing_items = []
    for s in in_test:
        ann = raw / f"{s}.txt"
        if ann.exists():
            total += sum(1 for l in ann.read_text(errors="ignore").splitlines() if l.strip())
        else:
            missing_items.append(s)
    if missing_items:
        print(f"\n[5] No annotations for: {missing_items}; the cycle count is incomplete.")
    print(f"[5] Cycles in those recordings: {total}   (footnote states 23)")

    print("\n" + "=" * 74)
    print("If any figure differs, correct the manuscript footnote with the actual value.")


if __name__ == "__main__":
    main()
