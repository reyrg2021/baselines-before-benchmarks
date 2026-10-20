#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 06b: Remove duplicated rows from feature_metadata.csv.

Script 06 merged on non-unique keys, so feature_metadata.csv contained ~31,559 rows
instead of the 6,729 respiratory cycles; the Mel-spectrogram .npy files themselves are
correct. This script keeps the first row of each cycle_id, checks that every cycle
appears exactly once and that the row count matches the .npy files, and overwrites the
CSV (a backup of the original is kept). Run it before Script 07.

Inputs:  data/features/feature_metadata.csv, data/features/mel_spectrograms/*.npy
Outputs: data/features/feature_metadata.csv (cleaned),
         data/features/feature_metadata_BACKUP_with_duplicates.csv (original backup)
Usage:
    python 06b_fix_feature_metadata.py
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
METADATA_CSV = FEATURES_DIR / "feature_metadata.csv"
BACKUP_CSV = FEATURES_DIR / "feature_metadata_BACKUP_with_duplicates.csv"

print("=" * 70)
print("Cleaning feature_metadata.csv")
print("=" * 70)

df = pd.read_csv(METADATA_CSV)
print(f"\n[BEFORE] Total rows: {len(df)}")
print(f"[BEFORE] Unique cycle_id: {df['cycle_id'].nunique()}")
print(f"[BEFORE] Columns: {len(df.columns)}")

df.to_csv(BACKUP_CSV, index=False)
print(f"\n  Backup saved to: {BACKUP_CSV}")

# Keep the first occurrence of each cycle_id
df_clean = df.drop_duplicates(subset=["cycle_id"], keep="first").copy()
df_clean = df_clean.reset_index(drop=True)

print(f"\n[AFTER] Total rows: {len(df_clean)}")
print(f"[AFTER] Unique cycle_id: {df_clean['cycle_id'].nunique()}")

assert len(df_clean) == df_clean["cycle_id"].nunique(), \
    "ERROR: duplicates remain after cleaning"

class_dist = df_clean["cycle_class"].value_counts()
print(f"\n  Class distribution:")
for cls, count in class_dist.items():
    print(f"    {cls:10s}: {count:5d} ({100*count/len(df_clean):.1f}%)")

# The cleaned CSV must have one row per .npy file on disk
mel_dir = FEATURES_DIR / "mel_spectrograms"
n_npy = len(list(mel_dir.glob("*.npy")))
print(f"\n  .npy files on disk: {n_npy}")
print(f"  Rows in cleaned CSV: {len(df_clean)}")

if n_npy == len(df_clean):
    print(f"  Exact match")
else:
    print(f"  WARNING: difference of {abs(n_npy - len(df_clean))} files")

df_clean.to_csv(METADATA_CSV, index=False)
print(f"\n  Cleaned CSV saved to: {METADATA_CSV}")

print(f"\n--- Preview of the cleaned CSV ---")
print(df_clean[["cycle_id", "cycle_class", "mel_filename",
                "mel_shape_rows", "mel_shape_cols"]].head(10).to_string())

print(f"\n{'='*70}")
print(f"CLEANING COMPLETED")
print(f"   {len(df_clean)} unique cycles in feature_metadata.csv")
print(f"   Script 07 can now be run")
print(f"{'='*70}")
