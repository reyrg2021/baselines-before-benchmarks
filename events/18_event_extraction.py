#!/usr/bin/env python3
r"""
Script 18: Event extraction from spectro-temporal change points.

For each ICBHI cycle (output of Script 05) builds a variable-length event sequence
[(segment_features, Δt), ...], where segments are the spans between change points
detected on the high-passed spectrum. This is the irregular representation consumed
by the continuous-time CfC (Scripts 19 and 20).
Choices validated on the real audio:
  - Band-pass 120-2000 Hz is required: 60-71% of cycle energy lies in 0-100 Hz (heart
    sound, friction, drift) and drowns the spectral flux; with the filter the detector
    separates crackle presence (crackle/both fragment more than normal/wheeze).
  - Δt in seconds: the CfC feeds Δt into sigmoid(time_a*Δt+time_b); in samples (~10^4
    at 16 kHz) the sigmoid saturates and the continuous-time mechanism vanishes.
  - Δt[t] is the duration of segment t, so wheezes give few long steps and crackles
    many short ones.

Inputs:  data/segments/ (cycle WAVs), data/segments/processed_cycles.csv
Outputs: data/events/events.pkl  (dict cycle_filename -> {"feats": float32 [T, F],
             "dt": float32 [T] seconds, "label": int 0..3, "class_name": str, "patient_id": str})
         data/events/events_index.csv  (cycle_filename, class_name, label, patient_id,
             n_events, dur_total, dt_mean); a per-class summary is printed to stdout.
Usage:
    python 18_event_extraction.py
    python 18_event_extraction.py --segments-dir data/segments \
        --cycles-csv data/segments/processed_cycles.csv --out-dir data/events --delta-k 2.0
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import stft, butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d


# Configuration
BASE_DIR = Path(os.environ.get("BBB_BASE_DIR",
                str(Path(__file__).resolve().parent.parent)))

# Diagnostic band (removes heart/friction energy; nothing useful above 2 kHz).
HP_LO, HP_HI = 120.0, 2000.0
# STFT window and hop in ms, converted with the actual sample rate of each WAV.
WIN_MS, HOP_MS = 32.0, 8.0
SMOOTH_SIGMA = 2.0          # novelty smoothing
PEAK_MIN_DIST_MS = 20.0     # minimum spacing between change points
DEFAULT_DELTA_K = 2.0       # relative peak-picking threshold
N_LOG_BANDS = 16            # log bands for per-segment features
CLASSES = ["normal", "crackle", "wheeze", "both"]


def class_from_flags(c, w):
    c, w = int(c), int(w)
    return "normal" if (c == 0 and w == 0) else "crackle" if (c == 1 and w == 0) else \
           "wheeze" if (c == 0 and w == 1) else "both"


# Preprocessing and change-point detector
def preprocess(x, sr):
    """Mono + band-pass 120-2000 Hz (removes the low band that drowns the flux)."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    sos = butter(4, [HP_LO / (sr / 2), HP_HI / (sr / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64))


def _stft_params(sr):
    nperseg = max(16, int(round(WIN_MS / 1000.0 * sr)))
    hop = max(1, int(round(HOP_MS / 1000.0 * sr)))
    return nperseg, hop


def _peak_pick(v, sr, hop, delta, pre=3, post=3, pre_avg=10, post_avg=10):
    """Adaptive local peak picking (a peak must exceed the local mean + delta)."""
    wait = max(1, int(round(PEAK_MIN_DIST_MS / 1000.0 * sr / hop)))
    peaks, last, n = [], -wait - 1, len(v)
    for i in range(n):
        if v[i] != v[max(0, i - pre):min(n, i + post + 1)].max():
            continue
        if v[i] < v[max(0, i - pre_avg):min(n, i + post_avg + 1)].mean() + delta:
            continue
        if i <= last + wait:
            continue
        peaks.append(i)
        last = i
    return np.array(peaks, dtype=int)


def detect_segments(x, sr, delta_k):
    """Return the spans (a, b) in seconds between change points (plus endpoints)."""
    nperseg, hop = _stft_params(sr)
    if len(x) < nperseg:
        return [(0.0, len(x) / sr)], None, None
    f, t, Z = stft(x, fs=sr, nperseg=nperseg, noverlap=nperseg - hop, boundary=None)
    S = np.abs(Z)
    flux = gaussian_filter1d(np.sum(np.maximum(np.diff(S, axis=1), 0.0), axis=0), SMOOTH_SIGMA)
    mad = np.median(np.abs(flux - np.median(flux))) + 1e-12
    pk = _peak_pick(flux, sr, hop, delta_k * 1.4826 * mad)
    cps = t[1:][pk]
    dur = len(x) / sr
    bounds = np.concatenate(([0.0], cps, [dur]))
    segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] > 1e-3]
    return (segs if segs else [(0.0, dur)]), (f, t, S), (nperseg, hop)


# Per-segment features
def _log_band_edges():
    return np.geomspace(HP_LO, HP_HI, N_LOG_BANDS + 1)


def segment_features(x, sr, segs, stft_cache):
    """Return (feats [T, F], dt [T] in seconds) for the segments.

    F = N_LOG_BANDS log energies + centroid + bandwidth + flatness + ZCR + duration.
    dt[t] is the duration of segment t, used by the CfC as its timespan.
    """
    f, t, S = stft_cache
    S = S + 1e-10
    band = (f >= HP_LO) & (f <= HP_HI)
    fb = f[band][:, None]
    edges = _log_band_edges()
    feats, dts = [], []
    for (a, b) in segs:
        fr = (t >= a) & (t < b)
        if fr.sum() < 1:                       # segment shorter than one frame
            fr = np.zeros(len(t), bool); fr[np.abs(t - (a + b) / 2).argmin()] = True
        Sf = S[:, fr]
        Sbf = S[band][:, fr]
        # Log band energies (mean over the segment frames)
        bands = [np.log(Sf[(f >= lo) & (f < hi), :].sum(axis=0).mean() + 1e-8)
                 for lo, hi in zip(edges[:-1], edges[1:])]
        # Spectral descriptors (mean over the segment)
        cen = float(((fb * Sbf).sum(0) / Sbf.sum(0)).mean())
        bw = float(np.sqrt(((fb - ((fb * Sbf).sum(0) / Sbf.sum(0))) ** 2 * Sbf).sum(0) / Sbf.sum(0)).mean())
        flat = float((np.exp(np.mean(np.log(Sbf), 0)) / np.mean(Sbf, 0)).mean())
        # Time-domain ZCR of the segment
        i0, i1 = int(round(a * sr)), int(round(b * sr))
        xs = x[i0:i1]
        zcr = float(np.mean(np.abs(np.diff(np.sign(xs))) > 0)) if len(xs) > 1 else 0.0
        dur = b - a
        feats.append(bands + [cen, bw, flat, zcr, dur])
        dts.append(dur)
    return np.asarray(feats, dtype=np.float32), np.asarray(dts, dtype=np.float32)


# Main
def main():
    ap = argparse.ArgumentParser(description="Event extraction from change points")
    ap.add_argument("--segments-dir", default=str(BASE_DIR / "data" / "segments"))
    ap.add_argument("--cycles-csv", default=str(BASE_DIR / "data" / "segments" / "processed_cycles.csv"))
    ap.add_argument("--out-dir", default=str(Path.cwd() / "data" / "events"))
    ap.add_argument("--delta-k", type=float, default=DEFAULT_DELTA_K)
    args = ap.parse_args()

    seg_dir = Path(args.segments_dir)
    csv_path = Path(args.cycles_csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}"); sys.exit(1)
    df = pd.read_csv(csv_path)
    if "patient_id" not in df.columns:
        df["patient_id"] = df["cycle_filename"].str.split("_").str[0]
    if "class_name" not in df.columns:
        df["class_name"] = [class_from_flags(c, w) for c, w in zip(df["has_crackles"], df["has_wheezes"])]

    print("=" * 78)
    print(f"18 - Event extraction | band-pass {HP_LO:.0f}-{HP_HI:.0f} Hz | delta_k={args.delta_k}")
    print(f"   cycles: {len(df)} | segments: {seg_dir}")
    print("=" * 78)

    events, rows = {}, []
    t0 = time.time(); n_ok = 0; n_skip = 0; F = None
    for k, r in enumerate(df.itertuples(index=False)):
        fn = r.cycle_filename
        wav = seg_dir / fn
        if not wav.exists():
            n_skip += 1; continue
        try:
            x, sr = sf.read(str(wav))
            dur = getattr(r, "duration", np.nan)
            if np.isfinite(dur) and dur > 0:
                n = int(round(dur * sr)); x = x[:n] if 0 < n < len(x) else x
            xp = preprocess(x, sr)
            segs, cache, _ = detect_segments(xp, sr, args.delta_k)
            if cache is None:
                n_skip += 1; continue
            feats, dt = segment_features(xp, sr, segs, cache)
        except Exception as e:
            print(f"  [skip] {fn}: {e}"); n_skip += 1; continue
        F = feats.shape[1]
        events[fn] = {"feats": feats, "dt": dt, "label": CLASSES.index(r.class_name),
                      "class_name": r.class_name, "patient_id": str(r.patient_id)}
        rows.append({"cycle_filename": fn, "class_name": r.class_name, "label": CLASSES.index(r.class_name),
                     "patient_id": str(r.patient_id), "n_events": len(dt),
                     "dur_total": float(dt.sum()), "dt_mean": float(dt.mean())})
        n_ok += 1
        if (k + 1) % 500 == 0:
            print(f"   ... {k + 1}/{len(df)} ({time.time()-t0:.0f}s)")

    with open(out_dir / "events.pkl", "wb") as fh:
        pickle.dump(events, fh)
    idx = pd.DataFrame(rows)
    idx.to_csv(out_dir / "events_index.csv", index=False)

    print(f"\nOK: {n_ok} cycles processed, {n_skip} skipped. F={F} features/segment.")
    print(f"Saved: {out_dir/'events.pkl'}  +  events_index.csv  ({time.time()-t0:.0f}s)")

    # Per-class summary on the full data
    print("\nPer-class summary (#events per cycle and Δt in seconds):")
    print(f"  {'class':<9} | {'#cycles':>7} | {'#ev (med)':>9} | {'dt mean':>9} | {'dt median':>10}")
    print("  " + "-" * 60)
    for c in CLASSES:
        sub = idx[idx.class_name == c]
        if len(sub) == 0:
            print(f"  {c:<9} | {0:>7} | {'-':>9} | {'-':>9} | {'-':>10}"); continue
        dt_all = np.concatenate([events[fn]["dt"] for fn in sub.cycle_filename])
        print(f"  {c:<9} | {len(sub):>7} | {sub.n_events.median():>9.1f} | "
              f"{dt_all.mean():>9.3f} | {np.median(dt_all):>10.3f}")


if __name__ == "__main__":
    main()
