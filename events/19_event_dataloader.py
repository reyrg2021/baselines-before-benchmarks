#!/usr/bin/env python3
r"""
Script 19: Dataloader for variable-length event sequences (CfC input).

Reads the events.pkl produced by Script 18 and the zero-leakage split of Script 07
(strategy2_custom), and returns PyTorch DataLoaders with padded variable-length event
sequences, a validity mask, and the per-step timespans required by the CfC.
`timespan_mode` controls which time intervals the model sees, isolating the contribution
of irregular timing on the same representation: "real" (segment durations in seconds),
"ones" (dt = 1 at every step) and "mean" (constant dt equal to the sequence mean, which
keeps the total duration but removes its variation).

Inputs:  data/events/events.pkl, data/splits/strategy2_custom/{train,val,test}_cycles.csv
Outputs: none (module); a shape sanity check when run directly.
Usage:
    from importlib import import_module
    dl = import_module("19_event_dataloader")
    loaders, F, scaler = dl.build_loaders(
        events_pkl="data/events/events.pkl",
        split_dir="data/splits/strategy2_custom",
        batch_size=32, timespan_mode="real")

    python 19_event_dataloader.py --events data/events/events.pkl \
        --split-dir data/splits/strategy2_custom --timespan-mode real
"""

import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class EventSequenceDataset(Dataset):
    """Per-cycle event sequences (features + timespans) for one split."""

    def __init__(self, events, filenames, scaler=None, timespan_mode="real"):
        assert timespan_mode in ("real", "ones", "mean")
        self.items = []
        mean, std = (scaler if scaler is not None else (None, None))
        for fn in filenames:
            e = events[fn]
            feats = e["feats"].astype(np.float32).copy()
            if scaler is not None:
                feats = (feats - mean) / std
            dt = e["dt"].astype(np.float32)
            if timespan_mode == "ones":
                ts = np.ones_like(dt)
            elif timespan_mode == "mean":
                ts = np.full_like(dt, float(dt.mean()) if len(dt) else 1.0)
            else:
                ts = dt
            self.items.append((feats, ts.astype(np.float32), int(e["label"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        feats, ts, label = self.items[i]
        return torch.from_numpy(feats), torch.from_numpy(ts), label


def collate_events(batch):
    """Zero-pad to the batch Tmax. Returns feats[B,T,F], ts[B,T,1], mask[B,T] (1 = valid), y[B]."""
    feats, ts, labels = zip(*batch)
    B = len(batch)
    Tmax = max(f.shape[0] for f in feats)
    F = feats[0].shape[1]
    X = torch.zeros(B, Tmax, F, dtype=torch.float32)
    TS = torch.ones(B, Tmax, 1, dtype=torch.float32)   # padded with 1 (neutral timespan)
    M = torch.zeros(B, Tmax, dtype=torch.float32)
    for i, (f, t) in enumerate(zip(feats, ts)):
        T = f.shape[0]
        X[i, :T] = f
        TS[i, :T, 0] = t
        M[i, :T] = 1.0
    y = torch.tensor(labels, dtype=torch.long)
    return X, TS, M, y


def fit_scaler(events, train_filenames):
    """Per-feature mean/std over all training segments only (no leakage)."""
    stacked = np.concatenate([events[fn]["feats"] for fn in train_filenames], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def load_split_filenames(split_csv, events):
    """Read a split CSV and return the cycle_filename entries present in events.pkl, plus the missing count."""
    df = pd.read_csv(split_csv)
    col = "cycle_filename" if "cycle_filename" in df.columns else df.columns[0]
    present = [fn for fn in df[col].tolist() if fn in events]
    missing = len(df) - len(present)
    return present, missing


def build_loaders(events_pkl, split_dir, batch_size=32, timespan_mode="real",
                  num_workers=0, shuffle_train=True):
    """Build train/val/test DataLoaders. Returns (loaders, F, scaler)."""
    with open(events_pkl, "rb") as fh:
        events = pickle.load(fh)
    split_dir = Path(split_dir)
    fnames = {}
    for split in ("train", "val", "test"):
        csv = split_dir / f"{split}_cycles.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Not found: {csv}")
        fnames[split], miss = load_split_filenames(csv, events)
        print(f"  {split:<5}: {len(fnames[split])} cycles ({miss} missing from events.pkl)")

    scaler = fit_scaler(events, fnames["train"])
    F = events[fnames["train"][0]]["feats"].shape[1]

    loaders = {}
    for split in ("train", "val", "test"):
        ds = EventSequenceDataset(events, fnames[split], scaler=scaler, timespan_mode=timespan_mode)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train" and shuffle_train),
            collate_fn=collate_events, num_workers=num_workers)
    return loaders, F, scaler


def main():
    ap = argparse.ArgumentParser(description="Sanity check of the event dataloader")
    ap.add_argument("--events", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--timespan-mode", default="real", choices=["real", "ones", "mean"])
    args = ap.parse_args()

    print(f"timespan_mode = {args.timespan_mode}")
    loaders, F, scaler = build_loaders(args.events, args.split_dir,
                                       batch_size=args.batch_size,
                                       timespan_mode=args.timespan_mode)
    print(f"F (features/segment) = {F}")
    X, TS, M, y = next(iter(loaders["train"]))
    print(f"batch -> feats {tuple(X.shape)} | timespans {tuple(TS.shape)} | "
          f"mask {tuple(M.shape)} | y {tuple(y.shape)}")
    print(f"  true lengths in the batch (mask sum): {M.sum(1).int().tolist()[:12]} ...")
    print(f"  valid timespans[0]: {TS[0,:int(M[0].sum()),0].numpy().round(3)}")
    print(f"  standardized feats: mean≈{float(X[M.bool()].mean()):.3f} std≈{float(X[M.bool()].std()):.3f}")


if __name__ == "__main__":
    main()
