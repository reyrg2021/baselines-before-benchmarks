#!/usr/bin/env python3
r"""
Script 20: CfC training on event sequences, multi-seed, with a time-gate ablation.

Trains a CfC on the irregular event sequences of Script 18 and reports the mean and
standard deviation of each metric over seeds. --compare contrasts three timespan modes:
real (true Δt), ones (Δt = 1) and mean (constant Δt), which isolates the continuous-time
mechanism from the representation. --no-duration removes the segment duration (the last
feature of the Script 18 vector) so that real-vs-ones measures the contribution of
continuous time alone, since otherwise every mode can read duration as a feature.
--study runs both conditions and answers: Q1 does irregular Δt help, Q2 does the event
representation beat the mel grid, Q3 does the Δt effect change without duration.

Inputs:  events pickle from Script 18, split directory (train/val/test CSVs),
         torch_cfc.py and 19_event_dataloader.py in this folder
Outputs: results printed to stdout
Usage:
    python 20_cfc_event_training.py --events data/events/events.pkl \
        --split-dir data/splits/strategy2_custom --compare --seeds 0 1 2 3 4
    python 20_cfc_event_training.py --events data/events/events.pkl \
        --split-dir data/splits/strategy2_custom --study --seeds 0 1 2 3 4
"""

import sys
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import torch_cfc  # noqa: E402


def _load_dataloader_module():
    path = SCRIPT_DIR / "19_event_dataloader.py"
    spec = importlib.util.spec_from_file_location("event_dataloader", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CLASSES = ["normal", "crackle", "wheeze", "both"]
# Mel-grid anchor: the official score of the mel CfC depends on the split.
#   zero-leakage (strategy2_custom): 0.365 (mean of 5 seeds).
#   strategy1_official: must be recomputed (not 0.365; that split has 156/218 leakage).
# The active anchor is passed with --anchor and is not hard-coded.
ANCHOR_ZEROLEAK = 0.365  # documented reference; unused unless passed via --anchor

HPARAMS = {
    "backbone_activation": "gelu",
    "backbone_units": 64,
    "backbone_layers": 2,
    "backbone_dr": 0.1,
    "init": 1.0,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def official_metric(y_true, y_pred):
    cm = np.zeros((4, 4), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    sp = cm[0, 0] / max(cm[0].sum(), 1)
    se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / max(cm[1:].sum(), 1)
    rec = np.array([cm[i, i] / max(cm[i].sum(), 1) for i in range(4)])
    acc = np.trace(cm) / max(cm.sum(), 1)
    return dict(sp=sp, se=se, score=(sp + se) / 2, acc=acc, rec=rec)


def binary_auc(y_true, p_abnormal):
    """Binary AUC Normal(0) vs abnormal(1,2,3), the anti-collapse safeguard.
    Averaged ranks (Mann-Whitney U), numpy only. Returns nan if a class is missing."""
    yb = (np.asarray(y_true) > 0).astype(int)
    s = np.asarray(p_abnormal, dtype=float)
    n_pos = int(yb.sum()); n_neg = len(yb) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks_sorted = np.empty(len(s), float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + j) / 2.0 + 1.0   # average rank (1-based)
        i = j + 1
    ranks = np.empty(len(s), float); ranks[order] = ranks_sorted
    return float((ranks[yb == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def last_valid_logits(seq, mask):
    lengths = mask.sum(1).long().clamp(min=1)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, seq.size(2))
    return seq.gather(1, idx).squeeze(1)


def run_epoch(model, loader, optimizer, criterion, device, train, drop_last_feat=False):
    model.train() if train else model.eval()
    ys, ps, prob_abn, tot = [], [], [], 0.0
    for X, TS, M, y in loader:
        if drop_last_feat:
            X = X[..., :-1]                      # drop duration (last feature)
        X, TS, M, y = X.to(device), TS.to(device), M.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            seq = model(X, timespans=TS)
            logits = last_valid_logits(seq, M)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot += loss.item() * len(y)
        ys += y.cpu().tolist(); ps += logits.argmax(1).cpu().tolist()
        prob_abn += (1.0 - torch.softmax(logits, dim=1)[:, 0]).detach().cpu().tolist()
    return tot / max(len(ys), 1), np.array(ys), np.array(ps), np.array(prob_abn)


def train_one(events_pkl, split_dir, timespan_mode, seed, epochs, batch_size, hidden,
              lr, patience, device, no_duration=False, verbose=False):
    """Train one CfC (in_features=F, out=4, timespans=per-step Δt in seconds) and return
    test metrics. Variable length is handled by taking the last valid step; `mask` is not
    passed to the CfC because that branch implements per-feature GRU-D masking."""
    set_seed(seed)                                # seed before loaders (shuffle) and model
    dl = _load_dataloader_module()
    loaders, F, _ = dl.build_loaders(events_pkl, split_dir, batch_size=batch_size,
                                     timespan_mode=timespan_mode)
    in_feat = F - 1 if no_duration else F
    model = torch_cfc.Cfc(in_features=in_feat, hidden_size=hidden, out_feature=4,
                          hparams=HPARAMS, return_sequences=True).to(device)

    ytr = np.array([y for *_, y in loaders["train"].dataset.items])
    counts = np.bincount(ytr, minlength=4).astype(float)
    w = torch.tensor(counts.sum() / (4 * np.clip(counts, 1, None)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, best_state, bad = -1, None, 0
    for ep in range(epochs):
        run_epoch(model, loaders["train"], optimizer, criterion, device, True, no_duration)
        _, yv, pv, _ = run_epoch(model, loaders["val"], optimizer, criterion, device, False, no_duration)
        vs = official_metric(yv, pv)["score"]
        if vs > best_val:
            best_val, best_state, bad = vs, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    _, yte, pte, prob_te = run_epoch(model, loaders["test"], optimizer, criterion, device, False, no_duration)
    m = official_metric(yte, pte)
    m["auc"] = binary_auc(yte, prob_te)           # anti-collapse safeguard
    if verbose:
        print(f"      seed {seed} [{timespan_mode:>4}{' no-dur' if no_duration else ''}]: "
              f"Score={m['score']:.3f} Se={m['se']:.3f} Sp={m['sp']:.3f} AUC={m['auc']:.3f}")
    return m


def agg(runs):
    """Mean and std over seeds for Score, Se, Sp, acc, AUC and per-class recall."""
    f = lambda k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
    rec = np.array([r["rec"] for r in runs])
    return dict(score=f("score"), se=f("se"), sp=f("sp"), acc=f("acc"), auc=f("auc"),
                rec_mean=rec.mean(0), rec_std=rec.std(0), n=len(runs))


def run_modes(events, split_dir, seeds, no_duration, args, device):
    """Run real/ones/mean over all seeds. Returns dict mode -> agg."""
    out = {}
    for mode in ["real", "ones", "mean"]:
        runs = [train_one(events, split_dir, mode, s, args.epochs, args.batch_size,
                          args.hidden, args.lr, args.patience, device,
                          no_duration=no_duration, verbose=True) for s in seeds]
        out[mode] = agg(runs)
    return out


def print_block(title, res, anchor=None):
    print(f"\n  {title}  (official Score, mean±std over {res['real']['n']} seeds)")
    print(f"  {'mode':>5} | {'Score':>13} | {'Se':>13} | {'Sp':>13} | {'AUC bin':>13}")
    print("  " + "-" * 70)
    for mode in ["real", "ones", "mean"]:
        a = res[mode]
        print(f"  {mode:>5} | {a['score'][0]:.3f} ± {a['score'][1]:.3f} | "
              f"{a['se'][0]:.3f} ± {a['se'][1]:.3f} | {a['sp'][0]:.3f} ± {a['sp'][1]:.3f} | "
              f"{a['auc'][0]:.3f} ± {a['auc'][1]:.3f}")
    d = res["real"]["score"][0] - max(res["ones"]["score"][0], res["mean"]["score"][0])
    noise = max(res["real"]["score"][1], res["ones"]["score"][1], res["mean"]["score"][1])
    q1 = "irregular Δt HELPS" if d > noise else "irregular Δt does NOT help (within seed-to-seed noise)"
    print(f"    Q1: real-max(ones,mean)={d:+.3f} (noise≈{noise:.3f}) -> {q1}")
    q2_best = max(res[m]["score"][0] for m in ["real", "ones", "mean"])
    if anchor is None:
        print(f"    Q2: no mel-grid anchor provided for this split (best event={q2_best:.3f}); "
              f"pass it with --anchor, recomputed on the SAME split. No cross-split comparison.")
    else:
        q2 = f"the representation BEATS the mel grid ({q2_best:.3f} vs {anchor})" if q2_best > anchor \
            else f"the representation does NOT beat the grid ({q2_best:.3f} vs {anchor})"
        print(f"    Q2: {q2}")


def main():
    ap = argparse.ArgumentParser(description="CfC on events: multi-seed + time-gate ablation")
    ap.add_argument("--events", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--timespan-mode", default="real", choices=["real", "ones", "mean"])
    ap.add_argument("--compare", action="store_true", help="run real/ones/mean")
    ap.add_argument("--study", action="store_true", help="with AND without duration, 3 modes, all seeds")
    ap.add_argument("--no-duration", action="store_true", help="drop duration from the features (ablation)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="list of seeds, e.g. --seeds 0 1 2 3 4")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--anchor", type=float, default=None,
                    help="Official score of the mel CfC ON THE SAME split (for Q2). "
                         "zero-leakage=0.365; must be recomputed for strategy1_official. "
                         "If omitted, Q2 is not evaluated.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f"20 - CfC on events | seeds={args.seeds} | device={device} | hidden={args.hidden}")
    print(f"   split: {args.split_dir}")
    if args.anchor is not None:
        print(f"   mel-grid anchor (SAME split): official Score {args.anchor}")
    else:
        print("   mel-grid anchor: NOT provided (--anchor) -> Q2 not evaluated")
    print("=" * 78)

    if args.study:
        print("\n>>> CONDITION A: WITH duration as a feature")
        resA = run_modes(args.events, args.split_dir, args.seeds, False, args, device)
        print("\n>>> CONDITION B: WITHOUT duration (time-gate ablation)")
        resB = run_modes(args.events, args.split_dir, args.seeds, True, args, device)
        print("\n" + "=" * 60)
        print("FULL STUDY")
        print("=" * 60)
        print_block("WITH duration", resA, anchor=args.anchor)
        print_block("WITHOUT duration (clean Δt ablation)", resB, anchor=args.anchor)
        print("\n  Q3 (does the Δt effect change without duration?):")
        dA = resA["real"]["score"][0] - max(resA["ones"]["score"][0], resA["mean"]["score"][0])
        dB = resB["real"]["score"][0] - max(resB["ones"]["score"][0], resB["mean"]["score"][0])
        print(f"    with duration: real-max(ones,mean)={dA:+.3f} | without duration: {dB:+.3f}")
        print("    -> if both ≈0, continuous time is genuinely useless in this representation.")
    elif args.compare:
        res = run_modes(args.events, args.split_dir, args.seeds, args.no_duration, args, device)
        cond = "WITHOUT duration (ablation)" if args.no_duration else "WITH duration"
        print_block(cond, res, anchor=args.anchor)
    else:
        runs = [train_one(args.events, args.split_dir, args.timespan_mode, s, args.epochs,
                          args.batch_size, args.hidden, args.lr, args.patience, device,
                          no_duration=args.no_duration, verbose=True) for s in args.seeds]
        a = agg(runs)
        print(f"\n  [{args.timespan_mode}{' no-dur' if args.no_duration else ''}] "
              f"Score {a['score'][0]:.3f} ± {a['score'][1]:.3f} | "
              f"Se {a['se'][0]:.3f} ± {a['se'][1]:.3f} | Sp {a['sp'][0]:.3f} ± {a['sp'][1]:.3f} | "
              f"AUC {a['auc'][0]:.3f} ± {a['auc'][1]:.3f}")


if __name__ == "__main__":
    main()
