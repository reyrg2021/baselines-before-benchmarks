#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 31: Recurrent baselines (LSTM / GRU / pure CfC) for the CfC-Hybrid negative result.

The CfC-Hybrid reaches binary AUC 0.501 on test but 0.687 on validation (CNN 0.690), so
the cell learns and the failure is generalization to the test partition. Conventional
recurrent baselines with the same parameter budget and recipe test whether the failure is
specific to continuous-time modelling. The CfC-Hybrid uses use_mixed=True (mmRNN): an
internal LSTMCell updates the state before the CfC cell at every step, so 278,784 of its
491,492 parameters (57 %) are already an LSTM (Conv2D frontend 23,520; CfC cell 188,928;
output 260). Three arms:
  lstm     nn.LSTM (one layer) + Linear, hidden size matched to 491,492 parameters (±1 %)
  gru      nn.GRU (one layer) + Linear, same matching
  cfcpure  CfCMelHybridV6 with use_mixed=False (CfC cell without the internal LSTM, ~213K)
Everything else is identical by construction: CfCMelHybridV6 from Script 11a is built
(3-block Conv2D frontend, 31 steps x 1024) and only its .cfc attribute is replaced. Training
uses Script 11a train_model() unchanged (CrossEntropy, AdamW base_lr 2e-4, decay 0.97/epoch,
weight_decay 1e-4, clipnorm 1.0, batch 64, class-weighted sampling, early stopping with
patience 20, checkpoint on val_loss), seeds [42, 123, 7, 99, 314], split strategy1_official.
LSTM/GRU heads apply Dropout(backbone_dr = 0.4) to the last hidden state, matching the
Dropout inside the CfC backbone without adding new hyperparameters.
Test and validation arrays are saved with the convention of Scripts 18/21/22 so that
Scripts 29 and 30 can include these runs.

Inputs:  data/splits/strategy1_official/, scripts 08 and 11a, results/master_results.json
Outputs: models/checkpoints/{cell}_official_seed{S}.pth,
         results/metrics/full_reeval/arrays/{cell}_official_seed{S}_{probs.npy,labels_preds.npz,
         val_probs.npy,val_labels_preds.npz},
         results/metrics/recurrent_baselines/recurrent_baselines_{runs.csv,summary.json,summary.md},
         script31_log.txt and one log per run
Usage:
    python 31_recurrent_baselines.py --cell lstm --seed 42     # one run
    python 31_recurrent_baselines.py --all                     # 3 cells x 5 seeds
    python 31_recurrent_baselines.py --evaluate                # re-evaluate checkpoints only
    python 31_recurrent_baselines.py --count_params            # count parameters only
    [--match {params,hidden}] [--epochs N] [--base_lr LR] [--tag SUFFIX] [--skip_existing]
"""

import sys
import copy
import json
import argparse
import logging
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR_FIXED = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
if BASE_DIR_FIXED.exists():
    BASE_DIR = BASE_DIR_FIXED
elif (SCRIPTS_DIR.parent / "models").exists():
    BASE_DIR = SCRIPTS_DIR.parent
else:
    BASE_DIR = SCRIPTS_DIR

CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
METRICS_DIR = BASE_DIR / "results" / "metrics"
ARRAYS_DIR = METRICS_DIR / "full_reeval" / "arrays"
OUT_DIR = METRICS_DIR / "recurrent_baselines"
SPLIT_DIR = BASE_DIR / "data" / "splits" / "strategy1_official"
MASTER_JSON = BASE_DIR / "results" / "master_results.json"

SEEDS = [42, 123, 7, 99, 314]
CELLS = ["lstm", "gru", "cfcpure"]
TARGET_PARAMS = 491_492          # CfC-Hybrid (master_results.json)
PARAM_TOL = 0.01
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_MAP = {c: i for i, c in enumerate(CLASS_NAMES)}

logger = logging.getLogger(__name__)


def setup_logging():
    log_file = BASE_DIR / "script31_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, mode="a", encoding="utf-8")],
        force=True,
    )
    return log_file


def import_script(name, filename):
    path = SCRIPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Metrics: same definitions as Scripts 18, 29 and 30

def icbhi_official(y_true, y_pred):
    cm = np.zeros((4, 4))
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    sp = cm[0, 0] / cm[0].sum() if cm[0].sum() else 0.0
    abn = cm[1:].sum()
    se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / abn if abn else 0.0
    return (se + sp) / 2, float(se), float(sp)


def auc_from_scores(y_bin, scores):
    pos = y_bin == 1; n_pos = int(pos.sum()); n_neg = len(y_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = stats.rankdata(scores)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics_from(y_true, probs32):
    y_pred = np.argmax(probs32, axis=1)
    p_abn = (np.float32(1.0) - probs32[:, 0]).astype(np.float64)
    s, se, sp = icbhi_official(y_true, y_pred)
    yt = y_true != 0; yp = y_pred != 0
    return {"icbhi": s, "se_off": se, "sp_off": sp,
            "bin_se": float((yt & yp).sum() / yt.sum()), "bin_sp": float((~yt & ~yp).sum() / (~yt).sum()),
            "auc": auc_from_scores(yt.astype(int), p_abn),
            "per_class_se": [float(((y_true == c) & (y_pred == c)).sum() / max((y_true == c).sum(), 1))
                             for c in range(4)]}


def mean_sd(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0) if len(x) else (np.nan, np.nan)


def make_rnn_head_class():
    import torch
    import torch.nn as nn

    class RNNHead(nn.Module):
        """
        Drop-in replacement for torch_cfc.Cfc: same forward(x, timespans=None) with x of
        shape (B, T, F); returns logits (B, n_classes) from the last step. Dropout(p) on the
        last hidden state mirrors the internal backbone_dr of the CfC cell.
        """
        def __init__(self, cell, in_features, hidden_size, out_features, dropout):
            super().__init__()
            rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
            self.cell = cell
            self.rnn = rnn_cls(in_features, hidden_size, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, out_features)

        def forward(self, x, timespans=None):
            out, _ = self.rnn(x)
            return self.fc(self.drop(out[:, -1]))

    return RNNHead


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def build_model(cell, hidden_size, seed_config, s11a):
    """
    Build CfCMelHybridV6 with the Script 11a config (identical frontend); for lstm/gru
    replace .cfc with RNNHead, for cfcpure only turn off use_mixed.
    Returns (model, config used).
    """
    cfg = copy.deepcopy(seed_config)
    cfg["cell"] = cell
    if cell == "cfcpure":
        cfg["use_mixed"] = False
        cfg["model_name"] = "CfC_HybridDeep_pure_noMixed"
        model = s11a.CfCMelHybridV6(n_classes=cfg["n_classes"], config=cfg)
        cfg["rnn_hidden_size"] = cfg["hidden_size"]
        return model, cfg
    model = s11a.CfCMelHybridV6(n_classes=cfg["n_classes"], config=cfg)   # the CfC built here is discarded
    RNNHead = make_rnn_head_class()
    model.cfc = RNNHead(cell, cfg["cfc_in_features"], hidden_size, cfg["n_classes"],
                        dropout=cfg.get("backbone_dr", 0.0))
    cfg["rnn_hidden_size"] = hidden_size
    cfg["model_name"] = f"{cell.upper()}_HybridDeep_h{hidden_size}"
    model.model_name = cfg["model_name"]
    return model, cfg


def find_hidden_for_params(cell, seed_config, s11a, target=TARGET_PARAMS, tol=PARAM_TOL):
    """
    Find the hidden_size whose total parameter count matches target (±tol).
    Analytic count (nn.LSTM: 4·(F·h + h² + 2h); nn.GRU: 3·(F·h + h² + 2h); output h·C + C)
    on top of the frontend measured once, then verified by building the model.
    """
    probe, cfg = build_model(cell, 8, seed_config, s11a)
    F = cfg["cfc_in_features"]; C = cfg["n_classes"]
    frontend = count_params(probe) - count_params(probe.cfc)
    gates = {"lstm": 4, "gru": 3}[cell]
    best = None
    for h in range(8, 2049):
        n = frontend + gates * (F * h + h * h + 2 * h) + (h * C + C)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (h, n)
        if n > target:
            break
    h, n_est = best
    m, _ = build_model(cell, h, seed_config, s11a)
    n = count_params(m)
    if n != n_est:
        logger.warning(f"  {cell}: analytic count {n_est:,} != actual {n:,}; using the actual count")
    if abs(n - target) / target > tol:
        logger.warning(f"  {cell}: ±{tol:.0%} not reached (h={h}, {n:,} vs {target:,})")
    return h, n


# Training (reuses train_model from Script 11a)

RUN_TAG = ""


def run_name_for(cell, seed):
    return f"{cell}{RUN_TAG}_official_seed{seed}"


def train_one(cell, seed, args, s11a, script08):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_ckpt = CHECKPOINTS_DIR / f"{run_name_for(cell, seed)}.pth"
    if args.skip_existing and final_ckpt.exists():
        logger.info(f"  {final_ckpt.name} exists; skipping (--skip_existing)")
        return final_ckpt

    config = copy.deepcopy(s11a.CONFIG_HYBRID)
    config["split_strategy"] = "strategy1_official"
    config["random_seed"] = seed
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.base_lr is not None:
        config["base_lr"] = args.base_lr          # sensitivity analysis only, not the main recipe
    # actual geometry (cfc_in_features, seq_length) is set by the model's __init__

    s11a.set_seed(seed)
    if cell == "cfcpure":
        hidden = config["hidden_size"]
    elif args.match == "hidden":
        hidden = config["hidden_size"]
    else:
        hidden, n = find_hidden_for_params(cell, config, s11a)
        logger.info(f"  {cell}: hidden_size={hidden} → {n:,} parameters (target {TARGET_PARAMS:,})")
    s11a.set_seed(seed)                                  # reseed after the search
    model, cfg = build_model(cell, hidden, config, s11a)
    model = model.to(device)
    n_params = count_params(model)
    logger.info(f"  Model {cfg['model_name']}: {n_params:,} parameters; device {device}")

    loaders = script08.create_dataloaders(
        strategy=cfg["split_strategy"], class_mode=cfg["class_mode"], augment_preset="none",
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        use_weighted_sampling=cfg["use_weighted_sampling"])

    run_name = run_name_for(cell, seed)
    # Per-run log in addition to script31_log.txt
    run_handler = logging.FileHandler(BASE_DIR / f"{run_name}_log.txt", mode="w", encoding="utf-8")
    run_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(run_handler)
    try:
        model, history, best_epoch = s11a.train_model(
            model, loaders["train_loader"], loaders["val_loader"], cfg, device, run_name)
    finally:
        logging.getLogger().removeHandler(run_handler); run_handler.close()

    # final checkpoint under the project name (Script 11a saves best_{run_name}.pth)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                "model_name": cfg["model_name"], "cell": cell, "n_params": n_params,
                "best_epoch": int(best_epoch), "history": history,
                "hidden_size_rnn": hidden, "seed": seed}, final_ckpt)
    logger.info(f"  Final checkpoint: {final_ckpt}")
    return final_ckpt


# Evaluation (test + val; arrays compatible with Scripts 18/21/22 and 29/30)

def load_final(ckpt_path, s11a, device):
    import torch
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ck["config"]
    model, _ = build_model(ck["cell"], ck.get("hidden_size_rnn", cfg.get("hidden_size")), cfg, s11a)
    model.load_state_dict(ck["model_state_dict"]); model.to(device).eval()
    return model, ck


def infer(model, loader, device):
    import torch
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            ps.append(torch.softmax(logits.float(), 1).cpu().numpy().astype(np.float32))
            ys.append(np.asarray(y))
    return np.concatenate(ys).astype(int), np.concatenate(ps)


def evaluate_all(args, s11a, script08):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = script08.create_dataloaders(
        strategy="strategy1_official", class_mode="4class", augment_preset="none",
        batch_size=64, num_workers=0, use_weighted_sampling=False)
    y_test_csv = pd.read_csv(SPLIT_DIR / "test_cycles.csv")["cycle_class"].map(CLASS_MAP).values
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell in (CELLS if args.cell is None else [args.cell]):
        for seed in (SEEDS if args.seed is None else [args.seed]):
            ckpt = CHECKPOINTS_DIR / f"{run_name_for(cell, seed)}.pth"
            if not ckpt.exists():
                logger.warning(f"  missing {ckpt.name}; skipping"); continue
            model, ck = load_final(ckpt, s11a, device)
            yv, pv = infer(model, loaders["val_loader"], device)
            yt, pt = infer(model, loaders["test_loader"], device)
            if not np.array_equal(yt, y_test_csv):
                raise ValueError(f"{ckpt.name}: test labels not aligned with test_cycles.csv")
            stem = run_name_for(cell, seed)
            np.save(ARRAYS_DIR / f"{stem}_probs.npy", pt)
            np.savez(ARRAYS_DIR / f"{stem}_labels_preds.npz", y_true=yt, y_pred=np.argmax(pt, 1))
            np.save(ARRAYS_DIR / f"{stem}_val_probs.npy", pv)
            np.savez(ARRAYS_DIR / f"{stem}_val_labels_preds.npz", y_true=yv, y_pred=np.argmax(pv, 1))
            mt, mv = metrics_from(yt, pt), metrics_from(yv, pv)
            row = {"cell": cell, "seed": seed, "n_params": ck["n_params"],
                   "hidden_size_rnn": ck["hidden_size_rnn"], "best_epoch": ck["best_epoch"] + 1,
                   "epochs_run": len(ck["history"]),
                   "val_icbhi": mv["icbhi"], "val_auc": mv["auc"],
                   "test_icbhi": mt["icbhi"], "test_se_off": mt["se_off"], "test_sp_off": mt["sp_off"],
                   "test_bin_se": mt["bin_se"], "test_bin_sp": mt["bin_sp"], "test_auc": mt["auc"]}
            for c, v in zip(CLASS_NAMES, mt["per_class_se"]):
                row[f"test_se_{c}"] = v
            rows.append(row)
            logger.info(f"  {stem:<24} params {row['n_params']:>8,}  ep {row['best_epoch']:>2}/{row['epochs_run']:<2} "
                        f"VAL ICBHI {mv['icbhi']:.3f} AUC {mv['auc']:.3f} | "
                        f"TEST ICBHI {mt['icbhi']:.3f} (Se {mt['se_off']:.3f}/Sp {mt['sp_off']:.3f}) AUC {mt['auc']:.3f}")
    if not rows:
        logger.warning("  No checkpoints to evaluate."); return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "recurrent_baselines_runs.csv", index=False)
    summarize(df)


def summarize(df):
    """Aggregate runs per cell and compare them, paired by seed, with CfC-Hybrid and CNN."""
    # reference values from master_results.json (same seeds)
    ref = {}
    if MASTER_JSON.exists():
        acc = json.load(open(MASTER_JSON, encoding="utf-8"))["accuracy"]
        for k, mk in [("CfC-Hybrid (mmRNN)", "CfC_Hybrid"), ("CNN-RespiratoryNet", "CNN_FP32")]:
            if mk in acc:
                ref[k] = {"seeds": acc[mk]["seeds"], "icbhi": acc[mk]["icbhi_official"]["values"],
                          "auc": acc[mk]["auc_binary"]["values"], "n_params": acc[mk]["n_params"]}
    agg, comp = [], []
    for cell, g in df.groupby("cell"):
        row = {"cell": cell, "n_seeds": len(g), "n_params": int(g["n_params"].iloc[0]),
               "hidden_size_rnn": int(g["hidden_size_rnn"].iloc[0])}
        for m in ["val_icbhi", "val_auc", "test_icbhi", "test_se_off", "test_sp_off", "test_auc",
                  "test_se_Wheeze", "test_se_Both", "best_epoch"]:
            row[f"{m}_mean"], row[f"{m}_sd"] = mean_sd(g[m])
        agg.append(row)
        for rname, r in ref.items():
            gi = g.set_index("seed")
            common = [s for s in r["seeds"] if s in gi.index]
            if len(common) < 3:
                continue
            for m in ["icbhi", "auc"]:
                a = np.array([gi.loc[s, f"test_{m}"] for s in common])
                b = np.array([r[m][r["seeds"].index(s)] for s in common])
                d = a - b; n = len(d); se = d.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.95, n - 1)
                comp.append({"cell": cell, "vs": rname, "metric": m, "n": n,
                             "diff_mean": float(d.mean()), "ci90_lo": float(d.mean() - t * se),
                             "ci90_hi": float(d.mean() + t * se),
                             "p_ttest": float(stats.ttest_rel(a, b).pvalue),
                             "p_wilcoxon": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else np.nan})
    out = {"timestamp": datetime.now().isoformat(timespec="seconds"), "target_params": TARGET_PARAMS,
           "aggregated": agg, "paired_vs_reference": comp, "reference": ref}
    with open(OUT_DIR / "recurrent_baselines_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    L = ["# Recurrent baselines (Script 31)\n", f"_{out['timestamp']} · target {TARGET_PARAMS:,} parameters_\n",
         "\n| Cell | Params | h | Val ICBHI | Val AUC | Test ICBHI | Off. Se | Off. Sp | Test AUC | Se Wheeze | Se Both | Best ep. |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def f(r, m, nd=3): return f"{r[f'{m}_mean']:.{nd}f} ± {r[f'{m}_sd']:.{nd}f}"
    for r in agg:
        L.append(f"| {r['cell']} | {r['n_params']:,} | {r['hidden_size_rnn']} | {f(r,'val_icbhi')} | {f(r,'val_auc')} "
                 f"| {f(r,'test_icbhi')} | {f(r,'test_se_off')} | {f(r,'test_sp_off')} | {f(r,'test_auc')} "
                 f"| {f(r,'test_se_Wheeze')} | {f(r,'test_se_Both')} | {f(r,'best_epoch',1)} |")
    for k, r in ref.items():
        L.append(f"| _{k}_ | {r['n_params']:,} | | | | {mean_sd(r['icbhi'])[0]:.3f} ± {mean_sd(r['icbhi'])[1]:.3f} | | "
                 f"| {mean_sd(r['auc'])[0]:.3f} ± {mean_sd(r['auc'])[1]:.3f} | | | |")
    if comp:
        L += ["\n**Paired differences by seed (cell − reference):**\n",
              "| Cell | vs | Metric | Δ | 90 % CI | t p | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
        for c in comp:
            L.append(f"| {c['cell']} | {c['vs']} | {c['metric']} | {c['diff_mean']:+.4f} "
                     f"| [{c['ci90_lo']:+.4f}, {c['ci90_hi']:+.4f}] | {c['p_ttest']:.3f} | {c['p_wilcoxon']:.3f} |")
    with open(OUT_DIR / "recurrent_baselines_summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    logger.info(f"  Summary: {OUT_DIR / 'recurrent_baselines_summary.md'}")


def parse_args():
    p = argparse.ArgumentParser(description="Script 31: recurrent baselines LSTM/GRU/pure CfC")
    p.add_argument("--cell", choices=CELLS, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all", action="store_true", help="3 cells x 5 seeds")
    p.add_argument("--evaluate", action="store_true", help="only evaluate existing checkpoints")
    p.add_argument("--count_params", action="store_true", help="only count parameters")
    p.add_argument("--match", choices=["params", "hidden"], default="params")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--base_lr", type=float, default=None,
                   help="Sensitivity: overrides the Script 11a base_lr (main recipe: 2e-4)")
    p.add_argument("--tag", default="", help="run_name suffix to avoid overwriting checkpoints (e.g. _lr5e4)")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    global RUN_TAG
    RUN_TAG = args.tag
    log_file = setup_logging()
    logger.info("=" * 78)
    logger.info("SCRIPT 31: recurrent baselines" + (f" (tag {RUN_TAG})" if RUN_TAG else ""))
    logger.info("=" * 78)
    logger.info(f"BASE_DIR: {BASE_DIR}")
    s11a = import_script("script11a", "11a_cfc_mel_v7.py")
    # Script 11a defines its global `logger` inside setup_logging(run_name); imported
    # without that call, train_model() would raise NameError. Inject a logger that
    # propagates to this script's handlers.
    s11a.logger = logging.getLogger("script11a")
    script08 = import_script("script08", "08_dataset_dataloader.py")

    if args.count_params:
        cfg = copy.deepcopy(s11a.CONFIG_HYBRID); cfg["split_strategy"] = "strategy1_official"
        orig = s11a.CfCMelHybridV6(n_classes=4, config=copy.deepcopy(cfg))
        pure, _ = build_model("cfcpure", cfg["hidden_size"], cfg, s11a)
        logger.info(f"  Original CfC-Hybrid (use_mixed=True): {count_params(orig):,}  "
                    f"(target {TARGET_PARAMS:,})")
        logger.info(f"  Pure CfC (use_mixed=False):           {count_params(pure):,}")
        for cell in ("lstm", "gru"):
            h, n = find_hidden_for_params(cell, cfg, s11a)
            m64, _ = build_model(cell, cfg["hidden_size"], cfg, s11a)
            logger.info(f"  {cell}: h={h} → {n:,} (match params);  h=64 → {count_params(m64):,} (match hidden)")
        return

    if args.evaluate:
        evaluate_all(args, s11a, script08); return

    jobs = [(c, s) for c in CELLS for s in SEEDS] if args.all else [(args.cell, args.seed)]
    if any(c is None or s is None for c, s in jobs):
        raise SystemExit("Specify --cell and --seed, or --all")
    for i, (cell, seed) in enumerate(jobs, 1):
        logger.info(f"\n[{i}/{len(jobs)}] {cell} seed {seed}")
        train_one(cell, seed, args, s11a, script08)
    evaluate_all(argparse.Namespace(cell=None if args.all else args.cell,
                                    seed=None if args.all else args.seed), s11a, script08)
    logger.info(f"Log: {log_file}"); logger.info("Done.")


if __name__ == "__main__":
    main()
