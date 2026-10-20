#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 19: FLOPs/MACs per inference and computational-efficiency table.

Complements parameter count, latency and memory (Script 15) with MACs per
inference for the CNN and the two CfC variants. Automatic counters (ptflops,
thop, fvcore) do not instrument recurrent layers (fvcore reports `aten::gru` as
unsupported and assigns it 0 FLOPs), so a raw count would make the CfC look
cheaper than the CNN even though its recurrent loop dominates its latency on the
Raspberry Pi 5. The count is therefore split in two transparent parts:
  1. Conv2D + Linear + BatchNorm, counted with ptflops (forward hooks as fallback).
  2. The recurrent layer, added analytically as
     T * K * (in_features*hidden + hidden*hidden),
     with T timesteps and K linear maps per cell (--cfc_linear_maps; default 4,
     a GRU-like approximation; the CfC cell has a cost of the same order).
FLOPs do not depend on the seed, so one checkpoint per model is used only to
rebuild the architecture.

Inputs:  models/checkpoints/{cnn,hybrid,transposed}_seed*.pth,
         scripts/09_cnn_training.py, scripts/11a_cfc_mel_v7.py
Outputs: results/metrics/flops/flops_results.json, flops_table.csv,
         flops_summary.md; flops_analysis_log.txt
Usage:
    python scripts/19_flops_analysis.py [--cfc_linear_maps 4] [--checkpoints_dir models/checkpoints]
        [--models cnn hybrid transposed]
"""

import os
import sys
import json
import argparse
import importlib.util
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent

if (SCRIPTS_DIR.parent / "models").exists():
    BASE_DIR = SCRIPTS_DIR.parent
elif (SCRIPTS_DIR / "models").exists():
    BASE_DIR = SCRIPTS_DIR
else:
    BASE_DIR = SCRIPTS_DIR

CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
METRICS_DIR = BASE_DIR / "results" / "metrics"
FLOPS_DIR = METRICS_DIR / "flops"
FLOPS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SHAPE = (1, 128, 251)  # log-mel spectrogram (C, H, W), without batch dimension

# One checkpoint per model (lowest seed available), used only to rebuild the architecture.
MODEL_SPECS = {
    "cnn":        {"display": "CNN-RespiratoryNet", "kind": "cnn",
                   "pattern": "cnn_seed*.pth"},
    "hybrid":     {"display": "CfC-Hybrid",         "kind": "cfc",
                   "pattern": "hybrid_seed*.pth"},
    "transposed": {"display": "CfC-Transposed",     "kind": "cfc",
                   "pattern": "transposed_seed*.pth"},
}


logger = logging.getLogger(__name__)


def setup_logging():
    log_file = BASE_DIR / "flops_analysis_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return log_file


def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_arch(checkpoint_path: Path, kind: str, script09, script11a,
                    device: torch.device):
    """Rebuild the model from a checkpoint (same constructors as Script 17)."""
    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = ck.get("config", {})

    if kind == "cnn":
        model = script09.RespiratoryNet(
            n_classes=config.get("n_classes", 4),
            input_channels=config.get("input_channels", 1),
            base_filters=config.get("base_filters", 16),
            n_conv_blocks=config.get("n_conv_blocks", 4),
            fc_hidden_dim=config.get("fc_hidden_dim", 128),
            dropout_rate=config.get("dropout_rate", 0.5),
        ).to(device)
    elif kind == "cfc":
        arch = config.get("architecture")
        n_classes = config.get("n_classes", 4)
        if arch == "hybrid":
            model = script11a.CfCMelHybridV6(n_classes=n_classes, config=config).to(device)
        elif arch == "transposed":
            model = script11a.CfCMelTransposedV6(n_classes=n_classes, config=config).to(device)
        else:
            raise ValueError(f"unknown architecture: {arch!r}")
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, config


def find_one_checkpoint(ckpt_dir: Path, pattern: str):
    """Return the first checkpoint (sorted by name) matching the pattern, or None."""
    cands = sorted(ckpt_dir.glob(pattern))
    return cands[0] if cands else None


def count_conv_linear_flops(model: nn.Module, input_shape, device):
    """
    Count MACs of the non-recurrent layers with ptflops (which ignores the
    recurrent part anyway). If ptflops is unavailable, count Conv2D and Linear
    with forward hooks. Returns (macs, method_description).
    """
    try:
        from ptflops import get_model_complexity_info

        macs, _ = get_model_complexity_info(
            model, input_shape, as_strings=False,
            print_per_layer_stat=False, verbose=False,
        )
        return float(macs), "ptflops (Conv2D+Linear+BN; recurrent excluded)"
    except Exception as e:
        logger.warning(f"  ptflops unavailable or failed ({e}); using manual hooks.")
        return _count_with_hooks(model, input_shape, device), "manual hooks (Conv2D+Linear)"


def _count_with_hooks(model, input_shape, device):
    """Fallback: count Conv2D and Linear MACs with forward hooks."""
    macs_total = [0]
    hooks = []

    def conv_hook(module, inp, out):
        # MACs = out_elements * (in_ch/groups * kh * kw)
        out_h, out_w = out.shape[2], out.shape[3]
        kh, kw = module.kernel_size
        in_ch = module.in_channels // module.groups
        macs = out.shape[1] * out_h * out_w * in_ch * kh * kw
        macs_total[0] += int(macs)

    def linear_hook(module, inp, out):
        macs_total[0] += int(module.in_features * module.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    with torch.no_grad():
        x = torch.randn(1, *input_shape, device=device)
        model(x)

    for h in hooks:
        h.remove()
    return float(macs_total[0])


def estimate_recurrent_macs(model: nn.Module, config: dict, input_shape,
                            device, n_linear_maps: int = 4):
    """
    Analytical MACs of the recurrent layer, which automatic counters ignore.

    A GRU/CfC-like cell applies, per timestep, n_linear_maps input maps
    (in_features -> hidden) and state maps (hidden -> hidden):
        MACs ~= T * n_linear_maps * (in_features*hidden + hidden*hidden)
    T is the number of timesteps fed to the CfC: the width after the Conv2D
    frontend for the hybrid model, the number of mel bands (128) for the
    transposed model. If an nn.GRU is found, n_linear_maps is set to its 3 gates.
    Returns a dict with the breakdown and assumptions.
    """
    in_features = config.get("cfc_in_features")
    hidden = config.get("hidden_size")

    if in_features is None or hidden is None:
        # Infer from the module itself
        for m in model.modules():
            if isinstance(m, nn.GRU):
                in_features = in_features or m.input_size
                hidden = hidden or m.hidden_size
                n_linear_maps = 3
                break

    if in_features is None or hidden is None:
        return {
            "macs": 0.0,
            "note": "Could not determine in_features/hidden of the CfC; "
                    "recurrent part NOT counted. Check config.",
            "countable": False,
        }

    T = _infer_timesteps(model, config, input_shape, device)

    per_timestep = n_linear_maps * (in_features * hidden + hidden * hidden)
    macs = float(T * per_timestep)

    return {
        "macs": macs,
        "timesteps_T": int(T),
        "in_features": int(in_features),
        "hidden_size": int(hidden),
        "n_linear_maps": int(n_linear_maps),
        "formula": "T * n_linear_maps * (in_features*hidden + hidden*hidden)",
        "countable": True,
    }


def _infer_timesteps(model, config, input_shape, device):
    """
    Number of timesteps fed to the CfC: the temporal width after the conv
    frontend (hybrid) or the number of mel bands (transposed).
    """
    arch = config.get("architecture")
    if arch == "transposed":
        # forward does x.squeeze(1) -> (B, 128, 251): mel bands are the timesteps
        return input_shape[1]

    if arch == "hybrid" and hasattr(model, "conv"):
        with torch.no_grad():
            x = torch.randn(1, *input_shape, device=device)
            feat = model.conv(x)  # (B, C, H, W)
        # In the hybrid forward, timesteps = W (temporal dimension)
        return feat.shape[3]

    return input_shape[2]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_memory_mb(model):
    """Parameter and buffer memory in MB (consistent with Script 15)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / (1024 ** 2)


def write_outputs(results: dict, sysinfo: dict):
    """Write the JSON breakdown, the flat CSV table and the Markdown summary."""
    json_path = FLOPS_DIR / "flops_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "system": sysinfo,
                   "input_shape": INPUT_SHAPE,
                   "results": results}, f, indent=2, default=str)
    logger.info(f"  JSON:        {json_path}")

    csv_path = FLOPS_DIR / "flops_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,params,memory_mb,macs_conv_linear,macs_recurrent,"
                "macs_total,flops_total_G\n")
        for mk, r in results.items():
            f.write(f"{mk},{r['params']},{r['memory_mb']:.4f},"
                    f"{r['macs_conv_linear']:.0f},{r['macs_recurrent']:.0f},"
                    f"{r['macs_total']:.0f},{r['flops_total_G']:.4f}\n")
    logger.info(f"  CSV:         {csv_path}")

    md_path = FLOPS_DIR / "flops_summary.md"
    lines = []
    lines.append("# FLOPs / computational efficiency analysis\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_  ")
    lines.append(f"_Input: {INPUT_SHAPE} (mel-spectrogram), batch=1_\n")
    lines.append("## Computational efficiency table\n")
    lines.append("| Model | Params | Memory (MB) | MACs conv/linear | "
                 "MACs recurrent | MACs total | GFLOPs* |")
    lines.append("|---|---|---|---|---|---|---|")
    for mk, r in results.items():
        disp = MODEL_SPECS[mk]["display"]
        lines.append(
            f"| {disp} | {r['params']:,} | {r['memory_mb']:.2f} | "
            f"{r['macs_conv_linear']:,.0f} | {r['macs_recurrent']:,.0f} | "
            f"{r['macs_total']:,.0f} | {r['flops_total_G']:.3f} |"
        )
    lines.append("")
    lines.append("*GFLOPs ≈ 2 × total MACs (1 MAC = 1 mult + 1 add).\n")
    lines.append("### Counting methodology\n")
    lines.append("Automatic counters (ptflops/thop/fvcore) do NOT instrument "
                 "recurrent layers (fvcore reports `aten::gru` as an unsupported "
                 "operator). The count is therefore done in two parts:\n")
    lines.append("1. **Conv2D + Linear + BN**: counted with ptflops.")
    lines.append("2. **Recurrent layer (CfC)**: added analytically with the formula "
                 "`T · K · (in·hidden + hidden²)`, where T = timesteps, "
                 "K = number of linear maps of the cell.\n")
    lines.append("Recurrent term per model:\n")
    for mk, r in results.items():
        rec = r.get("recurrent_detail", {})
        if rec.get("countable"):
            lines.append(f"- **{MODEL_SPECS[mk]['display']}**: T={rec['timesteps_T']}, "
                         f"in={rec['in_features']}, hidden={rec['hidden_size']}, "
                         f"K={rec['n_linear_maps']} → {rec['macs']:,.0f} MACs")
        elif r["macs_recurrent"] == 0 and MODEL_SPECS[mk]["kind"] == "cnn":
            lines.append(f"- **{MODEL_SPECS[mk]['display']}**: no recurrent layer (pure CNN).")
    lines.append("")
    lines.append("> **Note:** the efficiency argument is NOT 'fewer FLOPs than "
                 "the CNN' (the recurrent term can match or exceed the conv term). The "
                 "defensible argument is the comparison with the SOTA transformer "
                 "(~85M params), relative to which these models are ~2-3 orders of "
                 "magnitude smaller, combined with <2MB of memory and real-time "
                 "latency on the Pi 5.\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  MD:          {md_path}")
    return md_path


def main():
    global CHECKPOINTS_DIR

    parser = argparse.ArgumentParser(description="FLOPs/MACs and edge efficiency.")
    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--cfc_linear_maps", type=int, default=4,
                        help="Number of linear maps per recurrent cell (default 4).")
    parser.add_argument("--models", nargs="+", default=["cnn", "hybrid", "transposed"],
                        choices=["cnn", "hybrid", "transposed"])
    args = parser.parse_args()

    log_file = setup_logging()
    if args.checkpoints_dir is not None:
        CHECKPOINTS_DIR = Path(args.checkpoints_dir)

    logger.info("=" * 70)
    logger.info("Script 19: FLOPs / edge efficiency")
    logger.info("=" * 70)
    logger.info(f"  BASE_DIR:      {BASE_DIR}")
    logger.info(f"  Checkpoints:   {CHECKPOINTS_DIR}")
    logger.info(f"  Outputs:       {FLOPS_DIR}")
    logger.info(f"  Input shape:   {INPUT_SHAPE}")
    logger.info(f"  cfc_linear_maps (K): {args.cfc_linear_maps}")

    device = torch.device("cpu")  # FLOP counting does not need a GPU
    sysinfo = {"device": str(device), "torch_version": torch.__version__,
               "python_version": sys.version.split()[0]}

    logger.info("\n[Importing project modules]")
    script09 = script11a = None
    try:
        if "cnn" in args.models:
            script09 = import_module_from_path("script09", SCRIPTS_DIR / "09_cnn_training.py")
            logger.info("  Script 09 imported")
        if "hybrid" in args.models or "transposed" in args.models:
            script11a = import_module_from_path("script11a", SCRIPTS_DIR / "11a_cfc_mel_v7.py")
            logger.info("  Script 11a imported")
    except Exception as e:
        logger.error(f"  Error importing modules: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    results = {}
    for mk in args.models:
        spec = MODEL_SPECS[mk]
        ckpt = find_one_checkpoint(CHECKPOINTS_DIR, spec["pattern"])
        if ckpt is None:
            logger.warning(f"\n[WARN] No checkpoint for '{mk}' "
                           f"(pattern {spec['pattern']}). Skipped.")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"MODEL: {spec['display']}  (architecture from {ckpt.name})")
        logger.info(f"{'='*70}")

        try:
            model, config = load_model_arch(ckpt, spec["kind"], script09, script11a, device)
        except Exception as e:
            logger.error(f"  Error loading architecture: {e}")
            import traceback; traceback.print_exc()
            continue

        params = count_params(model)
        mem = model_memory_mb(model)

        macs_cl, method_cl = count_conv_linear_flops(model, INPUT_SHAPE, device)
        logger.info(f"  Params:              {params:,}")
        logger.info(f"  Memory:              {mem:.3f} MB")
        logger.info(f"  MACs conv/linear:    {macs_cl:,.0f}  [{method_cl}]")

        if spec["kind"] == "cfc":
            rec = estimate_recurrent_macs(model, config, INPUT_SHAPE, device,
                                          n_linear_maps=args.cfc_linear_maps)
            macs_rec = rec["macs"]
            if rec.get("countable"):
                logger.info(f"  MACs recurrent:      {macs_rec:,.0f}  "
                            f"[T={rec['timesteps_T']}, in={rec['in_features']}, "
                            f"hidden={rec['hidden_size']}, K={rec['n_linear_maps']}]")
            else:
                logger.warning(f"  MACs recurrent:      NOT counted: {rec.get('note')}")
        else:
            rec = {"countable": False, "macs": 0.0}
            macs_rec = 0.0
            logger.info(f"  MACs recurrent:      0 (pure CNN, no recurrence)")

        macs_total = macs_cl + macs_rec
        flops_g = 2 * macs_total / 1e9
        logger.info(f"  MACs TOTAL:          {macs_total:,.0f}")
        logger.info(f"  GFLOPs (≈2×MACs):    {flops_g:.4f}")

        results[mk] = {
            "display": spec["display"],
            "params": params,
            "memory_mb": mem,
            "macs_conv_linear": macs_cl,
            "macs_conv_linear_method": method_cl,
            "macs_recurrent": macs_rec,
            "recurrent_detail": rec,
            "macs_total": macs_total,
            "flops_total_G": flops_g,
        }

    if not results:
        logger.error("\n[ERROR] No model could be analysed.")
        sys.exit(1)

    logger.info(f"\n{'='*70}")
    logger.info("COMPARATIVE SUMMARY")
    logger.info(f"{'='*70}")
    for mk, r in results.items():
        logger.info(f"  {r['display']:22s}: {r['params']:>9,} params  |  "
                    f"{r['macs_total']:>14,.0f} MACs  |  "
                    f"{r['flops_total_G']:.3f} GFLOPs")

    logger.info(f"\n{'='*70}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'='*70}")
    md_path = write_outputs(results, sysinfo)

    logger.info(f"\n{'='*70}")
    logger.info("FLOPs ANALYSIS COMPLETED")
    logger.info(f"{'='*70}")
    logger.info(f"  Summary report: {md_path}")


if __name__ == "__main__":
    main()
