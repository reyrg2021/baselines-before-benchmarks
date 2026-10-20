#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 23: Post-training static INT8 quantization of the CNN for edge deployment.

Quantizes CNN-RespiratoryNet (FP32) to INT8 with static post-training
quantization (PTQ, FX graph mode), calibrated on the training set, and measures
the size reduction on disk and the official ICBHI score retained on the test set
(FP32 vs INT8). The INT8 model is exported as TorchScript for the Raspberry Pi.
Static PTQ is used because dynamic quantization only covers Linear/LSTM layers
and would leave the convolutions, which dominate this model, in FP32. FX graph
mode avoids inserting QuantStub/DeQuantStub into the model class.

The main result keeps the final Linear classifier in FP32 (--no_quantize_last),
because quantizing it collapses sensitivity. Without that flag the full INT8
model is produced as an ablation, and its files get the suffix '_full' so that
the main artifacts are not overwritten.

Inputs:  models/checkpoints/{ckpt_prefix}{seed}.pth, data/splits/<split>/,
         scripts/08_dataset_dataloader.py, scripts/09_cnn_training.py, scripts/focal_loss.py
Outputs: models/checkpoints/cnn_int8_scripted[_full].pt (TorchScript INT8),
         results/quantization/cnn_int8_quant_report[_full].json; quantize_cnn_int8_log.txt
Usage:
    python scripts/23_quantize_cnn_int8.py --seeds 42 123 7 99 314 --no_quantize_last
    python scripts/23_quantize_cnn_int8.py --checkpoint cnn_seed7 --no_quantize_last
    python scripts/23_quantize_cnn_int8.py --checkpoint_path models/checkpoints/cnn_seed7.pth
"""

import sys
import json
import statistics
import copy
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler("quantize_cnn_int8_log.txt", mode="w", encoding="utf-8")])
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
SPLITS_DIR = BASE_DIR / "data" / "splits"
FEATURES_DIR = BASE_DIR / "data" / "features"
RESULTS_DIR = BASE_DIR / "results" / "quantization"

INPUT_SHAPE = (1, 1, 128, 251)   # (batch, channel, mel_bins, frames): RespiratoryNet input


def _import_module(modname, filename):
    from importlib.util import spec_from_file_location, module_from_spec
    here = Path(__file__).resolve().parent
    for path in [here / filename, SCRIPTS_DIR / filename, BASE_DIR / filename]:
        if path.exists():
            sys.path.insert(0, str(path.parent))
            spec = spec_from_file_location(modname, str(path))
            mod = module_from_spec(spec); spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"File not found: {filename}")


def load_fp32_cnn(checkpoint_path, device="cpu"):
    """Load the FP32 CNN from a checkpoint (same logic as Script 15)."""
    mod = _import_module("mod_cnn", "09_cnn_training.py")
    RespiratoryNet = getattr(mod, "RespiratoryNet")
    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = ck.get("config", {})
    if not config and hasattr(mod, "TRAINING_CONFIG"):
        config = mod.TRAINING_CONFIG
    try:
        model = RespiratoryNet(config=config) if "config" in RespiratoryNet.__init__.__code__.co_varnames \
                else RespiratoryNet()
    except TypeError:
        model = RespiratoryNet()
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    val_icbhi = ck.get("val_icbhi", None)
    logger.info(f"  FP32 CNN loaded from {Path(checkpoint_path).name} "
                f"(val_icbhi={val_icbhi})")
    return model


def get_dataloaders(split_strategy="strategy1_official", batch_size=32):
    """Return the dataloader dict of Script 08 for the requested split.

    The split is passed explicitly because the Script 08 default is
    'strategy2_custom' (622 test cycles instead of 2733), which would give a
    wrong ICBHI score.
    """
    s08 = _import_module("mod_08", "08_dataset_dataloader.py")
    for fn_name in ["create_dataloaders", "build_dataloaders", "get_dataloaders"]:
        if hasattr(s08, fn_name):
            fn = getattr(s08, fn_name)
            varnames = fn.__code__.co_varnames
            kwargs = {}
            if "strategy" in varnames:
                kwargs["strategy"] = split_strategy
            elif "split_strategy" in varnames:
                kwargs["split_strategy"] = split_strategy
            if "batch_size" in varnames:
                kwargs["batch_size"] = batch_size
            data = fn(**kwargs)
            # Sanity check of the effective split through the test-set size
            try:
                n_test = len(data["test_loader"].dataset)
                logger.info(f"  [split] {split_strategy} → test set = {n_test} cycles "
                            f"({'OK strategy1_official' if n_test > 2000 else 'WARNING: looks like strategy2'})")
            except Exception:
                pass
            return data
    raise RuntimeError("Dataloader function not found in 08_dataset_dataloader.py.")


def evaluate_icbhi(model, loader, device="cpu"):
    """Official ICBHI score, Se and Sp on a loader (uses focal_loss.official_icbhi)."""
    fl = _import_module("mod_fl", "focal_loss.py")
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            logits = model(x)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(np.asarray(y))
    preds, labels = np.array(preds), np.array(labels)
    score, se, sp = fl.official_icbhi(labels, preds)
    return float(score), float(se), float(sp)


def serialized_size_mb(path):
    """Size on disk (MB) of a serialized model."""
    return Path(path).stat().st_size / (1024 * 1024)


def quantize_int8_static(fp32_model, calib_loader, device="cpu", n_calib_batches=0,
                         observer="minmax", quantize_last=True):
    """
    Static PTQ via FX. Returns the INT8 model (CPU).

    Settings that prevent collapse in compact models:
      n_calib_batches <= 0  -> calibrate on the whole training set.
      observer="minmax"     -> MinMaxObserver for activations (true range, without
                               the clipping of HistogramObserver, which collapses
                               small networks) and per-channel weights.
                               "histogram" uses the backend's default qconfig.
      quantize_last=False   -> keep the last Linear (classifier) in FP32, since
                               quantizing it collapses sensitivity.
    """
    from torch.ao.quantization import (QConfig, MinMaxObserver,
                                        PerChannelMinMaxObserver,
                                        get_default_qconfig_mapping)
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

    available = torch.backends.quantized.supported_engines
    engine = "fbgemm" if "fbgemm" in available else (
             "qnnpack" if "qnnpack" in available else (available[0] if available else "none"))
    torch.backends.quantized.engine = engine
    logger.info(f"  [PTQ] Backend: {engine} | activation observer: {observer} | "
                f"weights: per-channel | quantize last layer: {quantize_last}")

    if observer == "histogram":
        qconfig_mapping = get_default_qconfig_mapping(engine)
    else:
        act_obs = MinMaxObserver.with_args(dtype=torch.quint8,
                                           qscheme=torch.per_tensor_affine)
        wt_obs = PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
                                                    qscheme=torch.per_channel_symmetric)
        qconfig = QConfig(activation=act_obs, weight=wt_obs)
        qconfig_mapping = QConfigMapping().set_global(qconfig)

    model_to_quantize = copy.deepcopy(fp32_model).to("cpu").eval()
    example_inputs = (torch.randn(INPUT_SHAPE),)

    # Optionally leave the last Linear unquantized (logits in FP32)
    if not quantize_last:
        last_linear = None
        for name, mod in model_to_quantize.named_modules():
            if isinstance(mod, nn.Linear):
                last_linear = name
        if last_linear is not None:
            qconfig_mapping = qconfig_mapping.set_module_name(last_linear, None)
            logger.info(f"  [PTQ] Last Linear left unquantized: '{last_linear}'")

    logger.info("  [PTQ] prepare_fx...")
    prepared = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
    prepared.eval()

    total = len(calib_loader)
    n_use = total if n_calib_batches <= 0 else min(n_calib_batches, total)
    logger.info(f"  [PTQ] Calibrating with {n_use}/{total} training batches...")
    with torch.no_grad():
        for i, batch in enumerate(calib_loader):
            if n_calib_batches > 0 and i >= n_calib_batches:
                break
            prepared(batch[0].to("cpu"))
    logger.info("  [PTQ] convert_fx...")
    quantized = convert_fx(prepared)
    quantized.eval()
    return quantized


def parse_args():
    p = argparse.ArgumentParser(description="Script 23: INT8 quantization of the CNN")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint name without .pth (e.g. cnn_seed7), looked up in models/checkpoints/")
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="Full path to the .pth (alternative to --checkpoint)")
    p.add_argument("--split", type=str, default="strategy1_official")
    p.add_argument("--n_calib_batches", type=int, default=0,
                   help="Calibration batches. 0 or negative = whole training set (recommended).")
    p.add_argument("--observer", choices=["minmax", "histogram"], default="minmax",
                   help="Activation observer. minmax is more robust to collapse.")
    p.add_argument("--no_quantize_last", action="store_true",
                   help="Keep the last Linear (classifier) in FP32.")
    # Multi-seed mode: quantize several checkpoints and aggregate the mean ± std delta
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seeds to quantize (e.g. --seeds 42 123 7 99 314). Enables "
                        "multi-seed mode: quantizes {ckpt_prefix}{S}.pth for each one.")
    p.add_argument("--ckpt_prefix", type=str, default="cnn_official_seed",
                   help="Prefix of the multi-seed checkpoints (the seed is appended).")
    p.add_argument("--export_seed", type=int, default=None,
                   help="Seed whose TorchScript .pt is exported for the Pi "
                        "(default: the first). Latency does not depend on the seed.")
    return p.parse_args()


def _variant_tag(args):
    """File suffix of the quantization variant.

    Without --no_quantize_last the classifier is also quantized (full INT8,
    the ablation) -> suffix '_full', so that the artifacts of the main result
    (classifier in FP32) keep the base name.
    """
    return "" if args.no_quantize_last else "_full"


def quantize_one(ckpt_path, data, device, args, export_scripted):
    """Quantize ONE checkpoint and return its metrics (does not write the JSON).

    The already-built dataloaders are reused across seeds.
    """
    # Fresh FP32 model per seed
    fp32 = load_fp32_cnn(ckpt_path, device)
    fp32_score, fp32_se, fp32_sp = evaluate_icbhi(fp32, data["test_loader"], device)
    logger.info(f"    FP32 → ICBHI {fp32_score:.4f} (Se={fp32_se:.3f} Sp={fp32_sp:.3f})")

    int8 = quantize_int8_static(fp32, data["train_loader"], device,
                                n_calib_batches=args.n_calib_batches,
                                observer=args.observer,
                                quantize_last=not args.no_quantize_last)
    int8_score, int8_se, int8_sp = evaluate_icbhi(int8, data["test_loader"], device)
    logger.info(f"    INT8 → ICBHI {int8_score:.4f} (Se={int8_se:.3f} Sp={int8_sp:.3f}) "
                f"| Δ {int8_score - fp32_score:+.4f}")

    # Sizes (FP32 state_dict vs INT8 TorchScript). Size depends only on the
    # architecture; it is measured per seed for traceability.
    fp32_tmp = RESULTS_DIR / f"_tmp_fp32_{ckpt_path.stem}.pth"
    torch.save(fp32.state_dict(), fp32_tmp)
    fp32_mb = serialized_size_mb(fp32_tmp)
    fp32_tmp.unlink(missing_ok=True)

    scripted = torch.jit.script(int8)
    rel_score = None
    scripted_for_pi = None
    if export_scripted:
        # Export the portable model for the Pi and verify it by reloading
        int8_path = CHECKPOINTS_DIR / f"cnn_int8_scripted{_variant_tag(args)}.pt"
        torch.jit.save(scripted, str(int8_path))
        reloaded = torch.jit.load(str(int8_path))
        rel_score, _, _ = evaluate_icbhi(reloaded, data["test_loader"], device)
        int8_mb = serialized_size_mb(int8_path)
        scripted_for_pi = str(int8_path)
        logger.info(f"    TorchScript exported and verified (reloaded ICBHI {rel_score:.4f})")
    else:
        int8_tmp = CHECKPOINTS_DIR / f"_tmp_int8_{ckpt_path.stem}.pt"
        torch.jit.save(scripted, str(int8_tmp))
        int8_mb = serialized_size_mb(int8_tmp)
        int8_tmp.unlink(missing_ok=True)

    return {
        "checkpoint": ckpt_path.name,
        "fp32_icbhi": fp32_score, "fp32_se": fp32_se, "fp32_sp": fp32_sp,
        "int8_icbhi": int8_score, "int8_se": int8_se, "int8_sp": int8_sp,
        "delta": round(int8_score - fp32_score, 4),
        "fp32_mb": round(fp32_mb, 3), "int8_mb": round(int8_mb, 3),
        "size_reduction_x": round(fp32_mb / max(int8_mb, 1e-9), 2),
        "int8_reloaded_torchscript": rel_score,
        "scripted_model_for_pi": scripted_for_pi,
    }


def _agg(per_seed, key):
    """Mean ± population standard deviation of a metric over the seeds."""
    vals = [r[key] for r in per_seed]
    mean = float(statistics.mean(vals))
    std = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "values": vals}


def main():
    args = parse_args()
    device = "cpu"   # INT8 quantization and inference run on CPU
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Script 23: INT8 quantization of the CNN")
    logger.info("=" * 70)
    _variant = "FULL INT8: classifier quantized (ABLATION)" if not args.no_quantize_last \
                else "classifier in FP32 (main result)"
    logger.info(f"  Variant: {_variant}")
    logger.info(f"  Output files get the suffix: '{_variant_tag(args)}'  "
                f"(report = cnn_int8_quant_report{_variant_tag(args)}.json)")

    # Dataloaders built once (calibration = train; evaluation = test)
    logger.info("\n[1/2] Loading dataloaders (mels)...")
    data = get_dataloaders(split_strategy=args.split, batch_size=32)

    # Multi-seed mode
    if args.seeds:
        export_seed = args.export_seed if args.export_seed is not None else args.seeds[0]
        logger.info(f"\n[2/2] Quantizing {len(args.seeds)} seeds: {args.seeds} "
                    f"(exports the .pt of seed {export_seed})")
        per_seed = []
        for s in args.seeds:
            ckpt = CHECKPOINTS_DIR / f"{args.ckpt_prefix}{s}.pth"
            if not ckpt.exists():
                logger.error(f"  [!] {ckpt.name} not found; seed {s} SKIPPED."); continue
            logger.info(f"\n  -- Seed {s} ({ckpt.name}) --")
            r = quantize_one(ckpt, data, device, args, export_scripted=(s == export_seed))
            r["seed"] = s
            per_seed.append(r)

        if not per_seed:
            logger.error("No seed was quantized (missing checkpoints?)."); sys.exit(1)

        fp32_mb = per_seed[0]["fp32_mb"]; int8_mb = per_seed[0]["int8_mb"]
        report = {
            "mode": "multiseed",
            "variant": "full_int8" if not args.no_quantize_last else "classifier_fp32",
            "quantize_classifier": not args.no_quantize_last,
            "split": args.split,
            "input_shape": list(INPUT_SHAPE),
            "seeds": [r["seed"] for r in per_seed],
            "fp32_icbhi": _agg(per_seed, "fp32_icbhi"),
            "int8_icbhi": _agg(per_seed, "int8_icbhi"),
            "delta_icbhi_int8_minus_fp32": _agg(per_seed, "delta"),
            "size_mb": {"fp32": fp32_mb, "int8": int8_mb,
                        "reduction_x": round(fp32_mb / max(int8_mb, 1e-9), 2)},
            "scripted_model_for_pi": next(
                (r["scripted_model_for_pi"] for r in per_seed if r["scripted_model_for_pi"]), None),
            "per_seed": per_seed,
        }
        d = report["delta_icbhi_int8_minus_fp32"]; fi = report["fp32_icbhi"]; ii = report["int8_icbhi"]
        logger.info("\n" + "=" * 70)
        logger.info("MULTI-SEED QUANTIZATION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"  Seeds: {report['seeds']}")
        logger.info(f"  Size:     FP32 {fp32_mb:.2f} MB → INT8 {int8_mb:.2f} MB "
                    f"({report['size_mb']['reduction_x']}× smaller)")
        logger.info(f"  ICBHI FP32: {fi['mean']:.4f} ± {fi['std']:.4f}")
        logger.info(f"  ICBHI INT8: {ii['mean']:.4f} ± {ii['std']:.4f}")
        logger.info(f"  Δ (INT8−FP32): {d['mean']:+.4f} ± {d['std']:.4f}  (over {len(per_seed)} seeds)")

    # Single-checkpoint mode
    else:
        if args.checkpoint_path:
            ckpt_path = Path(args.checkpoint_path)
        elif args.checkpoint:
            ckpt_path = CHECKPOINTS_DIR / f"{args.checkpoint}.pth"
        else:
            logger.error("Pass --seeds, or --checkpoint NAME, or --checkpoint_path PATH")
            sys.exit(1)
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}"); sys.exit(1)

        logger.info(f"\n[2/2] Quantizing 1 checkpoint: {ckpt_path.name}")
        r = quantize_one(ckpt_path, data, device, args, export_scripted=True)
        report = {
            "checkpoint": r["checkpoint"], "split": args.split,
            "variant": "full_int8" if not args.no_quantize_last else "classifier_fp32",
            "quantize_classifier": not args.no_quantize_last,
            "input_shape": list(INPUT_SHAPE),
            "fp32": {"icbhi_official": r["fp32_icbhi"], "se": r["fp32_se"], "sp": r["fp32_sp"],
                     "size_mb": r["fp32_mb"]},
            "int8": {"icbhi_official": r["int8_icbhi"], "se": r["int8_se"], "sp": r["int8_sp"],
                     "size_mb": r["int8_mb"], "icbhi_reloaded_torchscript": r["int8_reloaded_torchscript"]},
            "delta_icbhi_int8_minus_fp32": r["delta"],
            "size_reduction_x": r["size_reduction_x"],
            "scripted_model_for_pi": r["scripted_model_for_pi"],
        }
        logger.info("\n" + "=" * 70)
        logger.info("QUANTIZATION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"  Size:   FP32 {r['fp32_mb']:.2f} MB → INT8 {r['int8_mb']:.2f} MB "
                    f"({r['size_reduction_x']}× smaller)")
        logger.info(f"  ICBHI:  FP32 {r['fp32_icbhi']:.4f} → INT8 {r['int8_icbhi']:.4f} "
                    f"(Δ {r['delta']:+.4f})")

    report_path = RESULTS_DIR / f"cnn_int8_quant_report{_variant_tag(args)}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"  Report: {report_path}")
    if args.no_quantize_last:
        logger.info("\n  → Copy 'cnn_int8_scripted.pt' to the Pi for the latency benchmark.")
    else:
        logger.info("\n  → ABLATION variant (full INT8): NOT for deployment; "
                    "it measures the sensitivity collapse caused by quantizing the classifier.")
    return report


if __name__ == "__main__":
    main()
