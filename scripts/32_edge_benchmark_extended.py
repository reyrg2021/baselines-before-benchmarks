#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 32: Extended edge profiling of all seven configurations on the Raspberry Pi 5.

The earlier benchmark (Script 15) profiled only CNN FP32, CNN INT8 and CfC-Hybrid. A
deployment Pareto frontier must also include MobileNetV3-Small and PANNs CNN6, so this
script profiles the seven configurations in one session with the same protocol as
Script 15 (its functions are imported, not reimplemented):

    cnn            CNN-RespiratoryNet FP32          mel input (1,1,128,251)
    cnn_int8       CNN-RespiratoryNet INT8 (TorchScript, qnnpack)
    hybrid         CfC-Hybrid (mmRNN)               mel input
    mnv3_imagenet  MobileNetV3-Small (ImageNet)     mel input (repeated to 3 channels)
    mnv3_scratch   MobileNetV3-Small (scratch)      mel input
    panns          PANNs CNN6 (AudioSet)            waveform input (1, 256000) @ 32 kHz
    panns_tune     PANNs CNN6 (AudioSet, tune)      waveform input

Per model and thread count: parameters, model memory (parameters + buffers), disk size,
MACs, batch-1 latency p50/p95/p99/mean/sd, peak process RSS, analytical A76 energy per
inference, and optional continuous thermal stress. For the CfC the valid MAC count is
the Script 19 value (101M), since neither ptflops nor thop instruments the recurrent
loop; it is reported as an external reference. Preprocessing is measured once: the
Script 06 mel of an 8 s cycle at 16 kHz (end-to-end latency of mel models =
preprocessing + inference) and 16 -> 32 kHz resampling (end-to-end for PANNs, whose mel
is internal). Measuring all models back to back with the same warmup and iterations
removes the initial thermal state as a confounder of the ranking; --cooldown_target_temp
additionally equalizes the starting temperature.

Inputs:  models/checkpoints/ (see resolve_checkpoint), scripts 06, 09, 11a, 15, 21, 22,
         panns_cnn6.py, torch_cfc.py
Outputs: results/edge_benchmark/pi5_extended/edge_extended_<label>_<timestamp>.json/.csv/.md,
         script32_log.txt
Usage (on the Raspberry Pi 5, from scripts/):
    python3 32_edge_benchmark_extended.py --num_threads_list 1 4 \\
        --num_iter 200 --num_warmup 20 --cooldown_target_temp 55
    Options: [--models ...] [--stress_test_minutes 10] [--skip_preproc] [--device_label LABEL]
"""

import os
import sys
import json
import time
import argparse
import logging
import platform
import tempfile
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import resource
except ImportError:          # Windows
    resource = None
try:
    import psutil
except ImportError:
    psutil = None


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
BASE_DIR_FIXED = Path(__file__).resolve().parent.parent
if BASE_DIR_FIXED.exists():
    BASE_DIR = BASE_DIR_FIXED
elif (SCRIPTS_DIR.parent / "models").exists():
    BASE_DIR = SCRIPTS_DIR.parent
else:
    BASE_DIR = SCRIPTS_DIR
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
OUT_DIR = BASE_DIR / "results" / "edge_benchmark" / "pi5_extended"

MEL_SHAPE = (1, 1, 128, 251)
WAVE_SHAPE = (1, 256000)          # 8 s @ 32 kHz (raw_audio_dataset)
CYCLE_SECONDS = 8.0
SR_MEL = 16000
SR_PANNS = 32000

# External MAC reference including the recurrent loop (Script 19, ptflops + manual count)
MACS_REFERENCE = {"cnn": 409e6, "cnn_int8": 409e6, "hybrid": 101e6}

MODELS = {
    "cnn":           {"display": "CNN-RespiratoryNet (FP32)", "kind": "cnn",
                      "prefix": "cnn_official", "input": "mel"},
    "cnn_int8":      {"display": "CNN-RespiratoryNet (INT8)", "kind": "torchscript",
                      "file": "cnn_int8_scripted.pt", "input": "mel"},
    "hybrid":        {"display": "CfC-Hybrid (mmRNN)", "kind": "cfc",
                      "prefix": "hybrid_official", "input": "mel"},
    "mnv3_imagenet": {"display": "MobileNetV3-Small (ImageNet)", "kind": "mnv3",
                      "prefix": "tl_mobilenet_v3_small_pretrained_official", "input": "mel"},
    "mnv3_scratch":  {"display": "MobileNetV3-Small (scratch)", "kind": "mnv3",
                      "prefix": "tl_mobilenet_v3_small_official", "input": "mel"},
    "panns":         {"display": "PANNs CNN6 (AudioSet)", "kind": "panns",
                      "prefix": "panns_cnn6_official", "input": "wave"},
    "panns_tune":    {"display": "PANNs CNN6 (AudioSet, tune)", "kind": "panns",
                      "prefix": "panns_cnn6_tune_official", "input": "wave"},
}
DEFAULT_ORDER = ["cnn", "cnn_int8", "hybrid", "mnv3_imagenet", "mnv3_scratch", "panns", "panns_tune"]

logger = logging.getLogger(__name__)


def setup_logging():
    """Log to stdout and append to script32_log.txt in BASE_DIR."""
    log_file = BASE_DIR / "script32_log.txt"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(log_file, mode="a", encoding="utf-8")], force=True)
    return log_file


def import_script(name, filename):
    """Import a numbered script from SCRIPTS_DIR as a module."""
    path = SCRIPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_checkpoint(key, s15):
    """Same policy as Script 15: the checkpoint with the highest val_icbhi for the prefix."""
    info = MODELS[key]
    if info["kind"] == "torchscript":
        p = CHECKPOINTS_DIR / info["file"]
        return (p if p.exists() else None), None, None
    return s15.find_best_checkpoint(info["prefix"])


def load_model(key, ckpt, device, s15, mods):
    """Load a model in eval mode; CNN, CfC and INT8 are delegated to Script 15's loader."""
    import torch
    info = MODELS[key]
    if info["kind"] in ("cnn", "cfc", "torchscript"):
        # Script 15 key map: cnn / hybrid / cnn_int8
        k15 = {"cnn": "cnn", "cfc": "hybrid", "torchscript": "cnn_int8"}[info["kind"]]
        return s15.load_model_for_inference(k15, ckpt, device)
    ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    if info["kind"] == "mnv3":
        s21 = mods["s21"]
        # pretrained=False: weights come from the checkpoint; avoids downloading ImageNet on the Pi
        model = s21.BackboneClassifier(
            backbone_name=cfg.get("backbone", "mobilenet_v3_small"), n_classes=cfg.get("n_classes", 4),
            pretrained=False, imagenet_norm=cfg.get("imagenet_norm", True), dropout=cfg.get("dropout", 0.3))
    elif info["kind"] == "panns":
        pann = mods["pann"]
        model = pann.Transfer_Cnn6(classes_num=cfg.get("classes_num", 4), freeze_base=False)
    else:
        raise ValueError(info["kind"])
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval()


def make_input(kind_input, device):
    """Synthetic input: a [0, 1] mel spectrogram or a [-1, 1] waveform."""
    import torch
    if kind_input == "mel":
        x = torch.randn(MEL_SHAPE, device=device) * 0.3 + 0.5
        return torch.clamp(x, 0.0, 1.0)
    x = torch.randn(WAVE_SHAPE, device=device) * 0.1     # waveform RMS ~0.1 (Script 03 normalization)
    return torch.clamp(x, -1.0, 1.0)


def rss_mb():
    """Current process RSS in MB (psutil if available, else ru_maxrss)."""
    if psutil is not None:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    if resource is not None:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024.0 if platform.system() == "Linux" else r / (1024 ** 2)
    return float("nan")


def peak_rss_mb():
    """Peak process RSS in MB (ru_maxrss on Linux)."""
    if resource is not None and platform.system() == "Linux":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return rss_mb()


def disk_size_mb(path):
    return path.stat().st_size / (1024 ** 2) if path and path.exists() else float("nan")


def count_macs(model, key, input_shape, device, s15):
    """MACs via ptflops if available (consistent with Script 19), else thop from Script 15."""
    import torch
    if key in MACS_REFERENCE and MODELS[key]["kind"] in ("cfc", "torchscript"):
        return {"macs": MACS_REFERENCE[key], "source": "Script 19 (reference)"}
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(model, tuple(input_shape[1:]), as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
        return {"macs": float(macs), "source": "ptflops"}
    except Exception as e:
        logger.info(f"    ptflops unavailable/not applicable ({e}); trying thop from Script 15")
    try:
        r = s15.estimate_macs_flops(model, input_shape, device)
        macs = r.get("macs") if isinstance(r, dict) else None
        if macs:
            return {"macs": float(macs), "source": "thop (Script 15)"}
    except Exception as e:
        logger.info(f"    thop unavailable ({e})")
    if key in MACS_REFERENCE:
        return {"macs": MACS_REFERENCE[key], "source": "Script 19 (reference)"}
    return {"macs": float("nan"), "source": "not available"}


def benchmark_preprocessing(num_iter, s06):
    """
    Per-cycle preprocessing latency on the device: Script 06 mel (librosa) of an 8 s WAV at
    16 kHz (mel models) and 16 -> 32 kHz resampling with librosa.resample, as in
    raw_audio_dataset (PANNs). Uses a synthetic WAV with RMS 0.1 (Script 03 normalization).
    """
    import soundfile as sf
    import librosa
    rng = np.random.default_rng(0)
    audio = (rng.normal(0, 0.1, int(CYCLE_SECONDS * SR_MEL))).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "cycle.wav"
        sf.write(wav, audio, SR_MEL, subtype="PCM_16")
        cfg = s06.FEATURE_CONFIG
        for _ in range(3):
            s06.compute_mel_spectrogram(wav, cfg)
        t_mel = []
        for _ in range(num_iter):
            t0 = time.perf_counter(); s06.compute_mel_spectrogram(wav, cfg); t_mel.append((time.perf_counter() - t0) * 1e3)
    for _ in range(3):
        librosa.resample(audio, orig_sr=SR_MEL, target_sr=SR_PANNS)
    t_rs = []
    for _ in range(num_iter):
        t0 = time.perf_counter(); librosa.resample(audio, orig_sr=SR_MEL, target_sr=SR_PANNS); t_rs.append((time.perf_counter() - t0) * 1e3)
    out = {}
    for name, t in [("mel_16k_librosa", t_mel), ("resample_16k_to_32k", t_rs)]:
        t = np.array(t)
        out[name] = {"num_iter": num_iter, "p50_ms": float(np.percentile(t, 50)),
                     "p95_ms": float(np.percentile(t, 95)), "mean_ms": float(t.mean()), "std_ms": float(t.std())}
        logger.info(f"  preprocessing {name:<22} p50 {out[name]['p50_ms']:.2f} ms  p95 {out[name]['p95_ms']:.2f} ms")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Script 32: extended edge profiling (Pi 5)")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=list(MODELS.keys()))
    p.add_argument("--num_threads_list", type=int, nargs="+", default=[1, 4])
    p.add_argument("--num_iter", type=int, default=200)
    p.add_argument("--num_warmup", type=int, default=20)
    p.add_argument("--stress_test_minutes", type=float, default=0.0)
    p.add_argument("--temp_sample_interval_seconds", type=float, default=5.0)
    p.add_argument("--cooldown_target_temp", type=float, default=0.0)
    p.add_argument("--cooldown_max_seconds", type=int, default=300)
    p.add_argument("--skip_preproc", action="store_true")
    p.add_argument("--no_temperature", action="store_true")
    p.add_argument("--device_label", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    log_file = setup_logging()
    import torch
    logger.info("=" * 78); logger.info("SCRIPT 32: extended edge profiling"); logger.info("=" * 78)
    logger.info(f"BASE_DIR: {BASE_DIR}   torch {torch.__version__}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    s15 = import_script("script15", "15_edge_benchmark.py")
    mods = {}
    need_mnv3 = any(MODELS[m]["kind"] == "mnv3" for m in args.models)
    need_panns = any(MODELS[m]["kind"] == "panns" for m in args.models)
    if need_mnv3:
        mods["s21"] = import_script("script21", "21_transfer_learning.py")
    if need_panns:
        mods["pann"] = import_script("panns_cnn6", "panns_cnn6.py")

    sysinfo = s15.system_info()
    sysinfo["torch_threads_available"] = torch.get_num_threads()
    label = args.device_label or sysinfo.get("device_label", platform.node())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"System: {sysinfo.get('platform', '')}  |  label {label}")

    results = {"timestamp": stamp, "system": sysinfo, "args": vars(args), "models": {},
               "macs_reference_note": "cnn/hybrid reference MACs = Script 19 (409M / 101M)."}

    if not args.skip_preproc:
        logger.info("\n[Preprocessing]")
        s06 = import_script("script06", "06_extract_mel_spectrograms.py")
        try:
            results["preprocessing"] = benchmark_preprocessing(min(args.num_iter, 100), s06)
        except Exception as e:
            logger.warning(f"  preprocessing not measured: {e}")

    rows = []
    for key in args.models:
        info = MODELS[key]
        ckpt, seed, val_icbhi = resolve_checkpoint(key, s15)
        if ckpt is None:
            logger.warning(f"\n[{key}] no checkpoint in {CHECKPOINTS_DIR}; skipped"); continue
        logger.info(f"\n[{key}] {info['display']}  ←  {ckpt.name}" + (f" (seed {seed}, val ICBHI {val_icbhi:.3f})" if seed is not None else ""))
        rss0 = rss_mb()
        try:
            model = load_model(key, ckpt, device, s15, mods)
        except Exception as e:
            logger.error(f"  could not load: {e}"); continue
        rss_loaded = rss_mb()
        n_params = int(sum(p.numel() for p in model.parameters())) if hasattr(model, "parameters") else None
        mem_mb = s15.model_memory_mb(model) if info["kind"] != "torchscript" else disk_size_mb(ckpt)
        shape = MEL_SHAPE if info["input"] == "mel" else WAVE_SHAPE
        macs = count_macs(model, key, shape, device, s15) if info["kind"] != "torchscript" else \
            {"macs": MACS_REFERENCE.get(key, float("nan")), "source": "Script 19 (reference)"}
        entry = {"display": info["display"], "checkpoint": ckpt.name, "seed": seed, "val_icbhi": val_icbhi,
                 "n_params": n_params, "model_memory_mb": mem_mb, "disk_size_mb": disk_size_mb(ckpt),
                 "input_shape": list(shape), "macs": macs["macs"], "macs_source": macs["source"],
                 "rss_before_load_mb": rss0, "rss_after_load_mb": rss_loaded, "by_threads": {}}
        logger.info(f"  params {n_params if n_params else 'n/a'}  mem {mem_mb:.2f} MB  disk {entry['disk_size_mb']:.2f} MB  "
                    f"MACs {macs['macs']/1e6 if macs['macs']==macs['macs'] else float('nan'):.0f}M ({macs['source']})  "
                    f"RSS +{rss_loaded - rss0:.1f} MB on load")

        for nt in args.num_threads_list:
            torch.set_num_threads(nt)
            if args.cooldown_target_temp > 0 and not args.no_temperature:
                s15.wait_for_cooldown(args.cooldown_target_temp, args.cooldown_max_seconds,
                                      label=f"{key}/{nt}h", verbose=False)
            x = make_input(info["input"], device)
            # Sanity probe: one real inference with the correct input shape
            with torch.no_grad():
                y = model(x)
                if isinstance(y, dict):
                    y = y.get("clipwise_output", next(iter(y.values())))
            rss_pre = peak_rss_mb()
            bench = s15.benchmark_inference(model, shape, device, num_warmup=args.num_warmup,
                                            num_iter=args.num_iter, verbose=False,
                                            measure_temperature=not args.no_temperature)
            bench.pop("raw_latencies_ms", None)
            bench["peak_rss_mb"] = peak_rss_mb()
            bench["rss_delta_inference_mb"] = bench["peak_rss_mb"] - rss_pre
            try:
                bench["energy"] = s15.estimate_energy_analytical_a76(bench["latency_ms_p50"], nt, sysinfo)
            except Exception as e:
                bench["energy"] = {"error": str(e)}
            if args.stress_test_minutes > 0:
                bench["thermal_stress"] = s15.run_thermal_stress_test(
                    model, shape, device, args.stress_test_minutes, args.temp_sample_interval_seconds,
                    nt, info["display"], verbose=False)
                bench["thermal_stress"].pop("temperature_samples", None)
            entry["by_threads"][str(nt)] = bench
            e_mj = bench["energy"].get("estimated_energy_per_inference_mj") if isinstance(bench["energy"], dict) else None
            logger.info(f"  threads={nt}: p50 {bench['latency_ms_p50']:.2f} ms  p95 {bench['latency_ms_p95']:.2f}  "
                        f"p99 {bench['latency_ms_p99']:.2f}  peak RSS {bench['peak_rss_mb']:.0f} MB"
                        + (f"  ~{e_mj:.2f} mJ/inf" if isinstance(e_mj, (int, float)) else "")
                        + (f"  ΔT {bench['temp_delta_c']:+.1f} °C" if bench.get("temp_delta_c") is not None else ""))
            rows.append({"model": key, "display": info["display"], "threads": nt, "n_params": n_params,
                         "model_memory_mb": mem_mb, "disk_size_mb": entry["disk_size_mb"],
                         "macs_M": macs["macs"] / 1e6 if macs["macs"] == macs["macs"] else float("nan"),
                         "p50_ms": bench["latency_ms_p50"], "p95_ms": bench["latency_ms_p95"],
                         "p99_ms": bench["latency_ms_p99"], "mean_ms": bench["latency_ms_mean"],
                         "std_ms": bench["latency_ms_std"], "peak_rss_mb": bench["peak_rss_mb"],
                         "energy_mj": e_mj, "temp_delta_c": bench.get("temp_delta_c")})
        results["models"][key] = entry
        del model

    # Outputs: JSON (everything), CSV (one row per model and thread count), Markdown tables
    import pandas as pd
    base = OUT_DIR / f"edge_extended_{label}_{stamp}"
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    df = pd.DataFrame(rows)
    df.to_csv(base.with_suffix(".csv"), index=False)
    L = [f"# Extended edge profiling (Script 32): {label}\n", f"_{stamp} · iter {args.num_iter} / warmup {args.num_warmup}_\n"]
    if "preprocessing" in results:
        pp = results["preprocessing"]
        L += ["\n## Per-cycle preprocessing (8 s)\n", "| Stage | p50 ms | p95 ms |", "|---|---|---|"]
        for k, v in pp.items():
            L.append(f"| {k} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} |")
    for nt in args.num_threads_list:
        sub = df[df["threads"] == nt]
        if sub.empty:
            continue
        L += [f"\n## Batch-1 inference, {nt} thread(s)\n",
              "| Model | Params | Mem (MB) | MACs (M) | p50 ms | p95 ms | p99 ms | Peak RSS (MB) | mJ/inf | E2E p50 ms |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            pre = None
            if "preprocessing" in results:
                pre = results["preprocessing"]["mel_16k_librosa"]["p50_ms"] if MODELS[r["model"]]["input"] == "mel" \
                    else results["preprocessing"]["resample_16k_to_32k"]["p50_ms"]
            e2e = f"{r['p50_ms'] + pre:.2f}" if pre is not None else "n/a"
            L.append(f"| {r['display']} | {r['n_params']:,} | {r['model_memory_mb']:.2f} | {r['macs_M']:.0f} "
                     f"| {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['peak_rss_mb']:.0f} "
                     f"| {r['energy_mj'] if r['energy_mj'] == r['energy_mj'] and r['energy_mj'] is not None else 'n/a'} | {e2e} |"
                     if r["n_params"] == r["n_params"] and r["n_params"] is not None else
                     f"| {r['display']} | n/a | {r['model_memory_mb']:.2f} | {r['macs_M']:.0f} "
                     f"| {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['peak_rss_mb']:.0f} "
                     f"| {r['energy_mj'] if r['energy_mj'] == r['energy_mj'] and r['energy_mj'] is not None else 'n/a'} | {e2e} |")
    with open(base.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    logger.info(f"\nOutputs: {base}.json / .csv / .md   Log: {log_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
