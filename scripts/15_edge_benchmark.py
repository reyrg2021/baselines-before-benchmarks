#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 15: Edge inference benchmark of the final models on the Raspberry Pi 5.

Characterizes each model for edge deployment: batch-1 latency, model memory,
MACs/FLOPs, thread scaling, an analytical energy estimate for the Cortex-A76, and
thermal sustainability under two regimes. The continuous stress test is the
worst case (back-to-back inference, hardware thermal ceiling); the duty-cycle
test runs one inference every N seconds, which represents clinical use (spaced
auscultations). An optional temperature-targeted cooldown before every model
makes all models start from the same thermal baseline, so that their latencies
and temperature deltas are comparable and not confounded by run order.

Inputs:  models/checkpoints/ (cnn_official_seed*.pth, transposed_official_seed*.pth,
         hybrid_official_seed*.pth, cnn_int8_scripted.pt),
         scripts/09_cnn_training.py, scripts/11a_cfc_mel_v7.py (model classes)
Outputs: results/edge_benchmark/edge_benchmark_<label>_<timestamp>.{json,md,log},
         thermal_stress_*.{csv,png}, duty_cycle_*.{csv,png}
Usage:
    python3 15_edge_benchmark.py --num_threads_list 1 4 --num_iter 100 \\
        --cooldown_target_temp 55 --cooldown_max_seconds 300 \\
        --stress_test_minutes 10 --duty_cycle_minutes 20 \\
        --duty_cycle_interval_seconds 30 --temp_sample_interval_seconds 5 \\
        --device_label "RPi5"

Without an active cooler the Pi 5 throttles (2400 -> 1500 MHz, soft limit 80 C,
hard limit ~85 C) under any sustained load; with the official Active Cooler the
continuous stress test does not throttle.
"""

import os
import sys
import json
import time
import argparse
import importlib.util
import subprocess
import threading
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import platform

import numpy as np
import torch
import torch.nn as nn


# Optional dependencies: thop (MACs/FLOPs) and matplotlib (figures).
try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend (no display on the Pi)
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Path detection: works from the repository layout or a flat layout on the Pi.
SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent

if (SCRIPTS_DIR.parent / "models").exists():
    BASE_DIR = SCRIPTS_DIR.parent
elif (SCRIPTS_DIR / "models").exists():
    BASE_DIR = SCRIPTS_DIR
else:
    # Flat layout: look for checkpoints in the current working directory
    if (Path.cwd() / "models" / "checkpoints").exists():
        BASE_DIR = Path.cwd()
    else:
        BASE_DIR = SCRIPTS_DIR

CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results" / "edge_benchmark"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


MODELS = {
    "cnn": {
        "display": "CNN-RespiratoryNet (FP32)",
        "script": "09_cnn_training.py",
        "class_name": "RespiratoryNet",
        "checkpoint_prefix": "cnn_official",
    },
    "cnn_int8": {
        "display": "CNN-RespiratoryNet (INT8)",
        # Quantized model: no class is instantiated, it is loaded with torch.jit.load.
        "is_torchscript": True,
        "torchscript_file": "cnn_int8_scripted.pt",
    },
    "transposed": {
        "display": "CfC-Transposed",
        "script": "11a_cfc_mel_v7.py",
        "class_name": "CfCMelTransposedV6",
        "checkpoint_prefix": "transposed_official",
    },
    "hybrid": {
        "display": "CfC-Hybrid",
        "script": "11a_cfc_mel_v7.py",
        "class_name": "CfCMelHybridV6",
        "checkpoint_prefix": "hybrid_official",
    },
}


def import_module_from_path(name: str, path: Path):
    """Import a Python script as a module."""
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)

    # 11a_cfc_mel_v7.py parses arguments at import time, so it needs a valid sys.argv
    old_argv = sys.argv
    if "11a_cfc_mel_v7" in str(path):
        sys.argv = ["dummy", "--architecture", "hybrid"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv

    return module


def find_best_checkpoint(prefix: str) -> tuple:
    """
    Find the checkpoint with the highest val_icbhi for a model prefix.
    Returns (path, seed, val_icbhi), or (None, None, None) if none exists.
    """
    candidates = list(CHECKPOINTS_DIR.glob(f"{prefix}_seed*.pth"))
    if not candidates:
        return None, None, None

    best_path, best_seed, best_icbhi = None, None, -1.0
    for path in candidates:
        try:
            ck = torch.load(str(path), map_location="cpu", weights_only=False)
            val_icbhi = ck.get("val_icbhi", -1.0)
            if val_icbhi is None:
                val_icbhi = -1.0
            # File name pattern: prefix_seed{N}.pth
            stem = path.stem
            seed_str = stem.split("seed")[-1]
            seed = int(seed_str)
            if val_icbhi > best_icbhi:
                best_path, best_seed, best_icbhi = path, seed, val_icbhi
        except Exception as e:
            print(f"  [WARN] Could not read {path.name}: {e}")
            continue

    # No checkpoint stored val_icbhi: still return a valid one
    if best_path is None and candidates:
        best_path = sorted(candidates)[0]
        try:
            best_seed = int(best_path.stem.split("seed")[-1])
        except Exception:
            best_seed = -1
        best_icbhi = -1.0

    return best_path, best_seed, best_icbhi


def resolve_model_path(model_key: str) -> tuple:
    """
    Resolve a model's artifact path: the TorchScript file for INT8 (no seed),
    otherwise the best checkpoint by prefix. Returns (path, seed, val_icbhi).
    """
    info = MODELS[model_key]
    if info.get("is_torchscript"):
        p = CHECKPOINTS_DIR / info["torchscript_file"]
        return (p if p.exists() else None), None, None
    return find_best_checkpoint(info["checkpoint_prefix"])


def load_model_for_inference(model_key: str, checkpoint_path: Path,
                              device: torch.device) -> nn.Module:
    """Load a model from its checkpoint, ready for CPU inference."""
    info = MODELS[model_key]

    # Quantized model (TorchScript)
    if info.get("is_torchscript"):
        # On ARM the quantization backend is qnnpack; setting it avoids errors
        # when running INT8 kernels (fbgemm/onednn on x86).
        try:
            engines = torch.backends.quantized.supported_engines
            if "qnnpack" in engines:
                torch.backends.quantized.engine = "qnnpack"
        except Exception:
            pass
        model = torch.jit.load(str(checkpoint_path), map_location=device)
        model.eval()
        return model

    # FP32 models (class + state_dict)
    script_path = SCRIPTS_DIR / info["script"]
    module = import_module_from_path(f"mod_{model_key}", script_path)
    ModelClass = getattr(module, info["class_name"])

    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = ck.get("config", {})

    # RespiratoryNet takes a config dict; the CfC models take n_classes + config
    try:
        if model_key == "cnn":
            if not config and hasattr(module, "TRAINING_CONFIG"):
                config = module.TRAINING_CONFIG
            model = ModelClass(config=config) if "config" in ModelClass.__init__.__code__.co_varnames \
                    else ModelClass()
        else:
            model = ModelClass(n_classes=4, config=config)
    except TypeError as e:
        print(f"  [WARN] Standard constructor failed ({e}); retrying without config")
        model = ModelClass()

    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def model_memory_mb(model: nn.Module) -> float:
    """Model memory in MB (parameters + buffers)."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024 * 1024)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def read_soc_temperature() -> float:
    """
    SoC temperature in degrees Celsius, or None if unreadable.
    Tries `vcgencmd measure_temp` (Raspberry Pi), then
    /sys/class/thermal/thermal_zone0/temp (generic Linux, millidegrees).
    """
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            # Output: "temp=58.5'C\n"
            raw = result.stdout.strip()
            if raw.startswith("temp="):
                value = raw.replace("temp=", "").replace("'C", "").rstrip("C")
                return float(value)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, Exception):
        pass

    try:
        thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_path.exists():
            with open(thermal_path, "r") as f:
                return int(f.read().strip()) / 1000.0
    except Exception:
        pass

    return None


def read_soc_clock_mhz() -> float:
    """
    Current ARM CPU frequency in MHz (used to detect thermal throttling),
    or None if unreadable.
    """
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_clock", "arm"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            # Output: "frequency(48)=1500000000\n" (Hz)
            raw = result.stdout.strip()
            if "=" in raw:
                hz = int(raw.split("=")[-1])
                return hz / 1_000_000.0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, Exception):
        pass

    try:
        path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if path.exists():
            with open(path, "r") as f:
                khz = int(f.read().strip())
                return khz / 1000.0  # kHz -> MHz
    except Exception:
        pass

    return None


def temperature_snapshot(label: str = "") -> dict:
    """Temperature, clock and timestamp at the current instant."""
    return {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "temp_c": read_soc_temperature(),
        "clock_mhz": read_soc_clock_mhz(),
    }


class DualLogger:
    """Mirror stdout to a .txt file (start() / stop())."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._stdout = None

    def start(self):
        self._file = open(self.log_path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        sys.stdout = self  # type: ignore[assignment]

    def stop(self):
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._file is not None:
            self._file.close()
        self._file = None
        self._stdout = None

    def write(self, msg):
        if self._stdout is not None:
            self._stdout.write(msg)
        if self._file is not None:
            self._file.write(msg)

    def flush(self):
        if self._stdout is not None:
            self._stdout.flush()
        if self._file is not None:
            self._file.flush()


def estimate_macs_flops(model: nn.Module, input_shape: tuple,
                         device: torch.device) -> dict:
    """
    MACs and FLOPs of one inference, counted with thop (1 MAC = 2 FLOPs).

    This is an architectural count: it does not capture memory or scheduling
    overhead, nor non-MAC operations. Returns a dict with macs, flops,
    params_thop, method and notes; on failure, a dict with an error entry
    instead of raising.
    """
    if not HAS_THOP:
        return {
            "available": False,
            "error": "thop not installed. Install with: pip install thop",
            "method": "n/a",
        }

    try:
        dummy = torch.randn(input_shape, device=device)
        model.eval()
        with torch.no_grad():
            macs, params = thop_profile(model, inputs=(dummy,), verbose=False)
        return {
            "available": True,
            "macs": int(macs),
            "flops": int(macs * 2),
            "params_thop": int(params),
            "method": "thop_profile",
            "notes": ("MACs count multiply-accumulate ops. "
                      "FLOPs = 2 × MACs. Does not capture memory overhead, "
                      "scheduling, or non-MAC operations (activations, etc.)."),
        }
    except Exception as e:
        return {
            "available": False,
            "error": f"thop_profile failed: {e}",
            "method": "n/a",
        }


# Power figures for the analytical energy model, derived from the ARM Cortex-A76
# reference documentation and published measurements. Order-of-magnitude values,
# not manufacturer specifications.
A76_POWER_BASE_W = 0.40       # SoC uncore (DRAM controller, L3 cache, etc.)
A76_POWER_PER_ACTIVE_CORE_W = 0.75  # one core at 2.4 GHz with NEON SIMD active


def estimate_energy_analytical_a76(latency_ms_p50: float,
                                     num_threads: int,
                                     sysinfo: dict) -> dict:
    """
    Analytical energy per inference for the Pi 5 (Cortex-A76).

    Model: P_avg = P_base + min(num_threads, N_physical) * P_per_active_core,
    E_per_inf = P_avg * latency. It assumes equal consumption per active thread,
    ignores dynamic clock gating and throttling, and assumes FP32. It applies only
    to the Cortex-A76 (x86 power figures are entirely different). The returned
    dict carries an explicit disclaimer.
    """
    is_a76 = bool(sysinfo.get("is_raspberry_pi"))
    n_physical_cores = 4  # Pi 5: 4 Cortex-A76 cores, no SMT

    if num_threads <= 0:
        effective_threads = n_physical_cores
    else:
        effective_threads = min(num_threads, n_physical_cores)

    power_avg_w = (A76_POWER_BASE_W +
                   effective_threads * A76_POWER_PER_ACTIVE_CORE_W)
    latency_s = latency_ms_p50 / 1000.0
    energy_per_inf_mj = power_avg_w * latency_s * 1000.0  # W*s = J -> mJ
    energy_per_inf_j = power_avg_w * latency_s

    return {
        "applicable_to_this_hardware": is_a76,
        "method": "analytical_arm_cortex_a76_v1",
        "power_base_w": A76_POWER_BASE_W,
        "power_per_core_w": A76_POWER_PER_ACTIVE_CORE_W,
        "effective_threads_assumed": effective_threads,
        "estimated_avg_power_w": power_avg_w,
        "estimated_energy_per_inference_mj": energy_per_inf_mj,
        "estimated_energy_per_inference_j": energy_per_inf_j,
        "disclaimer": (
            "ANALYTICAL ESTIMATE, not a direct measurement. "
            "Derived from ARM Cortex-A76 datasheet × measured latency. "
            "Expected error margin: ±30%. "
            "An exact measurement requires a USB-C power meter or INA219. "
            "Valid only for Raspberry Pi 5 (Cortex-A76, FP32, NEON)."
        ),
    }


def run_thermal_stress_test(model: nn.Module,
                              input_shape: tuple,
                              device: torch.device,
                              duration_minutes: float,
                              sample_interval_seconds: float,
                              num_threads_label: int,
                              model_name: str,
                              verbose: bool = True) -> dict:
    """
    Continuous inference for N minutes while a background thread samples the
    SoC temperature and clock (worst-case thermal regime).

    On the same SoC, with fixed thermal mass and the same duration, the
    temperature rise grows with the dissipated electrical energy, so comparing
    delta T across configurations is an empirical proxy for relative energy use
    without a power meter. Throttling is flagged when the clock falls below 90%
    of the nominal 2400 MHz. The sampler is cheap (<1 ms per read) and does not
    interfere with the inference loop.

    Returns the temperature series, initial/final/max temperature and delta,
    minimum clock, throttling flag, and throughput during the test.
    """
    duration_s = duration_minutes * 60.0
    samples = []
    stop_event = threading.Event()
    sample_lock = threading.Lock()

    NOMINAL_CLOCK_MHZ = 2400.0
    THROTTLE_THRESHOLD = NOMINAL_CLOCK_MHZ * 0.90

    def sampler():
        """Background sampler of temperature and clock."""
        t0 = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - t0
            snap = temperature_snapshot(f"stress_t={elapsed:.1f}s")
            with sample_lock:
                samples.append({
                    "elapsed_s": elapsed,
                    "temp_c": snap["temp_c"],
                    "clock_mhz": snap["clock_mhz"],
                    "timestamp": snap["timestamp"],
                })
            stop_event.wait(sample_interval_seconds)

    dummy_input = torch.randn(input_shape, device=device) * 0.3 + 0.5
    dummy_input = torch.clamp(dummy_input, 0.0, 1.0)

    # Short warmup so that the initial load is not counted as part of the test
    if verbose:
        print(f"    Pre-stress warmup (5 iters)...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    snap_initial = temperature_snapshot("stress_start")

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    if verbose:
        print(f"    Starting stress test: {duration_minutes:.1f} min "
              f"({duration_s:.0f}s), sampling every {sample_interval_seconds}s...")
    n_inferences = 0
    t_loop_start = time.time()
    with torch.no_grad():
        while (time.time() - t_loop_start) < duration_s:
            _ = model(dummy_input)
            n_inferences += 1
    t_loop_end = time.time()
    actual_duration_s = t_loop_end - t_loop_start

    stop_event.set()
    sampler_thread.join(timeout=sample_interval_seconds + 2)

    snap_final = temperature_snapshot("stress_end")

    temps = [s["temp_c"] for s in samples if s["temp_c"] is not None]
    clocks = [s["clock_mhz"] for s in samples if s["clock_mhz"] is not None]

    temp_initial_c = (snap_initial["temp_c"]
                      if snap_initial and snap_initial["temp_c"] is not None
                      else (temps[0] if temps else None))
    temp_final_c = (snap_final["temp_c"]
                    if snap_final and snap_final["temp_c"] is not None
                    else (temps[-1] if temps else None))
    temp_max_c = max(temps) if temps else None
    temp_delta_c = ((temp_final_c - temp_initial_c)
                    if temp_initial_c is not None and temp_final_c is not None
                    else None)

    clock_min_mhz = min(clocks) if clocks else None
    clock_throttled = (clock_min_mhz < THROTTLE_THRESHOLD
                       if clock_min_mhz is not None else None)

    result = {
        "model_name": model_name,
        "num_threads": num_threads_label,
        "duration_requested_s": duration_s,
        "duration_actual_s": actual_duration_s,
        "n_inferences_completed": n_inferences,
        "throughput_during_stress_inf_per_sec": (
            n_inferences / actual_duration_s if actual_duration_s > 0 else None
        ),
        "sample_interval_seconds": sample_interval_seconds,
        "n_samples": len(samples),
        "temperature_samples": samples,
        "temp_initial_c": temp_initial_c,
        "temp_final_c": temp_final_c,
        "temp_max_c": temp_max_c,
        "temp_delta_c": temp_delta_c,
        "clock_min_mhz": clock_min_mhz,
        "clock_throttled": clock_throttled,
        "throttle_threshold_mhz": THROTTLE_THRESHOLD,
    }

    if verbose:
        print(f"    Stress test finished:")
        print(f"      Inferences completed: {n_inferences:,}")
        print(f"      Wall time: {actual_duration_s:.1f}s")
        if temp_initial_c is not None and temp_final_c is not None:
            print(f"      SoC temp: {temp_initial_c:.1f}°C -> "
                  f"{temp_final_c:.1f}°C  "
                  f"(Δ {temp_delta_c:+.1f}°C, max {temp_max_c:.1f}°C)")
        if clock_min_mhz is not None:
            ok = "THROTTLING" if clock_throttled else "no throttling"
            print(f"      Min CPU clock: {clock_min_mhz:.0f} MHz  ({ok})")

    return result


def save_thermal_stress_csv(stress_result: dict, csv_path: Path):
    """Save the stress-test time series as CSV."""
    samples = stress_result.get("temperature_samples", [])
    if not samples:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["elapsed_s", "temp_c", "clock_mhz", "timestamp"]
        )
        writer.writeheader()
        for s in samples:
            writer.writerow(s)


def plot_thermal_stress(stress_result: dict, png_path: Path,
                         model_display: str):
    """
    Plot temperature vs time (clock on a secondary axis) for a stress test.
    Returns False if matplotlib or valid samples are unavailable.
    """
    if not HAS_MATPLOTLIB:
        return False

    samples = stress_result.get("temperature_samples", [])
    if not samples:
        return False

    valid_temp = [s for s in samples if s["temp_c"] is not None]
    valid_clock = [s for s in samples if s["clock_mhz"] is not None]
    if not valid_temp:
        return False

    times_t = [s["elapsed_s"] for s in valid_temp]
    temps = [s["temp_c"] for s in valid_temp]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(times_t, temps, "r-", linewidth=2,
             label="SoC Temperature (°C)")
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Temperature (°C)", color="r", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="r")
    ax1.grid(True, alpha=0.3)

    # Typical Pi 5 soft throttling threshold
    ax1.axhline(y=80, color="r", linestyle=":", alpha=0.5,
                label="Throttle threshold (80°C)")

    if valid_clock:
        times_c = [s["elapsed_s"] for s in valid_clock]
        clocks = [s["clock_mhz"] for s in valid_clock]
        ax2 = ax1.twinx()
        ax2.plot(times_c, clocks, "b--", linewidth=1.5, alpha=0.7,
                 label="CPU Clock (MHz)")
        ax2.set_ylabel("Clock (MHz)", color="b", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="b")
        ax2.axhline(y=stress_result.get("throttle_threshold_mhz", 2160),
                    color="b", linestyle=":", alpha=0.5)

    threads = stress_result.get("num_threads", "?")
    duration_min = stress_result.get("duration_actual_s", 0) / 60.0
    title = (f"Thermal Stress Test: {model_display} "
             f"({threads} threads, {duration_min:.1f} min)")
    plt.title(title, fontsize=12)

    throttled = stress_result.get("clock_throttled")
    delta = stress_result.get("temp_delta_c")
    annot_lines = []
    if delta is not None:
        annot_lines.append(f"ΔT = {delta:+.1f}°C")
    if throttled is not None:
        annot_lines.append(
            "Throttling: " + ("YES" if throttled else "NO")
        )
    if annot_lines:
        ax1.text(0.02, 0.95, "\n".join(annot_lines),
                 transform=ax1.transAxes, fontsize=10,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def wait_for_cooldown(target_temp_c: float,
                       max_wait_s: float,
                       poll_interval_s: float = 5.0,
                       label: str = "",
                       verbose: bool = True) -> dict:
    """
    Wait until the SoC cools to target_temp_c or max_wait_s elapses.

    A fixed wait does not guarantee a common starting point (60 s took the SoC
    from 83 °C only to 60 °C); waiting for a target temperature makes every model
    start equally cold, so latencies and stress-test delta T are comparable.
    Without a temperature sensor it falls back to a fixed wait of max_wait_s.

    Returns reached_target, timed_out, temp_initial_c, temp_final_c,
    wait_actual_s, target_temp_c and method.
    """
    t_start = time.time()
    temp_initial = read_soc_temperature()

    # No sensor: fixed wait
    if temp_initial is None:
        if verbose:
            print(f"    [COOLDOWN{(' ' + label) if label else ''}] "
                  f"No temperature sensor: waiting a fixed {max_wait_s:.0f}s...")
        time.sleep(max_wait_s)
        return {
            "reached_target": None,
            "timed_out": None,
            "temp_initial_c": None,
            "temp_final_c": None,
            "wait_actual_s": time.time() - t_start,
            "target_temp_c": target_temp_c,
            "method": "fixed_fallback_no_sensor",
        }

    if temp_initial <= target_temp_c:
        if verbose:
            print(f"    [COOLDOWN{(' ' + label) if label else ''}] "
                  f"Current temp {temp_initial:.1f}°C already ≤ target "
                  f"{target_temp_c:.0f}°C, no wait.")
        return {
            "reached_target": True,
            "timed_out": False,
            "temp_initial_c": temp_initial,
            "temp_final_c": temp_initial,
            "wait_actual_s": 0.0,
            "target_temp_c": target_temp_c,
            "method": "adaptive",
        }

    if verbose:
        print(f"    [COOLDOWN{(' ' + label) if label else ''}] "
              f"Waiting for temp ≤ {target_temp_c:.0f}°C "
              f"(current {temp_initial:.1f}°C, timeout {max_wait_s:.0f}s)...")

    last_temp = temp_initial
    reached = False
    while (time.time() - t_start) < max_wait_s:
        time.sleep(poll_interval_s)
        current = read_soc_temperature()
        if current is None:
            continue
        last_temp = current
        if current <= target_temp_c:
            reached = True
            break

    wait_actual = time.time() - t_start
    if verbose:
        if reached:
            print(f"      -> Reached {last_temp:.1f}°C in {wait_actual:.0f}s")
        else:
            print(f"      -> TIMEOUT after {wait_actual:.0f}s "
                  f"(temp stayed at {last_temp:.1f}°C, "
                  f"did not reach {target_temp_c:.0f}°C)")

    return {
        "reached_target": reached,
        "timed_out": not reached,
        "temp_initial_c": temp_initial,
        "temp_final_c": last_temp,
        "wait_actual_s": wait_actual,
        "target_temp_c": target_temp_c,
        "method": "adaptive",
    }


def run_duty_cycle_test(model: nn.Module,
                         input_shape: tuple,
                         device: torch.device,
                         duration_minutes: float,
                         inference_interval_s: float,
                         num_threads_label: int,
                         model_name: str,
                         temp_sample_interval_seconds: float = 5.0,
                         verbose: bool = True) -> dict:
    """
    Simulate clinical use: one inference every `inference_interval_s` seconds for
    `duration_minutes`, with the CPU idle (and the SoC cooling) in between.

    Unlike the continuous stress test (worst case), this regime checks whether,
    in normal operation, temperature stays stable and latency stays constant
    (no throttling drift). In respiratory triage the operator auscultates a
    patient, runs one inference, and minutes pass before the next, so a 30 s
    interval is already conservative.

    Per inference it records latency and the SoC temperature and clock just
    before it; a background sampler records the continuous temperature series.
    Returns per-inference records, the temperature series, temperature summary,
    latency mean/std/first/last, latency_drift_pct (last vs first), minimum
    clock, throttling flag, and `stable` (temperature stayed below throttling).
    """
    duration_s = duration_minutes * 60.0
    NOMINAL_CLOCK_MHZ = 2400.0
    THROTTLE_THRESHOLD = NOMINAL_CLOCK_MHZ * 0.90

    samples = []
    per_inference = []
    stop_event = threading.Event()
    sample_lock = threading.Lock()

    def sampler():
        t0 = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - t0
            snap = temperature_snapshot(f"duty_t={elapsed:.1f}s")
            with sample_lock:
                samples.append({
                    "elapsed_s": elapsed,
                    "temp_c": snap["temp_c"],
                    "clock_mhz": snap["clock_mhz"],
                    "timestamp": snap["timestamp"],
                })
            stop_event.wait(temp_sample_interval_seconds)

    dummy_input = torch.randn(input_shape, device=device) * 0.3 + 0.5
    dummy_input = torch.clamp(dummy_input, 0.0, 1.0)

    # Minimal warmup (1 inference) to load caches without heating the SoC
    with torch.no_grad():
        _ = model(dummy_input)

    snap_initial = temperature_snapshot("duty_start")

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    if verbose:
        n_expected = int(duration_s / inference_interval_s)
        print(f"    Duty-cycle: 1 inf every {inference_interval_s:.0f}s "
              f"for {duration_minutes:.1f} min "
              f"(~{n_expected} inferences expected)...")

    t_loop_start = time.time()
    idx = 0
    with torch.no_grad():
        while (time.time() - t_loop_start) < duration_s:
            t_inf_scheduled = t_loop_start + idx * inference_interval_s

            # Sleep until the next scheduled slot
            sleep_until = t_inf_scheduled - time.time()
            if sleep_until > 0:
                time.sleep(sleep_until)

            temp_before = read_soc_temperature()
            clock_before = read_soc_clock_mhz()

            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            per_inference.append({
                "idx": idx,
                "elapsed_s": time.time() - t_loop_start,
                "latency_ms": latency_ms,
                "temp_c": temp_before,
                "clock_mhz": clock_before,
            })
            idx += 1

    t_loop_end = time.time()
    actual_duration_s = t_loop_end - t_loop_start

    stop_event.set()
    sampler_thread.join(timeout=temp_sample_interval_seconds + 2)

    snap_final = temperature_snapshot("duty_end")

    temps = [s["temp_c"] for s in samples if s["temp_c"] is not None]
    clocks = [s["clock_mhz"] for s in samples if s["clock_mhz"] is not None]

    temp_initial_c = (snap_initial["temp_c"]
                      if snap_initial and snap_initial["temp_c"] is not None
                      else (temps[0] if temps else None))
    temp_final_c = (snap_final["temp_c"]
                    if snap_final and snap_final["temp_c"] is not None
                    else (temps[-1] if temps else None))
    temp_max_c = max(temps) if temps else None
    temp_delta_c = ((temp_final_c - temp_initial_c)
                    if temp_initial_c is not None and temp_final_c is not None
                    else None)
    clock_min_mhz = min(clocks) if clocks else None
    clock_throttled = (clock_min_mhz < THROTTLE_THRESHOLD
                       if clock_min_mhz is not None else None)

    # Latency drift (degradation due to throttling)
    lats = [p["latency_ms"] for p in per_inference]
    lat_mean = float(np.mean(lats)) if lats else None
    lat_std = float(np.std(lats)) if lats else None
    lat_first = lats[0] if lats else None
    lat_last = lats[-1] if lats else None
    lat_drift_pct = (((lat_last - lat_first) / lat_first * 100.0)
                     if lat_first and lat_last else None)

    # "Stable" = never reached the throttling threshold
    stable = (clock_throttled is False) if clock_throttled is not None else None

    result = {
        "model_name": model_name,
        "num_threads": num_threads_label,
        "mode": "duty_cycle",
        "inference_interval_s": inference_interval_s,
        "duration_requested_s": duration_s,
        "duration_actual_s": actual_duration_s,
        "n_inferences": len(per_inference),
        "per_inference": per_inference,
        "temperature_samples": samples,
        "n_samples": len(samples),
        "temp_initial_c": temp_initial_c,
        "temp_final_c": temp_final_c,
        "temp_max_c": temp_max_c,
        "temp_delta_c": temp_delta_c,
        "clock_min_mhz": clock_min_mhz,
        "clock_throttled": clock_throttled,
        "throttle_threshold_mhz": THROTTLE_THRESHOLD,
        "latency_ms_mean": lat_mean,
        "latency_ms_std": lat_std,
        "latency_ms_first": lat_first,
        "latency_ms_last": lat_last,
        "latency_drift_pct": lat_drift_pct,
        "stable": stable,
    }

    if verbose:
        print(f"    Duty-cycle finished:")
        print(f"      Inferences: {len(per_inference)}")
        if temp_initial_c is not None and temp_final_c is not None:
            print(f"      SoC temp: {temp_initial_c:.1f}°C -> "
                  f"{temp_final_c:.1f}°C  "
                  f"(Δ {temp_delta_c:+.1f}°C, max {temp_max_c:.1f}°C)")
        if lat_mean is not None:
            print(f"      Latency: {lat_mean:.1f} ± {lat_std:.1f} ms  "
                  f"(first: {lat_first:.1f}, last: {lat_last:.1f})")
        if lat_drift_pct is not None:
            print(f"      Latency drift: {lat_drift_pct:+.1f}% "
                  f"(high values indicate throttling degradation)")
        if clock_throttled is not None:
            status_text = ("STABLE (no throttling)" if stable
                      else "throttling detected")
            print(f"      Min clock: {clock_min_mhz:.0f} MHz  -> {status_text}")

    return result


def plot_duty_cycle(duty_result: dict, png_path: Path, model_display: str):
    """
    Duty-cycle figure: temperature vs time (top) and per-inference latency
    (bottom), showing whether temperature stabilizes and latency stays constant.
    """
    if not HAS_MATPLOTLIB:
        return False

    samples = duty_result.get("temperature_samples", [])
    per_inf = duty_result.get("per_inference", [])
    valid_temp = [s for s in samples if s["temp_c"] is not None]
    if not valid_temp and not per_inf:
        return False

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if valid_temp:
        times_t = [s["elapsed_s"] for s in valid_temp]
        temps = [s["temp_c"] for s in valid_temp]
        ax1.plot(times_t, temps, "r-", linewidth=2,
                 label="SoC Temperature (°C)")
        ax1.axhline(y=80, color="r", linestyle=":", alpha=0.5,
                    label="Throttle threshold (80°C)")
        ax1.set_ylabel("Temperature (°C)", color="r", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="r")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=9)

    interval = duty_result.get("inference_interval_s", "?")
    duration_min = duty_result.get("duration_actual_s", 0) / 60.0
    threads = duty_result.get("num_threads", "?")
    ax1.set_title(f"Duty-Cycle Test: {model_display} "
                  f"({threads}t, 1 inf/{interval}s, {duration_min:.1f} min)",
                  fontsize=12)

    stable = duty_result.get("stable")
    delta = duty_result.get("temp_delta_c")
    annot = []
    if delta is not None:
        annot.append(f"ΔT = {delta:+.1f}°C")
    if stable is not None:
        annot.append("Stable: " + ("YES" if stable else "NO"))
    if annot:
        ax1.text(0.02, 0.95, "\n".join(annot), transform=ax1.transAxes,
                 fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    if per_inf:
        inf_times = [p["elapsed_s"] for p in per_inf]
        inf_lats = [p["latency_ms"] for p in per_inf]
        ax3.plot(inf_times, inf_lats, "g.-", linewidth=1, markersize=6,
                 label="Per-inference latency (ms)")
        ax3.set_xlabel("Time (s)", fontsize=11)
        ax3.set_ylabel("Latency (ms)", color="g", fontsize=11)
        ax3.tick_params(axis="y", labelcolor="g")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right", fontsize=9)
        drift = duty_result.get("latency_drift_pct")
        if drift is not None:
            ax3.text(0.02, 0.95, f"Drift: {drift:+.1f}%",
                     transform=ax3.transAxes, fontsize=10,
                     verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def save_duty_cycle_csv(duty_result: dict, csv_path: Path):
    """Save per-inference latency and temperature of the duty-cycle test."""
    per_inf = duty_result.get("per_inference", [])
    if not per_inf:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["idx", "elapsed_s", "latency_ms",
                           "temp_c", "clock_mhz"]
        )
        writer.writeheader()
        for p in per_inf:
            writer.writerow(p)


def benchmark_inference(model: nn.Module, input_shape: tuple,
                         device: torch.device,
                         num_warmup: int = 20,
                         num_iter: int = 100,
                         verbose: bool = True,
                         measure_temperature: bool = True) -> dict:
    """
    Batch-1 inference latency (the real edge case).

    Runs num_warmup forward passes, then times num_iter individual passes with
    time.perf_counter() (CPU execution is synchronous). Optional temperature and
    clock snapshots are taken before warmup and after measurement. Returns
    p50/p95/p99/min/max/mean/std latency (ms), throughput, raw latencies and
    thermal deltas.
    """
    # Fixed dummy input: normalized mel spectrogram in a realistic range
    dummy_input = torch.randn(input_shape, device=device) * 0.3 + 0.5
    dummy_input = torch.clamp(dummy_input, 0.0, 1.0)

    # Taken before warmup so the delta reflects the whole benchmark
    snap_before = temperature_snapshot("before_benchmark") if measure_temperature else None

    if verbose:
        print(f"    Warmup ({num_warmup} iters)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    if verbose:
        print(f"    Measuring ({num_iter} iters)...")
    latencies_ms = np.zeros(num_iter, dtype=np.float64)
    with torch.no_grad():
        for i in range(num_iter):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms[i] = (t1 - t0) * 1000.0

    snap_after = temperature_snapshot("after_benchmark") if measure_temperature else None

    temp_delta_c = None
    clock_delta_mhz = None
    if snap_before and snap_after:
        if snap_before["temp_c"] is not None and snap_after["temp_c"] is not None:
            temp_delta_c = snap_after["temp_c"] - snap_before["temp_c"]
        if snap_before["clock_mhz"] is not None and snap_after["clock_mhz"] is not None:
            clock_delta_mhz = snap_after["clock_mhz"] - snap_before["clock_mhz"]

    return {
        "num_iter": num_iter,
        "num_warmup": num_warmup,
        "input_shape": list(input_shape),
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "latency_ms_p99": float(np.percentile(latencies_ms, 99)),
        "latency_ms_min": float(np.min(latencies_ms)),
        "latency_ms_max": float(np.max(latencies_ms)),
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_std": float(np.std(latencies_ms)),
        "throughput_inf_per_sec": float(1000.0 / np.median(latencies_ms)),
        "raw_latencies_ms": latencies_ms.tolist(),
        "temp_snapshot_before": snap_before,
        "temp_snapshot_after": snap_after,
        "temp_delta_c": temp_delta_c,
        "clock_delta_mhz": clock_delta_mhz,
    }


def benchmark_throughput_batched(model: nn.Module, input_shape: tuple,
                                  device: torch.device,
                                  batch_sizes: list = [1, 4, 8],
                                  num_iter: int = 30) -> dict:
    """Throughput for several batch sizes (whether batching pays off)."""
    results = {}
    for bs in batch_sizes:
        bshape = (bs,) + tuple(input_shape[1:])
        dummy = torch.randn(bshape, device=device) * 0.3 + 0.5
        dummy = torch.clamp(dummy, 0.0, 1.0)

        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iter):
                _ = model(dummy)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        results[f"batch_{bs}"] = {
            "elapsed_s": elapsed,
            "samples_processed": bs * num_iter,
            "throughput_samples_per_sec": bs * num_iter / elapsed,
            "latency_per_batch_ms": (elapsed / num_iter) * 1000.0,
        }
    return results


def system_info() -> dict:
    """System information recorded for reproducibility."""
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),       # x86_64, aarch64, etc.
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count_logical": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        info["cuda_device"] = torch.cuda.get_device_name(0)

    # Raspberry Pi detection
    try:
        if Path("/proc/cpuinfo").exists():
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
                if "Raspberry Pi" in content or "BCM2" in content:
                    info["is_raspberry_pi"] = True
                    for line in content.split("\n"):
                        if "Model" in line and "Raspberry" in line:
                            info["rpi_model"] = line.split(":", 1)[-1].strip()
                            break
    except Exception:
        pass

    return info


def print_table(rows: list, headers: list, title: str = ""):
    """Print a simple ASCII table to the console."""
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    total_w = sum(widths) + 3 * len(widths) + 1

    if title:
        print(f"\n{title}")
        print("=" * total_w)
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    print("| " + " | ".join(f"{str(h):<{w}}" for h, w in zip(headers, widths)) + " |")
    print(sep)
    for row in rows:
        print("| " + " | ".join(f"{str(c):<{w}}" for c, w in zip(row, widths)) + " |")
    print(sep)


def write_markdown_report(all_results: dict, sysinfo: dict, output_path: Path,
                           device_label: str):
    """Write the single-thread-count Markdown report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Edge AI Inference Benchmark\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"## System\n\n")
        f.write(f"- Device label: **{device_label}**\n")
        f.write(f"- Platform: {sysinfo['platform']}\n")
        f.write(f"- Machine: {sysinfo['machine']}\n")
        f.write(f"- Processor: {sysinfo.get('processor', 'N/A')}\n")
        f.write(f"- CPU cores (logical): {sysinfo['cpu_count_logical']}\n")
        f.write(f"- torch.get_num_threads(): {sysinfo['torch_num_threads']}\n")
        f.write(f"- Python: {sysinfo['python_version']}\n")
        f.write(f"- PyTorch: {sysinfo['torch_version']}\n")
        if sysinfo.get("is_raspberry_pi"):
            f.write(f"- **Raspberry Pi:** {sysinfo.get('rpi_model', 'detected')}\n")
        f.write(f"\n")

        f.write(f"## Main table (batch=1, FP32, CPU)\n\n")
        f.write("| Model | Params | Memory | Latency p50 | p95 | p99 | Throughput |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for mk, r in all_results.items():
            if not r:
                continue
            f.write(f"| {MODELS[mk]['display']} | "
                    f"{r['params']:,} | "
                    f"{r['memory_mb']:.2f} MB | "
                    f"{r['benchmark']['latency_ms_p50']:.2f} ms | "
                    f"{r['benchmark']['latency_ms_p95']:.2f} ms | "
                    f"{r['benchmark']['latency_ms_p99']:.2f} ms | "
                    f"{r['benchmark']['throughput_inf_per_sec']:.1f}/s |\n")

        f.write(f"\n## Per-model detail\n\n")
        for mk, r in all_results.items():
            if not r:
                continue
            f.write(f"### {MODELS[mk]['display']}\n\n")
            _seed = r.get("seed")
            _val = r.get("val_icbhi")
            if _val is not None and _seed is not None:
                f.write(f"- Checkpoint: `{Path(r['checkpoint_path']).name}` "
                        f"(seed {_seed}, val_ICBHI={_val:.4f})\n")
            else:
                f.write(f"- Artifact: `{Path(r['checkpoint_path']).name}`\n")
            f.write(f"- Parameters: {r['params']:,}\n")
            f.write(f"- Model memory (FP32): {r['memory_mb']:.2f} MB\n")
            f.write(f"- Latency (ms): "
                    f"p50={r['benchmark']['latency_ms_p50']:.2f}, "
                    f"p95={r['benchmark']['latency_ms_p95']:.2f}, "
                    f"p99={r['benchmark']['latency_ms_p99']:.2f}, "
                    f"mean={r['benchmark']['latency_ms_mean']:.2f}, "
                    f"std={r['benchmark']['latency_ms_std']:.2f}\n")
            f.write(f"- Throughput (batch=1): {r['benchmark']['throughput_inf_per_sec']:.1f} inferences/s\n")
            if "throughput_batched" in r:
                f.write(f"- Throughput per batch size:\n")
                for bk, bv in r["throughput_batched"].items():
                    f.write(f"  - {bk}: {bv['throughput_samples_per_sec']:.1f} samples/s "
                            f"(latency per batch: {bv['latency_per_batch_ms']:.2f} ms)\n")
            f.write("\n")

        f.write(f"## Interpretation\n\n")
        if all_results.get("hybrid"):
            h = all_results["hybrid"]["benchmark"]
            p50_ms = h["latency_ms_p50"]
            tput = h["throughput_inf_per_sec"]
            real_time_factor = 8000 / p50_ms  # one respiratory cycle lasts ~8 s
            f.write(f"- **CfC-Hybrid** processes one inference of "
                    f"an 8 s respiratory cycle in **{p50_ms:.1f} ms** (median).\n")
            f.write(f"- This equals **{tput:.0f} inferences/second** "
                    f"or {real_time_factor:.0f}× real time.\n")
            f.write(f"- In clinical deployment (one cycle every ~4 s while "
                    f"a patient breathes), the model uses **<1% of the CPU** of "
                    f"the edge device.\n")
        f.write(f"\n")

    print(f"\nMarkdown report saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Edge inference benchmark of the final models "
                    "(supports multi-thread sweep and temperature readout)"
    )
    parser.add_argument("--models", nargs="+",
                        default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Subset of models to benchmark")
    parser.add_argument("--num_iter", type=int, default=100,
                        help="Number of measurement iterations")
    parser.add_argument("--num_warmup", type=int, default=20,
                        help="Warmup iterations")
    parser.add_argument("--num_threads", type=int, default=0,
                        help="Single thread count for "
                             "torch.set_num_threads() (0 = automatic). "
                             "Ignored if --num_threads_list is given.")
    parser.add_argument("--num_threads_list", type=int, nargs="+", default=None,
                        help="List of thread counts to sweep, "
                             "e.g. --num_threads_list 1 2 4. The script "
                             "runs every value and writes a model × threads "
                             "cross-product table.")
    parser.add_argument("--cooldown_seconds", type=int, default=0,
                        help="Fixed pause between thread configurations so "
                             "the SoC returns to baseline temperature "
                             "(Pi 5 without active cooling: 30-60 s). "
                             "Used only if --cooldown_target_temp is 0.")
    parser.add_argument("--device_label", type=str, default=None,
                        help="Device label for the report")
    parser.add_argument("--skip_batched", action="store_true",
                        help="Skip the batched throughput benchmark")
    parser.add_argument("--no_temperature", action="store_true",
                        help="Disable SoC temperature readout "
                             "(for hardware without an accessible sensor).")
    parser.add_argument("--skip_thop", action="store_true",
                        help="Skip MACs/FLOPs counting with thop "
                             "(useful if thop is not installed).")
    parser.add_argument("--stress_test_minutes", type=float, default=0.0,
                        help="Duration of the continuous thermal stress test "
                             "per configuration (model × threads). "
                             "0 = disabled (default). "
                             "Recommended: 10 min to detect throttling.")
    parser.add_argument("--temp_sample_interval_seconds", type=float,
                        default=5.0,
                        help="Temperature sampling period during long tests "
                             "(default 5 s).")
    parser.add_argument("--cooldown_target_temp", type=float, default=0.0,
                        help="Target temperature (°C) to which the SoC is "
                             "cooled BEFORE EACH MODEL. "
                             "0 = disabled (uses fixed --cooldown_seconds). "
                             "Recommended: 55. Gives a common thermal start "
                             "so delta T is comparable across models.")
    parser.add_argument("--cooldown_max_seconds", type=int, default=300,
                        help="Timeout of the adaptive cooldown "
                             "(default 300 s). If the SoC does not reach the "
                             "target in this time, the run continues.")
    parser.add_argument("--duty_cycle_minutes", type=float, default=0.0,
                        help="Duration of the duty-cycle test "
                             "(clinical use). 0 = disabled. "
                             "Recommended: 20 min.")
    parser.add_argument("--duty_cycle_interval_seconds", type=float,
                        default=30.0,
                        help="Seconds between inferences in the "
                             "duty-cycle test (default 30 s), simulating "
                             "spaced auscultations in respiratory triage.")
    return parser.parse_args()


def run_models_for_thread_count(
    args, models_keys, threads_value, device, input_shape, sysinfo,
):
    """
    Run all requested models for ONE num_threads value (the atomic unit of the
    multi-thread sweep). Returns {model_key: result_dict | None, ...}.
    """
    if threads_value > 0:
        torch.set_num_threads(threads_value)
        print(f"\n[CONFIG] torch.set_num_threads({threads_value}) "
              f"-> effective: {torch.get_num_threads()}")
    else:
        print(f"\n[CONFIG] num_threads = automatic "
              f"(current: {torch.get_num_threads()})")

    measure_temp = not args.no_temperature
    results_for_threads = {}
    cooldown_log = {}

    for model_idx, mk in enumerate(models_keys):
        info = MODELS[mk]

        # Adaptive cooldown before each model: common thermal baseline
        if args.cooldown_target_temp > 0 and not args.no_temperature:
            cd = wait_for_cooldown(
                target_temp_c=args.cooldown_target_temp,
                max_wait_s=args.cooldown_max_seconds,
                poll_interval_s=5.0,
                label=f"pre-{mk}",
                verbose=True,
            )
            cooldown_log[mk] = cd

        print(f"\n{'─' * 70}")
        print(f"[{info['display']}]  (threads={threads_value})")
        print(f"{'─' * 70}")

        ckpt_path, seed, val_icbhi = resolve_model_path(mk)
        if ckpt_path is None:
            print(f"  ✗ No checkpoint for {mk} in {CHECKPOINTS_DIR}")
            results_for_threads[mk] = None
            continue
        if val_icbhi is not None and val_icbhi >= 0:
            print(f"  Best checkpoint: {ckpt_path.name} "
                  f"(seed {seed}, val_ICBHI={val_icbhi:.4f})")
        else:
            print(f"  Artifact: {ckpt_path.name}")

        try:
            model = load_model_for_inference(mk, ckpt_path, device)
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            results_for_threads[mk] = None
            continue

        n_params = count_params(model)
        mem_mb = model_memory_mb(model)
        print(f"  Params:           {n_params:,}")
        print(f"  Memory (FP32):    {mem_mb:.2f} MB")

        macs_info = None
        if not args.skip_thop:
            if not HAS_THOP:
                print(f"  [WARN] thop not installed, skipping MAC count."
                      f"  Install with: pip install thop")
                macs_info = {"available": False,
                             "error": "thop not installed"}
            else:
                macs_info = estimate_macs_flops(model, input_shape, device)
                if macs_info.get("available"):
                    macs_m = macs_info["macs"] / 1e6
                    flops_m = macs_info["flops"] / 1e6
                    print(f"  MACs/inf:         {macs_m:.2f} M  "
                          f"({flops_m:.2f} MFLOPs)")
                else:
                    print(f"  [WARN] thop failed: "
                          f"{macs_info.get('error', 'unknown')}")

        # Forward-pass sanity check
        try:
            dummy = torch.randn(input_shape, device=device) * 0.3 + 0.5
            with torch.no_grad():
                out = model(dummy)
            print(f"  Forward OK: input {tuple(dummy.shape)} -> "
                  f"output {tuple(out.shape)}")
        except Exception as e:
            print(f"  ✗ Forward failed: {e}")
            results_for_threads[mk] = None
            continue

        print(f"  Latency batch=1:")
        bench = benchmark_inference(
            model, input_shape, device,
            num_warmup=args.num_warmup,
            num_iter=args.num_iter,
            verbose=True,
            measure_temperature=measure_temp,
        )
        print(f"    p50 = {bench['latency_ms_p50']:.2f} ms")
        print(f"    p95 = {bench['latency_ms_p95']:.2f} ms")
        print(f"    p99 = {bench['latency_ms_p99']:.2f} ms")
        print(f"    Throughput = {bench['throughput_inf_per_sec']:.1f} inf/s")
        if bench.get("temp_delta_c") is not None:
            t_b = bench["temp_snapshot_before"]["temp_c"]
            t_a = bench["temp_snapshot_after"]["temp_c"]
            print(f"    SoC temp: {t_b:.1f}°C -> {t_a:.1f}°C  "
                  f"(Δ {bench['temp_delta_c']:+.1f}°C)")
        if bench.get("clock_delta_mhz") is not None:
            c_b = bench["temp_snapshot_before"]["clock_mhz"]
            c_a = bench["temp_snapshot_after"]["clock_mhz"]
            print(f"    CPU clock: {c_b:.0f} -> {c_a:.0f} MHz  "
                  f"(Δ {bench['clock_delta_mhz']:+.0f} MHz)")

        # Analytical energy estimate (Pi 5 / Cortex-A76 only)
        energy_estimate = estimate_energy_analytical_a76(
            latency_ms_p50=bench["latency_ms_p50"],
            num_threads=threads_value,
            sysinfo=sysinfo,
        )
        if energy_estimate["applicable_to_this_hardware"]:
            print(f"  Estimated energy (Cortex-A76):")
            print(f"    Avg power:    "
                  f"{energy_estimate['estimated_avg_power_w']:.2f} W "
                  f"(estimate, not a measurement)")
            print(f"    Energy/inf:   "
                  f"{energy_estimate['estimated_energy_per_inference_mj']:.2f} mJ")
        else:
            print(f"  [INFO] A76 energy estimate does not apply to this "
                  f"hardware ({sysinfo.get('machine', 'unknown')}).")

        throughput_batched = None
        if not args.skip_batched:
            print(f"  Throughput per batch size:")
            throughput_batched = benchmark_throughput_batched(
                model, input_shape, device,
                batch_sizes=[1, 4, 8],
                num_iter=max(10, args.num_iter // 4),
            )
            for bk, bv in throughput_batched.items():
                print(f"    {bk}: {bv['throughput_samples_per_sec']:.1f} samples/s")

        # Continuous thermal stress test (optional, worst case)
        stress_result = None
        if args.stress_test_minutes > 0:
            print(f"\n  [STRESS TEST] {args.stress_test_minutes:.1f} min "
                  f"of continuous inference (worst case)...")
            stress_result = run_thermal_stress_test(
                model=model,
                input_shape=input_shape,
                device=device,
                duration_minutes=args.stress_test_minutes,
                sample_interval_seconds=args.temp_sample_interval_seconds,
                num_threads_label=threads_value,
                model_name=info["display"],
                verbose=True,
            )

        # Duty-cycle test (clinical use), run after the stress test with its
        # own prior cooldown when adaptive cooldown is enabled.
        duty_result = None
        if args.duty_cycle_minutes > 0:
            if args.cooldown_target_temp > 0 and not args.no_temperature:
                print(f"\n  [COOLDOWN pre-duty-cycle]")
                wait_for_cooldown(
                    target_temp_c=args.cooldown_target_temp,
                    max_wait_s=args.cooldown_max_seconds,
                    poll_interval_s=5.0,
                    label=f"pre-duty-{mk}",
                    verbose=True,
                )
            print(f"\n  [DUTY-CYCLE] {args.duty_cycle_minutes:.1f} min, "
                  f"1 inf every {args.duty_cycle_interval_seconds:.0f}s "
                  f"(clinical use)...")
            duty_result = run_duty_cycle_test(
                model=model,
                input_shape=input_shape,
                device=device,
                duration_minutes=args.duty_cycle_minutes,
                inference_interval_s=args.duty_cycle_interval_seconds,
                num_threads_label=threads_value,
                model_name=info["display"],
                temp_sample_interval_seconds=args.temp_sample_interval_seconds,
                verbose=True,
            )

        results_for_threads[mk] = {
            "checkpoint_path": str(ckpt_path),
            "seed": seed,
            "val_icbhi": val_icbhi,
            "params": n_params,
            "memory_mb": mem_mb,
            "macs_info": macs_info,
            "benchmark": bench,
            "energy_estimate_a76": energy_estimate,
            "throughput_batched": throughput_batched,
            "torch_num_threads_effective": torch.get_num_threads(),
            "cooldown_before": cooldown_log.get(mk),
            "stress_test": stress_result,
            "duty_cycle": duty_result,
        }

    return results_for_threads


def print_cross_product_table(sweep_results, models_keys):
    """
    Print the model × threads cross-product table (p50 latency) and, when
    1 thread is included, the speedup table.
    barrido_results: {threads_int: {model_key: result | None}}
    """
    threads_sorted = sorted(sweep_results.keys())

    print(f"\n{'=' * 78}")
    print(f"CROSS-PRODUCT TABLE: p50 latency (ms) per model × threads")
    print(f"{'=' * 78}")

    headers = ["Model"] + [f"{t}t" for t in threads_sorted]
    rows = []
    for mk in models_keys:
        row = [MODELS[mk]["display"]]
        for t in threads_sorted:
            r = sweep_results.get(t, {}).get(mk)
            if r is None:
                row.append("—")
            else:
                row.append(f"{r['benchmark']['latency_ms_p50']:.1f}")
        rows.append(row)
    print_table(rows=rows, headers=headers)

    if len(threads_sorted) >= 2 and 1 in threads_sorted:
        print(f"\n{'=' * 78}")
        print(f"SPEEDUP vs 1 thread")
        print(f"{'=' * 78}")
        sp_rows = []
        for mk in models_keys:
            row = [MODELS[mk]["display"]]
            r1 = sweep_results.get(1, {}).get(mk)
            base_lat = r1["benchmark"]["latency_ms_p50"] if r1 else None
            for t in threads_sorted:
                if t == 1:
                    row.append("1.00×")
                    continue
                r = sweep_results.get(t, {}).get(mk)
                if r is None or base_lat is None:
                    row.append("—")
                else:
                    sp = base_lat / r["benchmark"]["latency_ms_p50"]
                    row.append(f"{sp:.2f}×")
            sp_rows.append(row)
        print_table(rows=sp_rows, headers=headers)


def write_cross_product_markdown(sweep_results, models_keys, sysinfo,
                                  output_path, device_label, args):
    """Write the Markdown report of the multi-thread cross-product sweep."""
    threads_sorted = sorted(sweep_results.keys())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Edge AI Inference Benchmark: multi-thread sweep\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"## System\n\n")
        f.write(f"- Device label: **{device_label}**\n")
        f.write(f"- Platform: {sysinfo['platform']}\n")
        f.write(f"- Machine: {sysinfo['machine']}\n")
        f.write(f"- CPU cores (logical): {sysinfo['cpu_count_logical']}\n")
        f.write(f"- Python: {sysinfo['python_version']}\n")
        f.write(f"- PyTorch: {sysinfo['torch_version']}\n")
        if sysinfo.get("is_raspberry_pi"):
            f.write(f"- **Raspberry Pi:** "
                    f"{sysinfo.get('rpi_model', 'detected')}\n")
        f.write(f"\n")

        f.write(f"## Sweep configuration\n\n")
        f.write(f"- Thread counts tested: {threads_sorted}\n")
        f.write(f"- Iterations per point: {args.num_iter} "
                f"(warmup: {args.num_warmup})\n")
        f.write(f"- Cooldown between sweeps: {args.cooldown_seconds} s\n")
        f.write(f"- Temperature readout: "
                f"{'NO' if args.no_temperature else 'YES'}\n\n")

        f.write(f"## Main table: p50 latency (ms)\n\n")
        f.write("| Model |")
        for t in threads_sorted:
            f.write(f" {t} thread{'s' if t > 1 else ''} |")
        f.write("\n|---|")
        for _ in threads_sorted:
            f.write("---:|")
        f.write("\n")
        for mk in models_keys:
            f.write(f"| {MODELS[mk]['display']} |")
            for t in threads_sorted:
                r = sweep_results.get(t, {}).get(mk)
                if r is None:
                    f.write(" — |")
                else:
                    f.write(f" {r['benchmark']['latency_ms_p50']:.2f} |")
            f.write("\n")
        f.write(f"\n")

        if 1 in threads_sorted and len(threads_sorted) >= 2:
            f.write(f"## Speedup vs 1 thread (scalability)\n\n")
            f.write("| Model |")
            for t in threads_sorted:
                f.write(f" {t}t |")
            f.write("\n|---|")
            for _ in threads_sorted:
                f.write("---:|")
            f.write("\n")
            for mk in models_keys:
                r1 = sweep_results.get(1, {}).get(mk)
                base = r1["benchmark"]["latency_ms_p50"] if r1 else None
                f.write(f"| {MODELS[mk]['display']} |")
                for t in threads_sorted:
                    r = sweep_results.get(t, {}).get(mk)
                    if r is None or base is None:
                        f.write(" — |")
                    elif t == 1:
                        f.write(" 1.00× |")
                    else:
                        sp = base / r["benchmark"]["latency_ms_p50"]
                        f.write(f" {sp:.2f}× |")
                f.write("\n")
            f.write(f"\nA speedup close to N indicates good parallelization "
                    f"(typical of CNNs with SIMD). A speedup close to 1 "
                    f"indicates a sequential model that does not benefit "
                    f"from more cores (typical of recurrent networks such as CfC).\n\n")

        any_temp = any(
            r and r["benchmark"].get("temp_delta_c") is not None
            for t_results in sweep_results.values()
            for r in t_results.values()
        )
        if any_temp:
            f.write(f"## SoC temperature during the benchmark\n\n")
            f.write("| Model | Threads | Temp before (°C) | Temp after (°C) "
                    "| Δ (°C) | Final clock (MHz) |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for t in threads_sorted:
                for mk in models_keys:
                    r = sweep_results.get(t, {}).get(mk)
                    if r is None:
                        continue
                    b = r["benchmark"]
                    sb = b.get("temp_snapshot_before") or {}
                    sa = b.get("temp_snapshot_after") or {}
                    t_before = sb.get("temp_c")
                    t_after = sa.get("temp_c")
                    delta = b.get("temp_delta_c")
                    clk_after = sa.get("clock_mhz")
                    f.write(f"| {MODELS[mk]['display']} | {t} | "
                            f"{t_before if t_before is not None else '—'} | "
                            f"{t_after if t_after is not None else '—'} | "
                            f"{delta if delta is not None else '—'} | "
                            f"{int(clk_after) if clk_after else '—'} |\n")
            f.write(f"\n")

        f.write(f"## Per-configuration detail\n\n")
        for t in threads_sorted:
            f.write(f"### {t} thread{'s' if t > 1 else ''}\n\n")
            f.write("| Model | Params | Memory | p50 (ms) | p95 (ms) "
                    "| p99 (ms) | Throughput |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for mk in models_keys:
                r = sweep_results.get(t, {}).get(mk)
                if r is None:
                    continue
                b = r["benchmark"]
                f.write(f"| {MODELS[mk]['display']} | "
                        f"{r['params']:,} | "
                        f"{r['memory_mb']:.2f} MB | "
                        f"{b['latency_ms_p50']:.2f} | "
                        f"{b['latency_ms_p95']:.2f} | "
                        f"{b['latency_ms_p99']:.2f} | "
                        f"{b['throughput_inf_per_sec']:.1f}/s |\n")
            f.write(f"\n")

        # Computational complexity (MACs/FLOPs)
        any_macs = any(
            r and r.get("macs_info") and r["macs_info"].get("available")
            for t_results in sweep_results.values()
            for r in t_results.values()
        )
        if any_macs:
            f.write(f"## Computational complexity (MACs / FLOPs)\n\n")
            f.write("Architectural count with `thop`. Hardware "
                    "independent. Convention: 1 MAC = 2 FLOPs.\n\n")
            f.write("| Model | Params | MACs/inf (M) | FLOPs/inf (M) |\n")
            f.write("|---|---:|---:|---:|\n")
            # MACs are static: take them from any thread count
            macs_seen = set()
            for t in threads_sorted:
                for mk in models_keys:
                    if mk in macs_seen:
                        continue
                    r = sweep_results.get(t, {}).get(mk)
                    if r is None or not r.get("macs_info"):
                        continue
                    mi = r["macs_info"]
                    if not mi.get("available"):
                        continue
                    f.write(f"| {MODELS[mk]['display']} | "
                            f"{r['params']:,} | "
                            f"{mi['macs'] / 1e6:.2f} | "
                            f"{mi['flops'] / 1e6:.2f} |\n")
                    macs_seen.add(mk)
            f.write(f"\nMACs (multiply-accumulate operations) are the standard "
                    f"computational-complexity metric in the edge AI "
                    f"literature. They do not capture memory or "
                    f"scheduling overhead, but allow a hardware-independent "
                    f"architectural comparison.\n\n")

        # Analytical energy estimate (Pi 5)
        any_energy = any(
            r and r.get("energy_estimate_a76")
              and r["energy_estimate_a76"].get("applicable_to_this_hardware")
            for t_results in sweep_results.values()
            for r in t_results.values()
        )
        if any_energy:
            f.write(f"## Analytical energy estimate (Cortex-A76)\n\n")
            f.write("> **Disclaimer:** the values below are "
                    "**analytical estimates**, NOT direct measurements. "
                    "Derived from published ARM Cortex-A76 power × "
                    "measured latency. Expected error margin: ±30%. "
                    "Sub-10% accuracy requires measurement with a "
                    "USB-C power meter or INA219.\n\n")
            f.write("Power model used:\n\n")
            f.write(f"- SoC base power: {A76_POWER_BASE_W:.2f} W\n")
            f.write(f"- Power per active core (2.4 GHz, NEON SIMD): "
                    f"{A76_POWER_PER_ACTIVE_CORE_W:.2f} W\n\n")
            f.write("| Model | Threads | Latency p50 (ms) "
                    "| Est. power (W) | Est. energy/inf (mJ) |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for t in threads_sorted:
                for mk in models_keys:
                    r = sweep_results.get(t, {}).get(mk)
                    if r is None or not r.get("energy_estimate_a76"):
                        continue
                    ee = r["energy_estimate_a76"]
                    if not ee.get("applicable_to_this_hardware"):
                        continue
                    lat = r["benchmark"]["latency_ms_p50"]
                    pw = ee["estimated_avg_power_w"]
                    en = ee["estimated_energy_per_inference_mj"]
                    f.write(f"| {MODELS[mk]['display']} | {t} | "
                            f"{lat:.2f} | {pw:.2f} | {en:.2f} |\n")
            f.write(f"\n")

        # Thermal sustainability (continuous stress test)
        any_stress = any(
            r and r.get("stress_test")
            for t_results in sweep_results.values()
            for r in t_results.values()
        )
        if any_stress:
            f.write(f"## Thermal sustainability (continuous stress test)\n\n")
            f.write("Continuous inference for "
                    f"{args.stress_test_minutes:.1f} min per configuration, "
                    f"sampling temperature and clock every "
                    f"{args.temp_sample_interval_seconds:.0f}s. "
                    f"Throttling detection: clock < 90% of nominal "
                    f"(< {int(0.9 * 2400)} MHz for Pi 5).\n\n")
            f.write("**Rationale:** on the same SoC with "
                    "constant thermal mass and the same cooling, the temperature "
                    "rise tracks the dissipated electrical energy "
                    "(first law of thermodynamics). Delta T is therefore "
                    "empirical comparative evidence of energy consumption "
                    "without a power meter.\n\n")
            f.write("| Model | Threads | Initial temp | Final temp | Max temp "
                    "| ΔT (°C) | Throttling | Inf. completed |\n")
            f.write("|---|---:|---:|---:|---:|---:|:-:|---:|\n")
            for t in threads_sorted:
                for mk in models_keys:
                    r = sweep_results.get(t, {}).get(mk)
                    if r is None or not r.get("stress_test"):
                        continue
                    st = r["stress_test"]
                    t_i = st.get("temp_initial_c")
                    t_f = st.get("temp_final_c")
                    t_m = st.get("temp_max_c")
                    dt = st.get("temp_delta_c")
                    throttled = st.get("clock_throttled")
                    n_inf = st.get("n_inferences_completed", 0)
                    f.write(f"| {MODELS[mk]['display']} | {t} | "
                            f"{t_i if t_i is not None else '—'} | "
                            f"{t_f if t_f is not None else '—'} | "
                            f"{t_m if t_m is not None else '—'} | "
                            f"{f'{dt:+.1f}' if dt is not None else '—'} | "
                            f"{'YES' if throttled else ('no' if throttled is False else '-')} | "
                            f"{n_inf:,} |\n")
            f.write(f"\nThe corresponding figures (temperature and clock vs time) "
                    f"are saved as `thermal_stress_*.png` in this same "
                    f"directory.\n\n")

        # Duty-cycle (simulated clinical use)
        any_duty = any(
            r and r.get("duty_cycle")
            for t_results in sweep_results.values()
            for r in t_results.values()
        )
        if any_duty:
            f.write(f"## Duty-cycle (simulated clinical use)\n\n")
            f.write(f"One inference every "
                    f"{args.duty_cycle_interval_seconds:.0f}s for "
                    f"{args.duty_cycle_minutes:.1f} min, simulating "
                    f"spaced auscultations in respiratory triage. "
                    f"Unlike the continuous stress test (worst case), this "
                    f"regime represents actual clinical operation: the "
                    f"device does not infer continuously but once per "
                    f"patient.\n\n")
            f.write("**Key metrics:** if temperature stays "
                    "stable without throttling and latency shows no drift "
                    "(degradation), the model is viable for sustained "
                    "clinical deployment without active cooling.\n\n")
            f.write("| Model | Threads | Inf. | Initial temp | Final temp "
                    "| Max temp | ΔT (°C) | Mean lat. (ms) | Lat. drift "
                    "| Stable |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|\n")
            for t in threads_sorted:
                for mk in models_keys:
                    r = sweep_results.get(t, {}).get(mk)
                    if r is None or not r.get("duty_cycle"):
                        continue
                    dc = r["duty_cycle"]
                    t_i = dc.get("temp_initial_c")
                    t_f = dc.get("temp_final_c")
                    t_m = dc.get("temp_max_c")
                    dt = dc.get("temp_delta_c")
                    lat = dc.get("latency_ms_mean")
                    drift = dc.get("latency_drift_pct")
                    stable = dc.get("stable")
                    f.write(f"| {MODELS[mk]['display']} | {t} | "
                            f"{dc.get('n_inferences', 0)} | "
                            f"{t_i if t_i is not None else '—'} | "
                            f"{t_f if t_f is not None else '—'} | "
                            f"{t_m if t_m is not None else '—'} | "
                            f"{f'{dt:+.1f}' if dt is not None else '—'} | "
                            f"{f'{lat:.1f}' if lat is not None else '—'} | "
                            f"{f'{drift:+.1f}%' if drift is not None else '—'} | "
                            f"{'YES' if stable else ('no' if stable is False else '-')} |\n")
            f.write(f"\nThe figures (`duty_cycle_*.png`) show temperature "
                    f"and per-inference latency over time.\n\n")

        # Cooldowns applied (methodological traceability)
        any_cooldown = any(
            r and r.get("cooldown_before")
            for t_results in sweep_results.values()
            for r in t_results.values()
        )
        if any_cooldown:
            f.write(f"## Cooldowns applied (methodological traceability)\n\n")
            f.write(f"Adaptive cooldown target: "
                    f"{args.cooldown_target_temp:.0f}°C "
                    f"(timeout {args.cooldown_max_seconds}s). Gives a "
                    f"common thermal start across models, so delta T is comparable.\n\n")
            f.write("| Model | Threads | Temp before cooldown | Temp after "
                    "| Target reached? | Wait (s) |\n")
            f.write("|---|---:|---:|---:|:-:|---:|\n")
            for t in threads_sorted:
                for mk in models_keys:
                    r = sweep_results.get(t, {}).get(mk)
                    if r is None or not r.get("cooldown_before"):
                        continue
                    cd = r["cooldown_before"]
                    f.write(f"| {MODELS[mk]['display']} | {t} | "
                            f"{cd.get('temp_initial_c', '—')} | "
                            f"{cd.get('temp_final_c', '—')} | "
                            f"{'✅' if cd.get('reached_target') else '⚠️ timeout'} | "
                            f"{cd.get('wait_actual_s', 0):.0f} |\n")
            f.write(f"\n")

        f.write(f"\n---\nGenerated by `15_edge_benchmark.py`: "
                f"multi-thread + temperature + MACs + continuous stress + "
                f"adaptive cooldown + duty-cycle + analytical "
                f"energy.\n")

    print(f"\nCross-product Markdown report: {output_path}")


def main():
    args = parse_args()

    if args.num_threads_list is not None and len(args.num_threads_list) > 0:
        thread_counts = list(args.num_threads_list)
    elif args.num_threads > 0:
        thread_counts = [args.num_threads]
    else:
        thread_counts = [0]  # 0 = automatic
    multi_thread_mode = len(thread_counts) > 1

    device = torch.device("cpu")
    sysinfo = system_info()
    device_label = args.device_label or sysinfo.get(
        "rpi_model", sysinfo.get("machine", "unknown")
    )

    # Dual logger (stdout + .log file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if c.isalnum() else "_" for c in device_label)[:40]
    log_path = RESULTS_DIR / f"edge_benchmark_{safe_label}_{timestamp}.log"
    logger = DualLogger(log_path)
    logger.start()

    try:
        print(f"[CONFIG] Device forced to CPU (CUDA ignored even if present)")

        print(f"\n{'=' * 70}")
        print(f"Script 15: Edge AI Inference Benchmark")
        print(f"{'=' * 70}")
        print(f"  Device label:    {device_label}")
        print(f"  Platform:        {sysinfo['platform']}")
        print(f"  Machine:         {sysinfo['machine']}")
        print(f"  CPU cores:       {sysinfo['cpu_count_logical']} "
              f"(PyTorch threads default: {sysinfo['torch_num_threads']})")
        print(f"  PyTorch:         {sysinfo['torch_version']}")
        if sysinfo.get("is_raspberry_pi"):
            print(f"  Raspberry Pi:    {sysinfo.get('rpi_model', 'detected')}")
        print(f"  Iterations:      {args.num_iter} (warmup: {args.num_warmup})")
        print(f"  Models:          {args.models}")
        print(f"  Thread counts:   {thread_counts}  "
              f"({'SWEEP' if multi_thread_mode else 'single run'})")
        if multi_thread_mode and args.cooldown_seconds > 0:
            print(f"  Cooldown:        {args.cooldown_seconds} s between points")
        print(f"  SoC temperature: "
              f"{'disabled' if args.no_temperature else 'enabled'}")
        print(f"  Stress test:     "
              f"{f'{args.stress_test_minutes:.1f} min/config' if args.stress_test_minutes > 0 else 'disabled'}")
        print(f"  Duty-cycle:      "
              f"{f'{args.duty_cycle_minutes:.1f} min (1 inf/{args.duty_cycle_interval_seconds:.0f}s)' if args.duty_cycle_minutes > 0 else 'disabled'}")
        print(f"  Cooldown:        "
              f"{f'adaptive -> {args.cooldown_target_temp:.0f}°C (timeout {args.cooldown_max_seconds}s)' if args.cooldown_target_temp > 0 else (f'fixed {args.cooldown_seconds}s' if args.cooldown_seconds > 0 else 'disabled')}")
        print(f"  thop (MACs):     "
              f"{'available' if HAS_THOP else 'NOT installed (pip install thop)'}")
        print(f"  matplotlib:      "
              f"{'available' if HAS_MATPLOTLIB else 'NOT installed (no PNG figures)'}")
        print(f"  Log txt:         {log_path.name}")
        print(f"{'=' * 70}")

        if not args.no_temperature:
            snap_init = temperature_snapshot("session_start")
            if snap_init["temp_c"] is not None:
                print(f"\n[TEMPERATURE] Session start: "
                      f"{snap_init['temp_c']:.1f}°C  "
                      f"(clock: {snap_init['clock_mhz']:.0f} MHz)"
                      if snap_init['clock_mhz'] else
                      f"\n[TEMPERATURE] Session start: "
                      f"{snap_init['temp_c']:.1f}°C")

        print(f"\n[Checking checkpoints in {CHECKPOINTS_DIR}]")
        if not CHECKPOINTS_DIR.exists():
            print(f"  ✗ Not found: {CHECKPOINTS_DIR}")
            print(f"  Fix: copy models/checkpoints/ into this directory.")
            sys.exit(1)

        INPUT_SHAPE = (1, 1, 128, 251)   # (B=1, C=1, H=mel_bands, W=time_frames)

        # Main loop: sweep over thread counts
        sweep_results = {}   # {threads_int: {model_key: result_dict}}

        for idx, tc in enumerate(thread_counts):
            print(f"\n\n{'#' * 70}")
            print(f"# SWEEP {idx + 1}/{len(thread_counts)}: num_threads = {tc}")
            print(f"{'#' * 70}")

            sweep_results[tc] = run_models_for_thread_count(
                args=args,
                models_keys=args.models,
                threads_value=tc,
                device=device,
                input_shape=INPUT_SHAPE,
                sysinfo=sysinfo,
            )

            # Cooldown between sweeps (not after the last one)
            if idx < len(thread_counts) - 1:
                if args.cooldown_target_temp > 0 and not args.no_temperature:
                    print(f"\n[COOLDOWN between sweeps] adaptive...")
                    wait_for_cooldown(
                        target_temp_c=args.cooldown_target_temp,
                        max_wait_s=args.cooldown_max_seconds,
                        poll_interval_s=5.0,
                        label="between-sweeps",
                        verbose=True,
                    )
                elif args.cooldown_seconds > 0:
                    # Fixed cooldown
                    print(f"\n[COOLDOWN] Pausing {args.cooldown_seconds} s "
                          f"so the SoC returns to baseline temperature...")
                    if not args.no_temperature:
                        snap_pre = temperature_snapshot("cooldown_start")
                        if snap_pre["temp_c"] is not None:
                            print(f"  Temp before cooldown: "
                                  f"{snap_pre['temp_c']:.1f}°C")
                    time.sleep(args.cooldown_seconds)
                    if not args.no_temperature:
                        snap_post = temperature_snapshot("cooldown_end")
                        if snap_post["temp_c"] is not None:
                            print(f"  Temp after cooldown: "
                                  f"{snap_post['temp_c']:.1f}°C")

        if multi_thread_mode:
            print_cross_product_table(sweep_results, args.models)
        else:
            # Single thread count: simple summary table
            single_threads = thread_counts[0]
            print(f"\n{'=' * 70}")
            print(f"SUMMARY ({device_label}, threads={single_threads})")
            print(f"{'=' * 70}")
            rows = []
            for mk in args.models:
                r = sweep_results[single_threads].get(mk)
                if not r:
                    rows.append([MODELS[mk]["display"],
                                 "—", "—", "—", "—", "—"])
                    continue
                b = r["benchmark"]
                rows.append([
                    MODELS[mk]["display"],
                    f"{r['params']:,}",
                    f"{r['memory_mb']:.2f} MB",
                    f"{b['latency_ms_p50']:.2f} ms",
                    f"{b['latency_ms_p95']:.2f} ms",
                    f"{b['throughput_inf_per_sec']:.0f}/s",
                ])
            print_table(
                rows=rows,
                headers=["Model", "Params", "Memory",
                         "Lat. p50", "Lat. p95", "Throughput"],
            )

        if not args.no_temperature:
            snap_end = temperature_snapshot("session_end")
            if snap_end["temp_c"] is not None:
                print(f"\n[TEMPERATURE] Session end: "
                      f"{snap_end['temp_c']:.1f}°C")

        json_path = RESULTS_DIR / f"edge_benchmark_{safe_label}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "device_label": device_label,
                "system_info": sysinfo,
                "args": {
                    "num_iter": args.num_iter,
                    "num_warmup": args.num_warmup,
                    "num_threads": args.num_threads,
                    "num_threads_list": args.num_threads_list,
                    "thread_counts_used": thread_counts,
                    "cooldown_seconds": args.cooldown_seconds,
                    "cooldown_target_temp": args.cooldown_target_temp,
                    "cooldown_max_seconds": args.cooldown_max_seconds,
                    "no_temperature": args.no_temperature,
                    "skip_thop": args.skip_thop,
                    "stress_test_minutes": args.stress_test_minutes,
                    "duty_cycle_minutes": args.duty_cycle_minutes,
                    "duty_cycle_interval_seconds":
                        args.duty_cycle_interval_seconds,
                    "temp_sample_interval_seconds":
                        args.temp_sample_interval_seconds,
                },
                "dependencies": {
                    "thop_available": HAS_THOP,
                    "matplotlib_available": HAS_MATPLOTLIB,
                },
                "sweep_results": {
                    str(tc): {
                        mk: r for mk, r in res.items() if r is not None
                    }
                    for tc, res in sweep_results.items()
                },
            }, f, indent=2, default=str)
        print(f"\nJSON: {json_path}")

        # Stress-test CSVs and figures
        if args.stress_test_minutes > 0:
            print(f"\n[Saving stress test files...]")
            stress_files = []
            for tc, t_results in sweep_results.items():
                for mk, r in t_results.items():
                    if r is None or not r.get("stress_test"):
                        continue
                    st = r["stress_test"]
                    base_name = (f"thermal_stress_{mk}_{tc}t_"
                                 f"{safe_label}_{timestamp}")
                    csv_path = RESULTS_DIR / f"{base_name}.csv"
                    save_thermal_stress_csv(st, csv_path)
                    stress_files.append(csv_path)
                    print(f"  CSV: {csv_path.name}")

                    if HAS_MATPLOTLIB:
                        png_path = RESULTS_DIR / f"{base_name}.png"
                        ok = plot_thermal_stress(
                            st, png_path, MODELS[mk]["display"]
                        )
                        if ok:
                            stress_files.append(png_path)
                            print(f"  PNG: {png_path.name}")
                    else:
                        print(f"  [INFO] matplotlib not available, "
                              f"skipping PNG figure. "
                              f"Install with: pip install matplotlib")

        # Duty-cycle CSVs and figures
        if args.duty_cycle_minutes > 0:
            print(f"\n[Saving duty-cycle files...]")
            for tc, t_results in sweep_results.items():
                for mk, r in t_results.items():
                    if r is None or not r.get("duty_cycle"):
                        continue
                    dc = r["duty_cycle"]
                    base_name = (f"duty_cycle_{mk}_{tc}t_"
                                 f"{safe_label}_{timestamp}")
                    csv_path = RESULTS_DIR / f"{base_name}.csv"
                    save_duty_cycle_csv(dc, csv_path)
                    print(f"  CSV: {csv_path.name}")

                    if HAS_MATPLOTLIB:
                        png_path = RESULTS_DIR / f"{base_name}.png"
                        ok = plot_duty_cycle(
                            dc, png_path, MODELS[mk]["display"]
                        )
                        if ok:
                            print(f"  PNG: {png_path.name}")

        # Markdown: simple report for one thread count, cross-product otherwise
        if multi_thread_mode:
            md_path = (RESULTS_DIR /
                       f"edge_benchmark_{safe_label}_{timestamp}_crossproduct.md")
            write_cross_product_markdown(
                sweep_results, args.models, sysinfo, md_path,
                device_label, args,
            )
        else:
            md_path = RESULTS_DIR / f"edge_benchmark_{safe_label}_{timestamp}.md"
            write_markdown_report(
                sweep_results[thread_counts[0]], sysinfo, md_path, device_label,
            )

        print(f"\n{'=' * 70}")
        print(f"BENCHMARK COMPLETED")
        print(f"{'=' * 70}")
        print(f"  Full report:         {md_path}")
        print(f"  Raw data:            {json_path}")
        print(f"  Run log:             {log_path}")
        if args.stress_test_minutes > 0:
            print(f"  Stress test files:   {RESULTS_DIR}/thermal_stress_*")

    finally:
        logger.stop()


if __name__ == "__main__":
    main()
