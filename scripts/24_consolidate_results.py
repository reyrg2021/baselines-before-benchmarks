#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 24: Consolidate all results into a single source of truth.

Aggregates the already computed results (official ICBHI metric on the patient-independent
split) without recomputing anything: accuracy of every model family, INT8 quantization,
and Raspberry Pi 5 deployment benchmarks.

Inputs:  results/metrics/full_reeval/full_reeval_results.json (CNN + CfC-Hybrid, 5 seeds),
         results/accuracy/tl_mobilenet_*_seed*.json (MobileNet, pretrained and scratch),
         results/accuracy/panns_cnn6_*_seed*.json (PANNs, official and tune),
         results/quantization/cnn_int8_quant_report.json,
         results/edge_benchmark/pi5/edge_benchmark_*.json (latency/throughput/thermal)
Outputs: results/master_results.json (all results), results/master_results.csv (flat accuracy table)
Usage:
    python 24_consolidate_results.py
"""

import sys
import json
import csv
import glob
import statistics
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
ACC_DIR = RESULTS_DIR / "accuracy"
QUANT_DIR = RESULTS_DIR / "quantization"
EDGE_DIR = RESULTS_DIR / "edge_benchmark" / "pi5"
# Script 18 writes its results here, not under accuracy/
REEVAL_DIR = RESULTS_DIR / "metrics" / "full_reeval"


def _mean_std(values):
    """Mean and population std of the non-None values; (None, None) if there are none."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.pstdev(vals))


def agg_from_per_seed_jsons(pattern):
    """Aggregate a model family from its per-seed JSON files (format of scripts 21/22)."""
    paths = sorted(glob.glob(str(ACC_DIR / pattern)))
    if not paths:
        return None
    icbhi, se, sp, auc, mf1, params, seeds = [], [], [], [], [], [], []
    per_class_acc = {c: [] for c in ["Normal", "Crackle", "Wheeze", "Both"]}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        t = d["test"]
        icbhi.append(t.get("icbhi_official"))
        se.append(t.get("se_official"))
        sp.append(t.get("sp_official"))
        mf1.append(t.get("macro_f1"))
        auc.append(t.get("binary", {}).get("auc_roc"))
        params.append(d.get("n_params"))
        seeds.append(d.get("seed"))
        for c in per_class_acc:
            if c in t.get("per_class", {}):
                per_class_acc[c].append(t["per_class"][c]["se"])
    m_ic, s_ic = _mean_std(icbhi)
    m_se, s_se = _mean_std(se)
    m_sp, s_sp = _mean_std(sp)
    m_auc, s_auc = _mean_std(auc)
    m_mf1, _ = _mean_std(mf1)
    return {
        "n_seeds": len(paths),
        "seeds": seeds,
        "n_params": params[0] if params else None,
        "icbhi_official": {"mean": m_ic, "std": s_ic, "values": icbhi},
        "se_official": {"mean": m_se, "std": s_se},
        "sp_official": {"mean": m_sp, "std": s_sp},
        "auc_binary": {"mean": m_auc, "std": s_auc, "values": auc},
        "macro_f1": {"mean": m_mf1},
        "per_class_se_mean": {c: _mean_std(v)[0] for c, v in per_class_acc.items()},
    }


def agg_from_full_reeval(model_key):
    """Read the 'aggregated' branch of the full re-evaluation (CNN / CfC-Hybrid)."""
    p = REEVAL_DIR / "full_reeval_results.json"
    if not p.exists():
        return None
    d = json.load(open(p, encoding="utf-8"))
    agg = d.get("aggregated", {}).get(model_key)
    if not agg:
        return None
    # The binary AUC is stored per seed under per_model_seed
    auc_vals = []
    params = None
    for x in d.get("per_model_seed", {}).get(model_key, []):
        b = x.get("binary", {})
        if b.get("auc_roc") is not None:
            auc_vals.append(b["auc_roc"])
        params = x.get("params", params)
    m_auc, s_auc = _mean_std(auc_vals)
    return {
        "n_seeds": agg.get("n_seeds"),
        "seeds": agg.get("seeds"),
        "n_params": params,
        "icbhi_official": {"mean": agg["icbhi_official"]["mean"],
                           "std": agg["icbhi_official"]["std"],
                           "values": agg["icbhi_official"].get("values")},
        "se_official": {"mean": agg["se_official"]["mean"], "std": agg["se_official"]["std"]},
        "sp_official": {"mean": agg["sp_official"]["mean"], "std": agg["sp_official"]["std"]},
        "auc_binary": {"mean": m_auc, "std": s_auc, "values": auc_vals},
    }


def load_quantization():
    """Load the INT8 quantization report, or None if it does not exist."""
    p = QUANT_DIR / "cnn_int8_quant_report.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def load_edge_benchmark():
    """Extract per-model latency, throughput and stress-test figures from the latest Pi 5 benchmark."""
    paths = sorted(glob.glob(str(EDGE_DIR / "edge_benchmark_*.json")))
    if not paths:
        return None
    d = json.load(open(paths[-1], encoding="utf-8"))   # most recent file
    mods = d.get("sweep_results", {})
    if mods and "cnn" not in mods:
        mods = mods[list(mods.keys())[0]]
    out = {"source_file": Path(paths[-1]).name, "device_label": d.get("device_label"), "per_model": {}}
    for mk in ["cnn", "cnn_int8", "hybrid"]:
        r = mods.get(mk)
        if not r:
            continue
        b = r.get("benchmark", {})
        st = r.get("stress_test", {})
        out["per_model"][mk] = {
            "latency_p50_ms": b.get("latency_ms_p50"),
            "latency_p95_ms": b.get("latency_ms_p95"),
            "throughput_inf_s": b.get("throughput_inf_s"),
            "macs_m": b.get("macs_millions") or b.get("macs_m"),
            "energy_mj_est": b.get("energy_per_inf_mj") or b.get("energy_mj"),
            "stress_temp_max_c": st.get("temp_max_c"),
            "stress_clock_min_mhz": st.get("clock_min_mhz"),
            "stress_throttled": st.get("clock_throttled"),
            "stress_n_inferences": st.get("n_inferences_completed"),
        }
    return out


def main():
    print("=" * 70); print("Script 24: Results consolidation"); print("=" * 70)

    families = {
        "CNN_FP32":            agg_from_full_reeval("cnn"),
        "CfC_Hybrid":          agg_from_full_reeval("hybrid"),
        "MobileNetV3_ImageNet": agg_from_per_seed_jsons("tl_mobilenet_v3_small_pretrained_official_seed*_metrics.json"),
        "MobileNetV3_scratch":  agg_from_per_seed_jsons("tl_mobilenet_v3_small_official_seed*_metrics.json"),
        "PANNs_CNN6_AudioSet":  agg_from_per_seed_jsons("panns_cnn6_official_seed*_metrics.json"),
        "PANNs_CNN6_tune":      agg_from_per_seed_jsons("panns_cnn6_tune_official_seed*_metrics.json"),
    }
    for name, agg in families.items():
        if agg is None:
            print(f"  [!] no data for {name}")
        else:
            ic = agg["icbhi_official"]; au = agg["auc_binary"]
            print(f"  {name:22s}: ICBHI {ic['mean']:.4f}±{ic['std']:.4f} | "
                  f"AUC bin {au['mean'] if au['mean'] else float('nan'):.4f} | n={agg['n_seeds']}")

    quant = load_quantization()
    edge = load_edge_benchmark()

    master = {
        "description": "Consolidated results: official ICBHI score, official split (strategy1_official, 2733 test cycles).",
        "metric_note": "Official ICBHI = (Se_abnormal_pooled + Sp_Normal)/2, not the macro average. Binary AUC = Normal-vs-abnormal safeguard against collapse.",
        "accuracy": families,
        "quantization_int8": quant,
        "edge_deployment_pi5": edge,
    }
    out_json = RESULTS_DIR / "master_results.json"
    json.dump(master, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_json}")

    # Flat accuracy table
    out_csv = RESULTS_DIR / "master_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["family", "n_params", "n_seeds", "icbhi_mean", "icbhi_std",
                    "se_mean", "sp_mean", "auc_bin_mean", "auc_bin_std"])
        for name, a in families.items():
            if a is None:
                continue
            w.writerow([name, a.get("n_params"), a["n_seeds"],
                        round(a["icbhi_official"]["mean"], 4), round(a["icbhi_official"]["std"], 4),
                        round(a["se_official"]["mean"], 4), round(a["sp_official"]["mean"], 4),
                        round(a["auc_binary"]["mean"], 4) if a["auc_binary"]["mean"] else "",
                        round(a["auc_binary"]["std"], 4) if a["auc_binary"]["std"] else ""])
    print(f"  Saved: {out_csv}")
    print("\nDone. master_results.json/.csv written.")
    return master


if __name__ == "__main__":
    main()
