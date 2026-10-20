#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 40: Robustness of CNN-RespiratoryNet to additive noise.

Pure inference on the five existing checkpoints: the same test cycles are
contaminated with white noise at several signal-to-noise ratios (SNR) and
re-evaluated. Noise is band-passed with the same 50-2000 Hz Butterworth filter
and added to the already filtered segments; by linearity of filtfilt this is
exact. The SNR is defined inside the band and over the non-padded samples. The
clean condition rebuilds the log-mel from the WAV and must reproduce the
official figures (sanity check). This is a sensitivity analysis only: the main
configuration of the article does not change with its outcome.

Inputs:  data/splits/strategy1_official/test_cycles.csv, data/segments/*.wav,
         models/checkpoints/cnn_official_seed{42,123,7,99,314}.pth,
         scripts/09_cnn_training.py (RespiratoryNet class)
Outputs: results/metrics/noise_robustness/noise_robustness_{runs,summary,perclass}.csv,
         results/metrics/noise_robustness/noise_robustness.json,
         results/metrics/noise_robustness/S16_noise_robustness_summary.md,
         noise_robustness_log.txt
Usage:
    python 40_noise_robustness.py [--snr 20 10 5 0] [--limit 200] [--device cpu]
"""

import sys
import json
import argparse
import importlib.util
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy import signal

import torch


SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR_FIXED = Path(__file__).resolve().parent.parent

if BASE_DIR_FIXED.exists():
    BASE_DIR = BASE_DIR_FIXED
elif (SCRIPTS_DIR.parent / "data").exists():
    BASE_DIR = SCRIPTS_DIR.parent
else:
    BASE_DIR = SCRIPTS_DIR

CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
CLASS_MAP = {"Normal": 0, "Crackle": 1, "Wheeze": 2, "Both": 3}

SEEDS = [42, 123, 7, 99, 314]

# SNR levels in dB. None = clean condition (sanity check).
SNR_LEVELS = [None, 20, 10, 5, 0]

# Front-end parameters. Must match Scripts 03 and 06, otherwise the sanity
# check is invalid.
AUDIO_CONFIG = {
    "target_sr": 16000,
    "bandpass_lowcut": 50,
    "bandpass_highcut": 2000,
    "bandpass_order": 5,
    "rms_target": 0.1,
}

MEL_CONFIG = {
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "fmin": 50,
    "fmax": 2000,
    "power": 2.0,
    "top_db": 80.0,
}

# Official figures of the article used by the sanity check.
OFFICIAL_ICBHI = 0.460
OFFICIAL_AUC = 0.572
TOLERANCE = 0.010   # tolerance on the five-seed mean

# Base seed of the noise generator. Fixed and independent of the model seeds,
# so that all five runs see exactly the same noise and the comparison is paired.
NOISE_SEED_BASE = 20260917


logger = logging.getLogger(__name__)


def setup_logging(base_dir):
    """Log to stdout and to noise_robustness_log.txt in BASE_DIR."""
    log_file = base_dir / "noise_robustness_log.txt"
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False   # avoid duplicated lines from the root logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return log_file


def section(title_text):
    logger.info("")
    logger.info("=" * 74)
    logger.info(f"  {title_text}")
    logger.info("=" * 74)


def design_bandpass(sr):
    """
    Order-5 Butterworth band-pass, identical to Script 03.

    Returned as (b, a) rather than SOS because Script 03 uses filtfilt with
    (b, a); the same implementation avoids numerical differences that would
    break the sanity check.
    """
    nyquist = 0.5 * sr
    low = AUDIO_CONFIG["bandpass_lowcut"] / nyquist
    high = AUDIO_CONFIG["bandpass_highcut"] / nyquist
    return signal.butter(AUDIO_CONFIG["bandpass_order"], [low, high], btype="band")


def generate_filtered_noise(audio, snr_db, b, a, rng):
    """
    Generate band-passed white noise scaled to the target SNR within 50-2000 Hz.

    Scaling is applied after filtering: full-band white noise at 16 kHz loses
    about 75 % of its power through the band-pass, so setting the SNR before
    the filter would not reflect what the model sees. Signal power is measured
    over the non-zero samples (the actual cycle, not the zero-padding added by
    Script 05); the noise spans the full 8 s window.

    Returns (noise, effective_snr_db); the effective SNR is recomputed on the
    scaled noise as a check and should match the target.
    """
    mask_sel = np.abs(audio) > 1e-8

    if mask_sel.sum() < 10:
        # Practically empty segment: the SNR cannot be set.
        return np.zeros_like(audio), float("nan")

    signal_power = np.mean(audio[mask_sel] ** 2)
    target_value = signal_power / (10.0 ** (snr_db / 10.0))

    noise = signal.filtfilt(b, a, rng.normal(0.0, 1.0, size=audio.shape))
    current_power = np.mean(noise[mask_sel] ** 2)

    if current_power < 1e-20:
        return np.zeros_like(audio), float("nan")

    noise = noise * np.sqrt(target_value / current_power)

    effective_snr = 10.0 * np.log10(signal_power / max(np.mean(noise[mask_sel] ** 2), 1e-20))

    return noise, float(effective_snr)


def build_mel(audio, sr):
    """
    Log-mel normalized per cycle to [0, 1]; exact replica of Script 06.

    The second normalization in the Script 08 Dataset is idempotent on an
    array already in [0, 1], so this is the same tensor the network sees
    elsewhere in the study.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=MEL_CONFIG["n_mels"],
        n_fft=MEL_CONFIG["n_fft"],
        hop_length=MEL_CONFIG["hop_length"],
        fmin=MEL_CONFIG["fmin"],
        fmax=MEL_CONFIG["fmax"],
        power=MEL_CONFIG["power"],
    )

    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=MEL_CONFIG["top_db"])

    mn, mx = mel_db.min(), mel_db.max()
    if mx - mn > 1e-6:
        mel_norm = (mel_db - mn) / (mx - mn)
    else:
        mel_norm = np.zeros_like(mel_db)

    return mel_norm.astype(np.float32)


def process_cycle(wav_path, snr_db, b, a, rng):
    """
    Per-cycle chain: read, add filtered noise, compute the mel.

    The signal is neither re-filtered nor RMS-renormalized, because the
    segments are already filtered and normalized by Scripts 03 and 05. Since
    filtfilt(s + n) = filtfilt(s) + filtfilt(n), adding filtered noise to the
    segment exactly emulates noise entering before signal conditioning.
    RMS renormalization is unnecessary because power_to_db uses ref=np.max and
    the mel is min-max normalized. With snr_db = None the cycle goes unchanged
    to the mel, reproducing Script 06.
    """
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float64)

    effective_snr = float("nan")
    if snr_db is not None:
        noise, effective_snr = generate_filtered_noise(audio, snr_db, b, a, rng)
        audio = audio + noise

    return build_mel(audio, sr), effective_snr


def import_module_from_path(name, path):
    """Import a project script as a module."""
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_models(base_dir, device):
    """
    Rebuild the five CNN-RespiratoryNet models from their checkpoints.

    Uses the configuration stored in each checkpoint instead of assuming
    hyperparameters (same logic as Script 18 for kind='cnn').
    """
    script09 = import_module_from_path("script09", SCRIPTS_DIR / "09_cnn_training.py")
    ckpt_dir = base_dir / "models" / "checkpoints"

    model_set = {}
    for seed in SEEDS:
        ckpt_path = ckpt_dir / f"cnn_official_seed{seed}.pth"
        if not ckpt_path.exists():
            logger.warning(f"  Checkpoint not found: {ckpt_path.name}")
            continue

        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        cfg = ck.get("config", {})

        model = script09.RespiratoryNet(
            n_classes=cfg.get("n_classes", 4),
            input_channels=cfg.get("input_channels", 1),
            base_filters=cfg.get("base_filters", 16),
            n_conv_blocks=cfg.get("n_conv_blocks", 4),
            fc_hidden_dim=cfg.get("fc_hidden_dim", 128),
            dropout_rate=cfg.get("dropout_rate", 0.5),
        ).to(device)

        model.load_state_dict(ck["model_state_dict"])
        model.eval()

        model_set[seed] = model
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Seed {seed:>3d}: loaded ({n_params:,} parameters)")

    return model_set


def icbhi_official(y_true, y_pred):
    """
    Official ICBHI score = (Se + Sp) / 2 with the strict subclass criterion.

    Se = abnormal cycles classified into their own subclass / abnormal cycles;
    Sp = normal cycles classified as Normal / normal cycles. A Crackle
    predicted as Wheeze does not count as a hit.
    """
    es_normal = (y_true == 0)
    n_normal = int(es_normal.sum())
    n_abnormal = int((~es_normal).sum())

    sp = float((y_pred[es_normal] == 0).sum()) / max(n_normal, 1)
    se = float((y_pred[~es_normal] == y_true[~es_normal]).sum()) / max(n_abnormal, 1)

    return (se + sp) / 2.0, se, sp


def binary_auc(y_true, p_abn):
    """
    Binary Normal-vs-abnormal AUC with half credit for ties.

    Rank-based implementation, identical to the Mann-Whitney estimator with
    ties and avoiding an O(Na*Nn) double loop.
    """
    y_bin = (y_true != 0).astype(int)
    n_pos = int(y_bin.sum())
    n_neg = len(y_bin) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    from scipy.stats import rankdata
    range_list = rankdata(p_abn)
    pos_sum = range_list[y_bin == 1].sum()

    return float((pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sensitivity_per_class(y_true, y_pred):
    """One-vs-rest sensitivity of each of the four classes."""
    out = {}
    for idx, name_text in enumerate(CLASS_NAMES):
        mask_sel = (y_true == idx)
        n = int(mask_sel.sum())
        out[name_text] = float((y_pred[mask_sel] == idx).sum()) / max(n, 1)
    return out


def build_inputs(test_df, segments_dir, snr_db, b, a):
    """
    Build the input tensor of the whole test partition for one SNR level.

    Returns (X, effective_snrs, valid_indices). Noise is deterministic: each
    cycle gets a seed derived from its position and the SNR level, so the
    contamination is identical across runs and executions.
    """
    label_text = "clean" if snr_db is None else f"{snr_db} dB"
    logger.info(f"  Building inputs ({label_text}) ...")

    mels, snrs, valid_vals = [], [], []
    missing_items = 0

    for i, row_data in enumerate(test_df.itertuples(index=False)):
        wav_path = segments_dir / row_data.cycle_filename

        if not wav_path.exists():
            missing_items += 1
            continue

        seed_value = NOISE_SEED_BASE + i * 1000 + (0 if snr_db is None else int(snr_db))
        rng = np.random.default_rng(seed_value)

        mel, snr_ef = process_cycle(wav_path, snr_db, b, a, rng)

        mels.append(mel)
        snrs.append(snr_ef)
        valid_vals.append(i)

        if (len(mels) % 500) == 0:
            logger.info(f"    {len(mels)} / {len(test_df)} cycles")

    if missing_items:
        logger.warning(f"    {missing_items} WAV segments not found, skipped")

    X = torch.from_numpy(np.stack(mels)).unsqueeze(1)   # (N, 1, 128, 251)
    return X, np.array(snrs), np.array(valid_vals)


@torch.no_grad()
def run_inference(model, X, device, batch_size=64):
    """Softmax over the whole partition; returns (N, 4) in float64."""
    output_paths = []
    for i in range(0, len(X), batch_size):
        batch_list = X[i:i + batch_size].to(device)
        logits = model(batch_list)
        output_paths.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(output_paths).astype(np.float64)


def parse_args():
    p = argparse.ArgumentParser(
        description="Script 40 - Robustness to additive noise"
    )
    p.add_argument("--base_dir", type=str, default=None, help="Override BASE_DIR")
    p.add_argument("--snr", type=float, nargs="+", default=None,
                   help="SNR levels in dB (the clean condition is always included)")
    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N cycles (quick test)")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else BASE_DIR
    log_file = setup_logging(base_dir)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    snr_levels_list = [None] + list(args.snr) if args.snr else SNR_LEVELS

    logger.info("=" * 74)
    logger.info("  SCRIPT 40 - ROBUSTNESS TO ADDITIVE NOISE")
    logger.info("=" * 74)
    logger.info(f"  Date:      {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"  BASE_DIR:  {base_dir}")
    logger.info(f"  Device:    {device}")
    logger.info(f"  Levels:    {['clean' if s is None else f'{s} dB' for s in snr_levels_list]}")
    logger.info(f"  Log:       {log_file}")

    split_csv = base_dir / "data" / "splits" / "strategy1_official" / "test_cycles.csv"
    segments_dir = base_dir / "data" / "segments"
    out_dir = base_dir / "results" / "metrics" / "noise_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path_str in [split_csv, segments_dir]:
        if not path_str.exists():
            logger.error(f"  Not found: {path_str}")
            return 1

    section("TEST PARTITION")
    test_df = pd.read_csv(split_csv)

    if "cycle_filename" not in test_df.columns:
        logger.error("  test_cycles.csv has no 'cycle_filename' column.")
        logger.error("  Without it the cycle WAV cannot be located.")
        return 1

    if args.limit:
        test_df = test_df.head(args.limit).copy()
        logger.warning(f"  TEST MODE: only {len(test_df)} cycles. "
                       f"Results are NOT publishable.")

    logger.info(f"  Cycles: {len(test_df)}")
    for name_text in CLASS_NAMES:
        n = int((test_df["cycle_class"] == name_text).sum())
        logger.info(f"    {name_text:>8s}: {n:5d}")

    y_true_full = test_df["cycle_class"].map(CLASS_MAP).to_numpy()

    section("CHECKPOINTS")
    model_set = load_models(base_dir, device)
    if not model_set:
        logger.error("  No checkpoint was loaded. Aborting.")
        return 1

    # Filter designed once
    b, a = design_bandpass(AUDIO_CONFIG["target_sr"])

    row_list, class_rows = [], []

    for snr_db in snr_levels_list:
        label_text = "clean" if snr_db is None else f"{snr_db:g}"
        section(f"CONDITION: {label_text}")

        X, snrs_ef, valid_vals = build_inputs(test_df, segments_dir, snr_db, b, a)
        y_true = y_true_full[valid_vals]

        if snr_db is not None:
            finite_vals = snrs_ef[np.isfinite(snrs_ef)]
            logger.info(f"    Effective SNR: {finite_vals.mean():.2f} +/- {finite_vals.std(ddof=1):.2f} dB")

        acc_icbhi, acc_auc, acc_se, acc_sp = [], [], [], []
        acc_class = {c: [] for c in CLASS_NAMES}

        for seed, model in model_set.items():
            probs = run_inference(model, X, device, args.batch_size)
            y_pred = np.argmax(probs, axis=1)
            p_abn = 1.0 - probs[:, 0]

            icbhi, se, sp = icbhi_official(y_true, y_pred)
            auc = binary_auc(y_true, p_abn)
            per_c = sensitivity_per_class(y_true, y_pred)

            acc_icbhi.append(icbhi)
            acc_auc.append(auc)
            acc_se.append(se)
            acc_sp.append(sp)
            for c in CLASS_NAMES:
                acc_class[c].append(per_c[c])

            row_list.append({
                "snr_db": label_text, "seed": seed,
                "icbhi": icbhi, "se": se, "sp": sp, "auc_bin": auc,
                **{f"se_{c.lower()}": per_c[c] for c in CLASS_NAMES},
            })

            logger.info(f"    seed {seed:>3d}: ICBHI {icbhi:.3f}  "
                        f"Se {se:.3f}  Sp {sp:.3f}  AUC {auc:.3f}")

        # Sample standard deviation (ddof=1), as everywhere in the article
        m_icbhi, s_icbhi = np.mean(acc_icbhi), np.std(acc_icbhi, ddof=1)
        m_auc, s_auc = np.mean(acc_auc), np.std(acc_auc, ddof=1)

        logger.info(f"    MEAN: ICBHI {m_icbhi:.3f} +/- {s_icbhi:.3f}   "
                    f"AUC {m_auc:.3f} +/- {s_auc:.3f}")

        for c in CLASS_NAMES:
            class_rows.append({
                "snr_db": label_text, "class": c,
                "se_mean": float(np.mean(acc_class[c])),
                "se_sd": float(np.std(acc_class[c], ddof=1)),
            })

        # Sanity check on the clean condition
        if snr_db is None and not args.limit:
            section("SANITY CHECK (clean condition vs official figures)")
            d_icbhi = abs(m_icbhi - OFFICIAL_ICBHI)
            d_auc = abs(m_auc - OFFICIAL_AUC)

            logger.info(f"  ICBHI: {m_icbhi:.4f} vs {OFFICIAL_ICBHI:.4f} official "
                        f"(delta {d_icbhi:.4f})")
            logger.info(f"  AUC:   {m_auc:.4f} vs {OFFICIAL_AUC:.4f} official "
                        f"(delta {d_auc:.4f})")

            if d_icbhi <= TOLERANCE and d_auc <= TOLERANCE:
                logger.info("  PASSED. The mel reconstruction reproduces the article.")
                logger.info("  The noise curves rest on a verified clean condition.")
            else:
                logger.error("")
                logger.error("  *** SANITY CHECK FAILED ***")
                logger.error("  The mel reconstruction does NOT reproduce the official figures.")
                logger.error("  Likely causes, in order: mel parameters differ from")
                logger.error("  Script 06, filter differs from Script 03, or the")
                logger.error("  WAV segments do not correspond to this split.")
                logger.error("  DO NOT REPORT THESE CURVES until this is resolved.")
                logger.error("")

    section("OUTPUTS")

    df_runs = pd.DataFrame(row_list)
    df_runs.to_csv(out_dir / "noise_robustness_runs.csv", index=False, float_format="%.6f")

    df_sum = (df_runs.groupby("snr_db", sort=False)
              .agg(icbhi_mean=("icbhi", "mean"), icbhi_sd=("icbhi", lambda x: x.std(ddof=1)),
                   se_mean=("se", "mean"), sp_mean=("sp", "mean"),
                   auc_mean=("auc_bin", "mean"), auc_sd=("auc_bin", lambda x: x.std(ddof=1)))
              .reset_index())
    df_sum.to_csv(out_dir / "noise_robustness_summary.csv", index=False, float_format="%.6f")

    df_class = pd.DataFrame(class_rows)
    df_class.to_csv(out_dir / "noise_robustness_perclass.csv", index=False, float_format="%.6f")

    with open(out_dir / "noise_robustness.json", "w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "snr_levels": ["clean" if s is None else s for s in snr_levels_list],
            "seeds": list(model_set.keys()),
            "n_cycles": int(len(test_df)),
            "audio_config": AUDIO_CONFIG,
            "mel_config": MEL_CONFIG,
            "noise_seed_base": NOISE_SEED_BASE,
            "summary": df_sum.to_dict(orient="records"),
            "per_run": df_runs.to_dict(orient="records"),
            "per_class": df_class.to_dict(orient="records"),
        }, f, indent=2, ensure_ascii=False)

    # Markdown summary for the supplementary material
    md = out_dir / "S16_noise_robustness_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# S16 - Robustness to additive noise\n\n")
        f.write(f"Generated on {datetime.now():%Y-%m-%d}.\n\n")
        f.write("CNN-RespiratoryNet evaluated on the official ICBHI 2017 test set ")
        f.write("with white Gaussian noise added before the 50-2000 Hz front end, ")
        f.write("at several signal-to-noise ratios. The five checkpoints of the ")
        f.write("article are reused without retraining, and the noise realization ")
        f.write("is identical across seeds, so the comparison stays paired.\n\n")
        f.write("Signal power is measured over the non-padded samples of each cycle. ")
        f.write("Values are the mean over five seeds, with the sample standard ")
        f.write("deviation.\n\n")
        f.write("| SNR | ICBHI | Se | Sp | AUC_bin |\n")
        f.write("|---|---|---|---|---|\n")
        for _, r in df_sum.iterrows():
            f.write(f"| {r['snr_db']} | {r['icbhi_mean']:.3f} ± {r['icbhi_sd']:.3f} | "
                    f"{r['se_mean']:.3f} | {r['sp_mean']:.3f} | "
                    f"{r['auc_mean']:.3f} ± {r['auc_sd']:.3f} |\n")
        f.write("\n## Per-class sensitivity\n\n")
        f.write("| SNR | " + " | ".join(CLASS_NAMES) + " |\n")
        f.write("|---|" + "---|" * len(CLASS_NAMES) + "\n")
        for label_text in df_class["snr_db"].unique():
            sub = df_class[df_class["snr_db"] == label_text].set_index("class")
            vals = " | ".join(f"{sub.loc[c, 'se_mean']:.3f}" for c in CLASS_NAMES)
            f.write(f"| {label_text} | {vals} |\n")
        f.write("\nThe clean condition rebuilds the log-mel from the segmented WAV ")
        f.write("instead of reading the precomputed array, and reproduces the ")
        f.write("figures of Table 4, which validates the reconstruction.\n")

    for p in sorted(out_dir.glob("noise_robustness*")) + [md]:
        logger.info(f"  {p.name}")

    logger.info("")
    logger.info("=" * 74)
    logger.info("  END OF SCRIPT 40")
    logger.info("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
