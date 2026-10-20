#!/usr/bin/env python3
"""
Script 02: Signal quality assessment of the ICBHI 2017 recordings.

Computes per-recording SNR, clipping, prolonged silence and annotation/audio
synchronization, and summarizes sample rates and respiratory-cycle counts. The list of
problematic recordings (low SNR, severe clipping, sync issues) is consumed by script 03
to decide which recordings to exclude; its "File:", "Severe clipping:" and "Low SNR:" lines are
parsed there and must keep that exact wording.

Inputs:  data/raw/ (ICBHI 2017 .wav and .txt annotation files)
Outputs: results/quality_analysis/audio_quality_summary_FULL.csv,
         results/quality_analysis/statistical_report.txt,
         results/quality_analysis/quality_histograms.png,
         results/quality_analysis/problematic_files.txt
Usage:
    python scripts/02_analisis_calidad_completo.py
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def load_audio_file(filepath):
    """Load a WAV file, averaging channels to mono."""
    audio, sr = sf.read(filepath)
    info = sf.info(filepath)

    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    return audio, sr, info

def calculate_snr(audio, sr):
    """Envelope-based SNR in dB: 80th vs 10th percentile of the Hilbert envelope."""
    analytic_signal = signal.hilbert(audio)
    envelope = np.abs(analytic_signal)

    signal_level = np.percentile(envelope, 80)
    noise_level = np.percentile(envelope, 10)

    if noise_level < 1e-10:
        noise_level = 1e-10

    snr_linear = signal_level / noise_level
    snr_db = 20 * np.log10(snr_linear)

    return snr_db

def detect_clipping(audio, threshold=0.99):
    """Percentage of samples above the threshold after peak normalization; clipped if > 0.1%."""
    audio_normalized = audio / (np.max(np.abs(audio)) + 1e-10)
    clipped_samples = np.sum(np.abs(audio_normalized) > threshold)
    clipping_percentage = (clipped_samples / len(audio)) * 100
    is_clipped = clipping_percentage > 0.1

    return clipping_percentage, is_clipped

def detect_silence(audio, sr, threshold_db=-40, min_duration=0.5):
    """Detect silent segments (frame RMS below threshold_db) lasting at least min_duration s."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    silence_frames = rms_db < threshold_db
    times = librosa.frames_to_time(np.arange(len(silence_frames)), sr=sr, hop_length=hop_length)

    silence_segments = []
    in_silence = False
    start_time = 0

    for i, is_silent in enumerate(silence_frames):
        if is_silent and not in_silence:
            start_time = times[i]
            in_silence = True
        elif not is_silent and in_silence:
            end_time = times[i]
            if end_time - start_time >= min_duration:
                silence_segments.append((start_time, end_time))
            in_silence = False

    total_silence = sum([end - start for start, end in silence_segments])

    return silence_segments, total_silence

def load_annotations(annotation_file):
    """Load an ICBHI .txt annotation file (start, end, crackles, wheezes)."""
    if not annotation_file.exists():
        return None

    annotations = pd.read_csv(annotation_file, sep='\t', header=None,
                             names=['start', 'end', 'crackles', 'wheezes'])
    return annotations

def validate_annotation_sync(audio_duration, annotations):
    """Check annotations against the audio: out-of-range, overlapping or invalid cycles."""
    issues = []

    max_annotation_time = annotations['end'].max()
    if max_annotation_time > audio_duration + 0.1:
        issues.append(f"Annotation exceeds audio duration: {max_annotation_time:.3f}s > {audio_duration:.3f}s")

    for i in range(len(annotations) - 1):
        if annotations.iloc[i]['end'] > annotations.iloc[i+1]['start']:
            issues.append(f"Overlap detected: cycle {i} ends at {annotations.iloc[i]['end']:.3f}s, "
                         f"cycle {i+1} starts at {annotations.iloc[i+1]['start']:.3f}s")

    for i, row in annotations.iterrows():
        if row['start'] >= row['end']:
            issues.append(f"Invalid cycle {i}: start={row['start']:.3f}s >= end={row['end']:.3f}s")

    return issues

def analyze_audio_file(wav_path, annotation_path):
    """Compute all quality and annotation statistics for one recording."""
    audio, sr, info = load_audio_file(wav_path)
    duration = len(audio) / sr

    annotations = load_annotations(annotation_path)

    snr_db = calculate_snr(audio, sr)
    clipping_pct, is_clipped = detect_clipping(audio)
    silence_segments, total_silence = detect_silence(audio, sr)

    sync_issues = []
    if annotations is not None:
        sync_issues = validate_annotation_sync(duration, annotations)
        cycle_durations = annotations['end'] - annotations['start']
        total_cycles = len(annotations)
        cycles_with_crackles = annotations['crackles'].sum()
        cycles_with_wheezes = annotations['wheezes'].sum()
        cycles_with_both = ((annotations['crackles'] == 1) & (annotations['wheezes'] == 1)).sum()
    else:
        cycle_durations = pd.Series()
        total_cycles = 0
        cycles_with_crackles = 0
        cycles_with_wheezes = 0
        cycles_with_both = 0

    results = {
        'filename': os.path.basename(wav_path),
        'sample_rate': sr,
        'channels': info.channels,
        'subtype': info.subtype,
        'duration_sec': duration,
        'num_samples': len(audio),
        'snr_db': snr_db,
        'clipping_pct': clipping_pct,
        'is_clipped': is_clipped,
        'num_silence_segments': len(silence_segments),
        'total_silence_sec': total_silence,
        'silence_pct': (total_silence / duration) * 100 if duration > 0 else 0,
        'num_cycles': total_cycles,
        'cycles_with_crackles': cycles_with_crackles,
        'cycles_with_wheezes': cycles_with_wheezes,
        'cycles_with_both': cycles_with_both,
        'mean_cycle_duration': cycle_durations.mean() if len(cycle_durations) > 0 else 0,
        'std_cycle_duration': cycle_durations.std() if len(cycle_durations) > 0 else 0,
        'min_cycle_duration': cycle_durations.min() if len(cycle_durations) > 0 else 0,
        'max_cycle_duration': cycle_durations.max() if len(cycle_durations) > 0 else 0,
        'sync_issues': len(sync_issues),
        'sync_issue_details': '; '.join(sync_issues) if sync_issues else 'None',
        'annotation_found': annotations is not None
    }

    return results, audio, sr, annotations

def find_project_root():
    """Return the repository root (the parent of scripts/)."""
    return Path(__file__).resolve().parent.parent

def main():
    """Run the quality analysis over the whole dataset and write the reports."""

    project_root = find_project_root()

    if project_root is None:
        print("\nError: could not locate the repository root")
        print("Create the project folder structure first (data/raw/, results/)")
        return

    print("="*80)
    print("Script 02: Audio quality assessment - FULL DATASET")
    print("="*80)
    print(f"\nProject: {project_root}")

    data_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'results' / 'quality_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"\nError: data folder not found at {data_dir}")
        print("\nPlease:")
        print("1. Copy the ICBHI dataset to data/raw/")
        print("2. Or create a symbolic link:")
        print('   mklink /D "data\\raw" "<path_to>\\ICBHI_final_database"')
        return

    print(f"Data: {data_dir}")
    print(f"Outputs: {output_dir}")

    wav_files = sorted(data_dir.glob('*.wav'))

    if len(wav_files) == 0:
        print(f"\nError: no WAV files found in {data_dir}")
        return

    print(f"\nFound {len(wav_files)} WAV files")
    print("\nStarting analysis...")

    all_results = []

    for wav_file in tqdm(wav_files, desc="Analyzing files"):
        txt_file = wav_file.with_suffix('.txt')

        try:
            results, audio, sr, annotations = analyze_audio_file(str(wav_file), txt_file)
            all_results.append(results)
        except Exception as e:
            print(f"\nWARNING: error processing {wav_file.name}: {str(e)}")
            continue

    print("\nGenerating reports...")
    df_results = pd.DataFrame(all_results)

    csv_path = output_dir / 'audio_quality_summary_FULL.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    report_path = output_dir / 'statistical_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SIGNAL QUALITY REPORT - FULL DATASET\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total files analyzed: {len(df_results)}\n")
        f.write(f"Files with annotations: {df_results['annotation_found'].sum()}\n")
        f.write(f"Files without annotations: {(~df_results['annotation_found']).sum()}\n\n")

        f.write("SAMPLE RATES\n")
        f.write("-"*80 + "\n")
        sr_counts = df_results['sample_rate'].value_counts().sort_index(ascending=False)
        for sr, count in sr_counts.items():
            f.write(f"  {sr:>6} Hz: {count:>4} files ({count/len(df_results)*100:>5.1f}%)\n")

        f.write("\n\nSIGNAL QUALITY (SNR)\n")
        f.write("-"*80 + "\n")
        f.write(f"  Mean: {df_results['snr_db'].mean():.2f} ± {df_results['snr_db'].std():.2f} dB\n")
        f.write(f"  Range: [{df_results['snr_db'].min():.2f}, {df_results['snr_db'].max():.2f}] dB\n")
        f.write(f"  Median: {df_results['snr_db'].median():.2f} dB\n")

        f.write("\n\nCLIPPING\n")
        f.write("-"*80 + "\n")
        f.write(f"  Files with clipping: {df_results['is_clipped'].sum()} ({df_results['is_clipped'].sum()/len(df_results)*100:.1f}%)\n")
        f.write(f"  Mean clipping: {df_results['clipping_pct'].mean():.4f}%\n")
        f.write(f"  Max clipping: {df_results['clipping_pct'].max():.4f}%\n")

        f.write("\n\nDURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"  Mean: {df_results['duration_sec'].mean():.2f} ± {df_results['duration_sec'].std():.2f} s\n")
        f.write(f"  Range: [{df_results['duration_sec'].min():.2f}, {df_results['duration_sec'].max():.2f}] s\n")
        f.write(f"  Total: {df_results['duration_sec'].sum()/3600:.2f} hours\n")

        f.write("\n\nRESPIRATORY CYCLES\n")
        f.write("-"*80 + "\n")
        f.write(f"  Total cycles: {df_results['num_cycles'].sum()}\n")
        f.write(f"  Cycles with crackles: {df_results['cycles_with_crackles'].sum()}\n")
        f.write(f"  Cycles with wheezes: {df_results['cycles_with_wheezes'].sum()}\n")
        f.write(f"  Cycles with both: {df_results['cycles_with_both'].sum()}\n")
        f.write(f"  Mean cycle duration: {df_results['mean_cycle_duration'].mean():.2f} ± {df_results['std_cycle_duration'].mean():.2f} s\n")

        f.write("\n\nPROBLEMATIC FILES\n")
        f.write("-"*80 + "\n")

        low_snr = df_results[df_results['snr_db'] < 8.0]
        if len(low_snr) > 0:
            f.write(f"\n  Files with SNR < 8 dB ({len(low_snr)} files):\n")
            for _, row in low_snr.iterrows():
                f.write(f"    - {row['filename']}: {row['snr_db']:.2f} dB\n")

        severe_clipping = df_results[df_results['clipping_pct'] > 1.0]
        if len(severe_clipping) > 0:
            f.write(f"\n  Files with clipping > 1% ({len(severe_clipping)} files):\n")
            for _, row in severe_clipping.iterrows():
                f.write(f"    - {row['filename']}: {row['clipping_pct']:.2f}%\n")

        sync_problems = df_results[df_results['sync_issues'] > 0]
        if len(sync_problems) > 0:
            f.write(f"\n  Files with synchronization issues ({len(sync_problems)} files):\n")
            for _, row in sync_problems.iterrows():
                f.write(f"    - {row['filename']}: {row['sync_issue_details']}\n")

    print(f"Report saved: {report_path}")

    print("Generating histograms...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Signal Quality Analysis - Full ICBHI Dataset', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    sr_counts = df_results['sample_rate'].value_counts().sort_index(ascending=False)
    ax.bar(range(len(sr_counts)), sr_counts.values, color='steelblue')
    ax.set_xticks(range(len(sr_counts)))
    ax.set_xticklabels([f'{int(sr/1000)}kHz' for sr in sr_counts.index], rotation=0)
    ax.set_ylabel('Number of files')
    ax.set_title('Sample Rate Distribution')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(df_results['snr_db'], bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(df_results['snr_db'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_results["snr_db"].mean():.1f} dB')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Frequency')
    ax.set_title('SNR Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.hist(df_results['clipping_pct'], bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Clipping (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Clipping Distribution')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(df_results['duration_sec'], bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Duration Distribution')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(df_results['num_cycles'], bins=range(0, df_results['num_cycles'].max()+2),
            color='teal', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Respiratory cycles per file')
    ax.set_ylabel('Frequency')
    ax.set_title('Cycles per File Distribution')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    event_counts = ['Crackles', 'Wheezes', 'Both']
    values_list = [
        df_results['cycles_with_crackles'].sum(),
        df_results['cycles_with_wheezes'].sum(),
        df_results['cycles_with_both'].sum()
    ]
    colors = ['orange', 'purple', 'red']
    ax.bar(event_counts, values_list, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of cycles')
    ax.set_title('Adventitious Events Detected')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / 'quality_histograms.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Histograms saved: {plot_path}")
    plt.close()

    # The "File:", "Severe clipping:" and "Low SNR:" lines below are parsed by script 03: keep both scripts in sync
    problems_path = output_dir / 'problematic_files.txt'
    with open(problems_path, 'w', encoding='utf-8') as f:
        f.write("PROBLEMATIC FILES REQUIRING ATTENTION\n")
        f.write("="*80 + "\n\n")

        problematic = df_results[
            (df_results['snr_db'] < 8.0) |
            (df_results['clipping_pct'] > 1.0) |
            (df_results['sync_issues'] > 0)
        ]

        f.write(f"Total problematic files: {len(problematic)}/{len(df_results)} ({len(problematic)/len(df_results)*100:.1f}%)\n\n")

        if len(problematic) > 0:
            f.write("FILE LIST:\n")
            f.write("-"*80 + "\n")
            for _, row in problematic.iterrows():
                f.write(f"\nFile: {row['filename']}\n")
                if row['snr_db'] < 8.0:
                    f.write(f"  Low SNR: {row['snr_db']:.2f} dB\n")
                if row['clipping_pct'] > 1.0:
                    f.write(f"  Severe clipping: {row['clipping_pct']:.2f}%\n")
                if row['sync_issues'] > 0:
                    f.write(f"  Synchronization issue: {row['sync_issue_details']}\n")

    print(f"Problematic file list: {problems_path}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nFiles analyzed: {len(df_results)}")
    print(f"Mean SNR: {df_results['snr_db'].mean():.2f} ± {df_results['snr_db'].std():.2f} dB")
    print(f"Files with clipping: {df_results['is_clipped'].sum()} ({df_results['is_clipped'].sum()/len(df_results)*100:.1f}%)")
    print(f"Total cycles: {df_results['num_cycles'].sum()}")
    print(f"Cycles with events: {(df_results['cycles_with_crackles'] + df_results['cycles_with_wheezes']).sum()}")

    print(f"\nAll results saved to: {output_dir}")
    print("\nQuality analysis COMPLETED")
    print("="*80)

    print("\nRECOMMENDATIONS:")
    if df_results['sample_rate'].nunique() > 1:
        print("  WARNING: heterogeneous sample rates detected - resampling to 16 kHz required")
    if df_results['is_clipped'].sum() > len(df_results) * 0.05:
        print(f"  WARNING: {df_results['is_clipped'].sum()/len(df_results)*100:.1f}% of files with clipping - apply correction")
    if len(problematic) > 0:
        print(f"  WARNING: {len(problematic)} problematic files - see problematic_files.txt")

    print("\nNext step: script 03 (preprocessing)")

if __name__ == '__main__':
    main()
