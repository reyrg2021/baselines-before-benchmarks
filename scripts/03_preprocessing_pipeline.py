#!/usr/bin/env python3
"""
Script 03: Audio preprocessing pipeline for the ICBHI 2017 recordings.

Excludes recordings with severe clipping (>10 % of samples) and recordings flagged
with very low SNR, then resamples to 16 kHz, applies a zero-phase order-5
Butterworth band-pass (50-2000 Hz), softens moderate clipping (1-10 %) with a
tanh compression, and normalizes to a fixed RMS of 0.1 to remove gain differences
between devices and sessions. Per-file metadata records every transformation.

Inputs:  data/raw/*.wav, results/quality_analysis/problematic_files.txt (Script 02)
Outputs: data/processed/*.wav, data/processed/preprocessing_metadata.csv,
         data/processed/excluded_files.csv, data/processed/preprocessing_config.json
Usage:
    python scripts/03_preprocessing_pipeline.py
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal
from pathlib import Path
import warnings
from tqdm import tqdm
import json
warnings.filterwarnings('ignore')


class RespiratoryAudioPreprocessor:
    """
    Preprocessing pipeline for respiratory audio.

    A single object applies exactly the same transformation in every context
    (training, validation, inference).
    """

    def __init__(self,
                 target_sr=16000,
                 bandpass_lowcut=50,
                 bandpass_highcut=2000,
                 clipping_correction_threshold=1.0,
                 clipping_exclude_threshold=10.0,
                 normalize_rms_target=0.1):
        """
        Args:
            target_sr (int): Target sample rate in Hz. 16 kHz (8 kHz Nyquist) covers
                crackles and wheezes (up to ~2 kHz) with margin at low cost.
            bandpass_lowcut (float): Lower cutoff in Hz; removes movement, friction
                and structural vibration noise.
            bandpass_highcut (float): Upper cutoff in Hz; keeps the diagnostic
                content of crackles and wheezes.
            clipping_correction_threshold (float): Clipping percentage above which
                tanh correction is applied.
            clipping_exclude_threshold (float): Clipping percentage above which the
                file is excluded as unrecoverable.
            normalize_rms_target (float): Target RMS on the [-1, 1] scale; leaves
                headroom against saturation.
        """
        self.target_sr = target_sr
        self.bandpass_lowcut = bandpass_lowcut
        self.bandpass_highcut = bandpass_highcut
        self.clipping_correction_threshold = clipping_correction_threshold
        self.clipping_exclude_threshold = clipping_exclude_threshold
        self.normalize_rms_target = normalize_rms_target

        self.stats = {
            'files_processed': 0,
            'files_excluded': 0,
            'files_corrected_clipping': 0,
            'files_resampled': 0
        }

    def butter_bandpass(self, lowcut, highcut, sr, order=5):
        """
        Design a Butterworth band-pass filter and return its (b, a) coefficients.

        Butterworth gives a maximally flat passband. Order 5 balances selectivity
        and numerical stability. Cutoffs are normalized by the Nyquist frequency.
        """
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(self, audio, sr):
        """
        Apply the band-pass filter with filtfilt (forward-backward).

        Zero-phase filtering preserves the timing of events such as crackles and
        doubles the effective filter order; filtfilt pads the edges to limit
        boundary artifacts. Output has the same length as the input.
        """
        b, a = self.butter_bandpass(self.bandpass_lowcut, self.bandpass_highcut, sr)
        filtered_audio = signal.filtfilt(b, a, audio)
        return filtered_audio

    def detect_clipping(self, audio, threshold=0.99):
        """
        Estimate clipping severity.

        The signal is peak-normalized to [-1, 1] and samples with |x| > threshold
        are counted. Returns (clipping_percentage, is_severely_clipped), the latter
        being True above the exclusion threshold.
        """
        audio_normalized = audio / (np.max(np.abs(audio)) + 1e-10)

        clipped_samples = np.sum(np.abs(audio_normalized) > threshold)
        clipping_percentage = (clipped_samples / len(audio)) * 100

        is_severely_clipped = clipping_percentage > self.clipping_exclude_threshold

        return clipping_percentage, is_severely_clipped

    def correct_clipping(self, audio):
        """
        Soften moderate clipping with a tanh compression.

        tanh is near-linear for small values and saturates smoothly for large
        ones, so it rounds the clipped plateaus and reduces the spurious
        high-frequency harmonics they introduce. Lost information is not
        recovered.
        """
        max_val = np.max(np.abs(audio))
        if max_val < 1e-10:  # silent signal
            return audio

        audio_normalized = audio / max_val

        # The 0.95 factor limits compression to the extreme peaks
        audio_corrected = np.tanh(audio_normalized / 0.95)

        # Rescale to approximately the original amplitude
        audio_corrected = audio_corrected * max_val * 0.95

        return audio_corrected

    def normalize_rms(self, audio, target_rms=None):
        """
        Scale the signal to a target RMS (default: normalize_rms_target).

        Equalizing RMS removes gain differences between devices, sessions and
        operators; RMS is less sensitive to isolated peaks than the maximum.
        """
        if target_rms is None:
            target_rms = self.normalize_rms_target

        current_rms = np.sqrt(np.mean(audio**2))

        if current_rms < 1e-10:
            return audio

        scaling_factor = target_rms / current_rms
        audio_normalized = audio * scaling_factor

        return audio_normalized

    def process_single_file(self, wav_path, original_sr=None):
        """
        Run the full pipeline on one WAV file.

        Order: load, detect clipping on the original signal (exclude if severe),
        resample to target_sr (the filter is designed for that rate), band-pass,
        correct moderate clipping, and finally normalize RMS so that all outputs
        share the same energy level.

        Returns (processed_audio, metadata), or (None, metadata) if excluded.
        The metadata records the transformations applied, for reproducibility.
        """
        audio, sr = sf.read(wav_path)

        # Stereo to mono by channel averaging
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        metadata = {
            'filename': os.path.basename(wav_path),
            'original_sr': sr,
            'original_duration': len(audio) / sr,
            'excluded': False,
            'exclusion_reason': None,
            'transformations_applied': []
        }

        # Clipping is measured on the original signal, before any processing
        clipping_pct, is_severely_clipped = self.detect_clipping(audio)
        metadata['original_clipping_pct'] = clipping_pct

        if is_severely_clipped:
            metadata['excluded'] = True
            metadata['exclusion_reason'] = f'Severe clipping: {clipping_pct:.2f}%'
            self.stats['files_excluded'] += 1
            return None, metadata

        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
            metadata['transformations_applied'].append(f'Resampled to {self.target_sr} Hz')
            self.stats['files_resampled'] += 1

        audio = self.apply_bandpass_filter(audio, sr)
        metadata['transformations_applied'].append(f'Bandpass filter {self.bandpass_lowcut}-{self.bandpass_highcut} Hz')

        if clipping_pct > self.clipping_correction_threshold:
            audio = self.correct_clipping(audio)
            metadata['transformations_applied'].append(f'Clipping correction applied ({clipping_pct:.2f}%)')
            self.stats['files_corrected_clipping'] += 1

        audio = self.normalize_rms(audio)
        metadata['transformations_applied'].append(f'RMS normalization to {self.normalize_rms_target}')

        metadata['processed_duration'] = len(audio) / sr
        metadata['processed_sr'] = sr

        self.stats['files_processed'] += 1

        return audio, metadata

    def get_statistics(self):
        """Return the processing counters."""
        return self.stats


def load_problematic_files_list(results_dir, clipping_threshold=10.0):
    """
    Read the quality report of Script 02 and return the set of files to exclude.

    A file is excluded if its reported clipping is >= clipping_threshold (files
    with lower clipping are kept and corrected in software), if its clipping value
    cannot be parsed (conservative choice), or if its reported SNR is below 5 dB.
    The original report file is left untouched.
    """
    problematic_file = results_dir / 'problematic_files.txt'

    if not problematic_file.exists():
        print("Warning: problematic_files.txt not found")
        print("   Proceeding without a predefined exclusion list.")
        print("   The preprocessor will rely on on-the-fly detection only.")
        return set()

    exclude_files = set()
    total_problematic = 0
    filtered_out = 0

    with open(problematic_file, 'r', encoding='utf-8') as f:
        current_filename = None

        for line in f:
            # Line format: "File: name.wav"
            if '.wav' in line and 'File:' in line:
                total_problematic += 1
                filename = line.split('File:')[1].strip().split()[0]
                current_filename = filename

            # Line format: "  Severe clipping: X.XX%"
            elif current_filename and 'Severe clipping:' in line:
                try:
                    clipping_str = line.split('clipping:')[1].split('%')[0].strip()
                    clipping_pct = float(clipping_str)

                    if clipping_pct >= clipping_threshold:
                        exclude_files.add(current_filename)
                    else:
                        # Moderate clipping: kept and corrected in software
                        filtered_out += 1

                except (ValueError, IndexError) as e:
                    print(f"Warning: could not parse clipping for {current_filename}, excluding it as a precaution")
                    exclude_files.add(current_filename)

                current_filename = None

            # Line format: "  Low SNR: X.XX dB"; excluded below 5 dB even without clipping
            elif current_filename and 'Low SNR:' in line:
                try:
                    snr_str = line.split('SNR:')[1].split('dB')[0].strip()
                    snr_db = float(snr_str)

                    if snr_db < 5.0:
                        exclude_files.add(current_filename)

                except (ValueError, IndexError):
                    pass  # unparseable SNR: not excluded by this criterion

                current_filename = None

    return exclude_files


def main():
    """Locate the project, process every valid WAV file, and save outputs and metadata."""
    print("="*80)
    print("SCRIPT 03: PREPROCESSING PIPELINE")
    print("="*80)
    print()

    # Repository root: the parent of scripts/
    project_root = Path(__file__).resolve().parent.parent

    if project_root is None:
        print("Error: could not locate the repository root")
        print("   Run the script from the project directory.")
        return

    print(f"Project found: {project_root}")
    print()

    data_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'data' / 'processed'
    results_dir = project_root / 'results' / 'quality_analysis'

    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Error: data directory not found at {data_dir}")
        return

    # The preprocessor is created first so its exclusion threshold is used
    # when loading the list of problematic files
    print("Initializing preprocessor with configuration:")
    print(f"   - Target sample rate: 16,000 Hz")
    print(f"   - Bandpass filter: 50-2,000 Hz")
    print(f"   - Clipping correction threshold: 1.0%")
    print(f"   - Clipping exclusion threshold: 10.0%")
    print(f"   - RMS normalization target: 0.1")
    print()

    preprocessor = RespiratoryAudioPreprocessor(
        target_sr=16000,
        bandpass_lowcut=50,
        bandpass_highcut=2000,
        clipping_correction_threshold=1.0,
        clipping_exclude_threshold=10.0,
        normalize_rms_target=0.1
    )

    print("Loading and filtering the list of problematic files...")
    print(f"   Exclusion criterion: Clipping > {preprocessor.clipping_exclude_threshold}%")
    exclude_files = load_problematic_files_list(results_dir,
                                                clipping_threshold=preprocessor.clipping_exclude_threshold)
    print(f"   Files flagged for exclusion (severe clipping): {len(exclude_files)}")
    print()

    wav_files = sorted(data_dir.glob('*.wav'))
    print(f"WAV files found: {len(wav_files)}")
    print()

    print("Processing files...")
    print()

    all_metadata = []
    processed_count = 0
    excluded_count = 0

    for wav_file in tqdm(wav_files, desc="Processing"):
        filename = wav_file.name

        if filename in exclude_files:
            metadata = {
                'filename': filename,
                'excluded': True,
                'exclusion_reason': 'Listed in problematic_files.txt (script 02)',
                'transformations_applied': []
            }
            all_metadata.append(metadata)
            excluded_count += 1
            continue

        try:
            processed_audio, metadata = preprocessor.process_single_file(str(wav_file))
            all_metadata.append(metadata)

            if metadata['excluded']:
                excluded_count += 1
            else:
                output_path = output_dir / filename
                sf.write(output_path, processed_audio, preprocessor.target_sr)
                processed_count += 1

        except Exception as e:
            print(f"\nWarning: error processing {filename}: {str(e)}")
            metadata = {
                'filename': filename,
                'excluded': True,
                'exclusion_reason': f'Processing error: {str(e)}',
                'transformations_applied': []
            }
            all_metadata.append(metadata)
            excluded_count += 1

    print()
    print("="*80)
    print("PROCESSING COMPLETED")
    print("="*80)
    print()

    stats = preprocessor.get_statistics()

    print(f"PROCESSING STATISTICS:")
    print(f"   Total files in dataset: {len(wav_files)}")
    print(f"   Files processed successfully: {processed_count}")
    print(f"   Files excluded: {excluded_count}")
    print(f"   Files that required resampling: {stats['files_resampled']}")
    print(f"   Files with clipping correction: {stats['files_corrected_clipping']}")
    print()

    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = output_dir / 'preprocessing_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")

    excluded_df = metadata_df[metadata_df['excluded'] == True]
    excluded_path = output_dir / 'excluded_files.csv'
    excluded_df.to_csv(excluded_path, index=False)
    print(f"Excluded-file list saved to: {excluded_path}")

    # Preprocessor configuration saved for reproducibility
    config = {
        'target_sr': preprocessor.target_sr,
        'bandpass_lowcut': preprocessor.bandpass_lowcut,
        'bandpass_highcut': preprocessor.bandpass_highcut,
        'clipping_correction_threshold': preprocessor.clipping_correction_threshold,
        'clipping_exclude_threshold': preprocessor.clipping_exclude_threshold,
        'normalize_rms_target': preprocessor.normalize_rms_target
    }
    config_path = output_dir / 'preprocessing_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    print()
    print(f"Processed files saved to: {output_dir}")
    print()
    print("Preprocessing pipeline completed successfully")
    print("="*80)


if __name__ == '__main__':
    main()
