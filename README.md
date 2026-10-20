# Baselines Before Benchmarks

Code, cycle-level split, per-run predictions, and training logs for the article

> Rodriguez-Guillen R., Garcia-Garcia Y. **Baselines Before Benchmarks: A Patient-Level
> Evaluation of Compact, Pretrained, and Continuous-Time Models for Respiratory Sound
> Classification at the Edge.** Submitted to *PeerJ Computer Science*, 2026.

The study compares six configurations on the official ICBHI 2017 split with the official
ICBHI score, a binary Normal-versus-abnormal AUC, five seeds {42, 123, 7, 99, 314},
patient-level bootstrap intervals, a pre-specified equivalence margin, and reference
baselines that read no audio. The configurations are a compact CNN trained from scratch
(CNN-RespiratoryNet, with an INT8 version), a closed-form continuous-time hybrid
(CfC-Hybrid), MobileNetV3-Small (ImageNet and from scratch), and PANNs CNN6 (AudioSet,
two fine-tuning regimes). Every configuration is profiled on a Raspberry Pi 5.

The archived release of this repository is available on Zenodo:
**DOI: [to be added after the first release]**

---

## Contents

| Path | Content |
|---|---|
| `scripts/` | The analysis and training pipeline, numbered in execution order, plus shared modules and the figure scripts. |
| `events/` | The event-representation study of the CfC model (Section 4.6, Table 10). |
| `data/splits/strategy1_official/` | Cycle-level composition of the split used in the article (3350 / 635 / 2733 cycles). |
| `data/README.md` | Where to obtain ICBHI 2017, SPRSound, and the PANNs weights, and where to place them. |
| `results/master_results.json` | Consolidated per-seed metrics. Script 29 checks every array against this file before computing anything. |
| `results/metrics/` | Per-cycle test and validation probabilities of every run, and the outputs of the statistical scripts. |
| `logs/` | Training and analysis logs written by the scripts during the experiments. |
| `models/checkpoints/` | Empty. Trained weights are not distributed in this repository (see below). |
| `LICENSES/`, `THIRD_PARTY_NOTICES.md` | Licenses of the two vendored third-party files. |

### Not included

- **Audio and audio-derived features.** The ICBHI 2017 and SPRSound recordings are not
  redistributed. They are freely available from their providers; `data/README.md`
  gives the links and the expected folder layout. Segmented cycles, log-mel
  spectrograms, and the event file of the CfC study are regenerated from the audio by
  the scripts.
- **Trained checkpoints** (several hundred megabytes for the configurations in the article). The
  serialized INT8 model profiled on the Raspberry Pi 5 is distributed as Supplemental
  Data S15 of the article. Every metric of the article can be recomputed from the
  per-cycle predictions in `results/metrics/full_reeval/arrays/` without any checkpoint.
- **Pretrained weights** of PANNs CNN6 (`Cnn6_mAP=0.343.pth`), available from
  Kong et al. at [doi:10.5281/zenodo.3987831](https://doi.org/10.5281/zenodo.3987831)
  under CC BY 4.0. MobileNetV3-Small ImageNet weights are downloaded by `torchvision`.

---

## Three levels of reproduction

| Level | Needs | Reproduces |
|---|---|---|
| **A. From saved predictions** | This repository and Python. No audio, no GPU. | Tables 4, 5, 6, 7, and 9; the device decomposition (Supplemental Data S3 to S6); Figures 1 to 5. |
| **B. From audio** | ICBHI 2017 (and SPRSound for Table 15), a CUDA GPU. | Every table and figure, including training. |
| **C. On the edge device** | A Raspberry Pi 5 and the trained checkpoints. | Table 13 and the measurements behind Figure 5. |

Level A was verified on a clean copy of this repository: scripts 29, 35, 36, and 38
regenerate the statistical outputs of Supplemental Data S3 to S6 with the same values,
and the figure scripts regenerate Figures 1 to 5 pixel for pixel. The expected
differences are documented below.

---

## Installation

The experiments used two Conda environments on a Windows workstation with an NVIDIA
GeForce RTX 3060 Laptop GPU, and a separate Python installation on the Raspberry Pi 5.
The versions pinned in `requirements.txt` and `requirements-events.txt` are those of the
two environments; `results/metrics/full_reeval/full_reeval_results.json` records
Python 3.10.19 and PyTorch 2.10.0+cu126 for the main pipeline, and `logs/script37_log.txt`
records PyTorch 2.12.0+cu130 for the event study.

```bash
# Main pipeline (scripts/): Python 3.10
conda create -n bbb python=3.10
conda activate bbb
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Event study (events/): Python 3.13
conda create -n bbb-events python=3.13
conda activate bbb-events
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-events.txt
```

Level A only needs `numpy`, `pandas`, `scipy`, and `matplotlib`. It was verified with
Python 3.13.16, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, and matplotlib 3.11.2, which
shows that the statistical outputs do not depend on the exact library versions.

The Raspberry Pi 5 environment is described in `requirements-pi5.txt`.

---

## Level A: from saved predictions

Run from the `scripts/` folder. Each script writes a log `*_log.txt` at the repository
root and its outputs under `results/metrics/`.

```bash
cd scripts
python 29_reanalysis_test_probs.py     # Tables 4, 5, 7; S4: baselines, TOST, patient bootstrap, devices
python 35_device_confound_auc.py       # S5: pooled versus within-device AUC, device-only baselines
python 36_pretraining_contrasts.py     # Table 9; S6: pretraining and recipe contrasts
python 38_seed_fit.py                  # Table 6; S3: training fit per run, read from logs/
```

Script 29 first aligns every prediction array with `test_cycles.csv` row by row and
recomputes the official score and AUC of the 50 runs against `results/master_results.json`.
It stops with an error if any value differs.

Figures (set `OUT_DIR` to choose where each figure is written):

```bash
OUT_DIR=../results/figures/fig1 python rodri1_pipeline.py      # Figure 1 (writes rodri1.png)
OUT_DIR=../results/figures/fig2 python rodri5_arch.py          # Figure 2 (writes rodri5.png)
OUT_DIR=../results/figures/fig3 python rodri2_roc.py           # Figure 3 (writes rodri2.png)
OUT_DIR=../results/figures/fig4 python rodri3_confusion.py     # Figure 4 (writes rodri3.png)
python rodri5_pareto_auc.py --edge_json S11_edge_pi5_4threads.json --out_dir ../results/figures/fig5   # Figure 5
```

On Windows PowerShell, set the variable first: `$env:OUT_DIR = "..\results\figures\fig1"`.
The figure scripts keep the file names they had when the figures were produced; the
article renumbered them. Figure 5 reads the Raspberry Pi 5 measurements published as
Supplemental Data S11 (`S11_edge_pi5_4threads.json`).

**Expected differences in Level A.**

1. The patient-level AUC defined by diagnosis (one column of S4) needs
   `ICBHI_Challenge_diagnosis.txt`, which is part of the ICBHI download and is not
   redistributed here. Place it in `data/raw/` to obtain that column; without it the
   script reports `n/a` and every other value is unchanged.
2. In `seed_fit_runs.csv` (S3), the column that lists the candidate training records
   also named older history files of an earlier, non-reported split that are not part of
   this repository. The selected record, the epochs, and every metric are identical.
3. On a different platform or library version, unrounded values may differ in the last
   digit of double precision (relative differences around 1e-16). No reported value
   changes.

---

## Level B: from audio

Place the data as described in `data/README.md`. Run every script from `scripts/`,
in this order. Hyperparameters are the defaults of each script and match Table 3.

**1. Signal quality, preprocessing, and segmentation**

```bash
python 02_analisis_calidad_completo.py          # clipping and quality per recording (S1)
python 03_preprocessing_pipeline.py             # exclusion (clipping >= 10 %), filtering, RMS normalization
python 05_segment_respiratory_cycles.py         # one WAV per respiratory cycle
python 06_extract_mel_spectrograms.py           # 1 x 128 x 251 log-mel per cycle
python 06b_fix_feature_metadata.py              # repairs feature_metadata.csv written by 06
python 07_create_stratified_splits_v2.py        # data/splits/ (strategy1_official is the one used)
python 28_supplementary_excluded_recordings.py  # S1: the 22 excluded recordings
python verify_qc_footnote.py                    # checks the quality-control figures of Section 3.1
```

`data/splits/strategy1_official/` already contains the split used in the article. Script
07 regenerates it from the official ICBHI train/test list; the official list is used
without modification, and the training partition is subdivided 80/20 into training and
validation with no patient shared between them.

**2. Training (five seeds each)**

```bash
# CNN-RespiratoryNet and CfC-Hybrid
python 14_multiseed_runs.py --split strategy1_official --models cnn hybrid --seeds 42 123 7 99 314

# MobileNetV3-Small, ImageNet weights and from scratch (repeat for each seed S)
python 21_transfer_learning.py --backbone mobilenet_v3_small --split strategy1_official --seed S
python 21_transfer_learning.py --backbone mobilenet_v3_small --split strategy1_official --seed S \
       --no_pretrained --tag tl_mobilenet_v3_small_official_seedS

# PANNs CNN6, official and tune regimes (repeat for each seed S)
python 22_panns_transfer.py --split strategy1_official --seed S \
       --weights_path "../models/checkpoints/Cnn6_mAP=0.343.pth"
python 22_panns_transfer.py --split strategy1_official --seed S \
       --weights_path "../models/checkpoints/Cnn6_mAP=0.343.pth" \
       --lr 3e-5 --mixup_alpha 0 --tag panns_cnn6_tune_official_seedS

# Recurrent controls (LSTM, GRU, CfC without mixed memory) and recipe ablation
python 31_recurrent_baselines.py --all
python 34_recipe_ablation.py --all
```

The `--tag` of the from-scratch MobileNetV3 reproduces the checkpoint name used in the
article, which predates the regime token that script 21 now adds by default.

**3. Evaluation and consolidation**

```bash
python 18_full_reeval_multiseed.py --split strategy1_official --models cnn hybrid
python 21_transfer_learning.py --eval_only --checkpoint ../models/checkpoints/<run>.pth   # per MobileNetV3 run
python 22_panns_transfer.py   --eval_only --checkpoint ../models/checkpoints/<run>.pth   # per PANNs run
python 23_quantize_cnn_int8.py --seeds 42 123 7 99 314 --export_seed 42 --no_quantize_last   # INT8, classifier kept in FP32
python 33_int8_auc.py                      # INT8 AUC (S10)
python 33_int8_auc.py --quantize_last      # control with the classifier also quantized (Table 14)
python 24_consolidate_results.py           # results/master_results.json
```

**4. Analysis**

```bash
python 29_reanalysis_test_probs.py
python 30_val_fulltest_linear.py --stage val        # validation probabilities and thresholds (S9)
python 30_val_fulltest_linear.py --stage fulltest   # official test with the QC exclusions reinstated
python 30_val_fulltest_linear.py --stage linear     # logistic regression and linear SVM baselines
python 35_device_confound_auc.py
python 36_pretraining_contrasts.py
python 38_seed_fit.py
python 40_noise_robustness.py                       # Table 16 (S16)
python 42_diagnose_mel_reconstruction.py            # checks the input reconstruction used by 40
python 41_export_supplementary_artifacts.py         # S14 predictions and the S15 INT8 artifact
```

**5. Generalization to SPRSound (Table 15)**

```bash
python 25_sprsound_zeroshot_eval.py
python 26_sprsound_indomain_train.py
python 27_sprsound_zeroshot_intertest.py
```

**6. MACs**

```bash
python 19_flops_analysis.py
```

---

## Event representation of the CfC (Table 10)

The CfC is retrained on an irregular sequence of acoustic events per cycle, with the real
time between events as the time step. The code lives in `events/` and uses the
`bbb-events` environment.

```bash
cd events
python 18_event_extraction.py --out-dir ../data/events          # needs data/segments/ from step 1
python 37_cfc_events_rerun.py --events ../data/events/events.pkl \
       --split-dir ../data/splits/strategy1_official
```

Script 37 trains the six event configurations (with and without duration; real, unit,
and mean time steps) over the five seeds, under two checkpoint-selection rules, and
writes `results/metrics/events_rerun/` (S7). Scripts `19_event_dataloader.py` and
`20_cfc_event_training.py` are imported by script 37. The numbering of this folder
belongs to the original event-study project and is independent of `scripts/`.

---

## Level C: Raspberry Pi 5 (Table 13, Figure 5)

Hardware and software used in the article: Raspberry Pi 5 Model B Rev 1.1 with the
official Active Cooler and the CPU governor in `performance` mode. The measurement files
record the platform `Linux-6.18.33+rpt-rpi-2712-aarch64-with-glibc2.41`, Python 3.13.5,
and PyTorch 2.12.0+cpu.

Copy `scripts/` and the checkpoints listed in `resolve_checkpoint()` of script 32 to the
Pi. The article profiles the seed 123 checkpoint of each FP32 configuration and the
INT8 artifact exported from seed 42 (Supplemental Data S15).

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
cd scripts
python3 32_edge_benchmark_extended.py --num_threads_list 4 --num_iter 200 --num_warmup 50 \
        --stress_test_minutes 10 --cooldown_target_temp 0 --device_label RPi5_performance_v7_ext_4h
python3 32_edge_benchmark_extended.py --num_threads_list 1 --num_iter 200 --num_warmup 50 \
        --stress_test_minutes 10 --cooldown_target_temp 0 --device_label RPi5_performance_v7_ext_1h
```

These are the arguments recorded in the two output files (Supplemental Data S11).
`15_edge_benchmark.py` is the earlier benchmark that script 32 imports.

---

## Map of tables and figures

| Article | Script(s) | Supplemental Data |
|---|---|---|
| Table 1, split composition | 07 | S2 |
| Table 2, models compared | 09, 11a, 21, 22 | |
| Table 3, training protocol | defaults of 09, 11a, 21, 22 | |
| Table 4, main comparison and baselines | 18, 21, 22, 23, 33, 24, then 29, 30 (`linear`), 35 | S4, S5, S13, S14 |
| Table 5, paired comparisons | 29 | S4 |
| Table 6, training fit | 38 | S3 |
| Table 7, stratification by device | 29 | S4 |
| Table 8, per-class metrics | 18 | S13 |
| Table 9, pretraining and recipe contrasts | 34, 36 | S6 |
| Table 10, CfC on events | `events/` 18, 37 | S7 |
| Table 11, recurrent controls | 31 | S8 |
| Table 12, official score versus macro average | 18, 24 | S13 |
| Table 13, deployment cost | 32, 19 | S11 |
| Table 14, quantized classifier | 23, 33 | S10 |
| Table 15, SPRSound | 25, 26, 27 | S12 |
| Table 16, additive noise | 40, 42 | S16 |
| Tables 17 and 18 | figures taken from the cited works; limitations | |
| Figure 1, pipeline | `rodri1_pipeline.py` | |
| Figure 2, architecture | `rodri5_arch.py` | |
| Figure 3, ROC curves | `rodri2_roc.py` | S14 |
| Figure 4, confusion matrix | `rodri3_confusion.py` | S13 |
| Figure 5, accuracy-efficiency plane | `rodri5_pareto_auc.py` | S11 |
| Operating points on validation | 30 (`val`) | S9 |
| Quality control | 02, 03, 28, `verify_qc_footnote.py` | S1 |

---

## Provenance notes

- **Relation to the code that produced the results.** The scripts were written and
  run with Spanish comments and messages. For publication they were translated into
  English and their comments condensed, without changing any executable statement. Each
  file was checked against the version that produced the results by comparing abstract
  syntax trees: apart from comments, docstrings, and message text, the only differences
  are a consistent renaming of Spanish identifiers and output keys, and the replacement
  of absolute workstation paths by paths relative to the repository. On the same
  machine, the published scripts regenerate the statistical outputs of the original
  scripts with identical values and the figures with identical pixels. See `CHANGELOG.md`.
- **Logs and reports.** The training logs and the text of the result reports were
  translated in the same way. Every number, timestamp, key, and per-epoch line that
  script 38 parses is unchanged.
- **Numbering.** Numbers missing from `scripts/` (01, 04, 06c, 10, 12, 13, 16, 17, 20,
  39, 43, 44) belong to exploratory steps, earlier evaluations superseded by script 18,
  or manuscript tooling, none of which produces a reported result. Script 37 is in
  `events/`.
- **Earlier configurations.** `results/metrics/full_reeval/arrays/` also contains the
  predictions of a CfC-Transposed configuration evaluated during development and not
  reported in the article.
- **Training logs.** The training logs of MobileNetV3 and PANNs CNN6 were overwritten by
  later evaluation runs, as the article states; their training fit is therefore absent
  from Table 6.
- **Official split.** The official ICBHI 2017 train/test list assigns patients 156 and
  218 to both partitions (120 of the 2733 test cycles). The list is used unmodified, as
  in the literature it is compared with; `data/splits/` makes this visible.

---

## License

The code in this repository is released under the MIT License (`LICENSE`).
Two files are third-party and keep their own licenses (`THIRD_PARTY_NOTICES.md`):
`torch_cfc.py` (Apache License 2.0, Hasani et al.) and the CNN6 architecture in
`panns_cnn6.py` (MIT, Kong et al.). The split CSVs and the prediction arrays are
derived from the ICBHI 2017 database; please cite the database (references below) when
using them.

## Citation

If you use this code, please cite the article above and the archived software
(`CITATION.cff`). Data sources:

- Rocha BM, et al. A respiratory sound database for the development of automated
  classification. In: *Precision Medicine Powered by pHealth and Connected Health
  (ICBHI 2017)*, IFMBE Proceedings 66, 33-37. Springer (2018).
- Rocha BM, et al. An open access database for the evaluation of respiratory sound
  classification algorithms. *Physiological Measurement* 40, 035001 (2019). This is the
  reference requested by the database providers.
- Zhang Q, et al. SPRSound: Open-source SJTU Paediatric Respiratory Sound Database.
  *IEEE Transactions on Biomedical Circuits and Systems* 16(5), 867-881 (2022).
- Kong Q, et al. PANNs: Large-scale pretrained audio neural networks for audio pattern
  recognition. *IEEE/ACM Transactions on Audio, Speech, and Language Processing* 28,
  2880-2894 (2020).
- Hasani R, et al. Closed-form continuous-time neural networks. *Nature Machine
  Intelligence* 4, 992-1003 (2022).
