# S16 - Robustness to additive noise

Generated on 2026-09-18.

CNN-RespiratoryNet evaluated on the official ICBHI 2017 test set with white Gaussian noise added before the 50-2000 Hz front end, at several signal-to-noise ratios. The five checkpoints of the article are reused without retraining, and the noise realization is identical across seeds, so the comparison stays paired.

Signal power is measured over the non-padded samples of each cycle. Values are the mean over five seeds, with the sample standard deviation.

| SNR | ICBHI | Se | Sp | AUC_bin |
|---|---|---|---|---|
| clean | 0.460 ± 0.017 | 0.408 | 0.511 | 0.572 ± 0.026 |
| 20 | 0.483 ± 0.022 | 0.113 | 0.852 | 0.587 ± 0.030 |
| 10 | 0.485 ± 0.015 | 0.059 | 0.911 | 0.576 ± 0.039 |
| 5 | 0.487 ± 0.009 | 0.029 | 0.945 | 0.556 ± 0.054 |
| 0 | 0.490 ± 0.008 | 0.010 | 0.970 | 0.536 ± 0.042 |

## Per-class sensitivity

| SNR | Normal | Crackle | Wheeze | Both |
|---|---|---|---|---|
| clean | 0.511 | 0.570 | 0.213 | 0.191 |
| 20 | 0.852 | 0.128 | 0.112 | 0.043 |
| 10 | 0.911 | 0.066 | 0.065 | 0.007 |
| 5 | 0.945 | 0.029 | 0.039 | 0.004 |
| 0 | 0.970 | 0.001 | 0.028 | 0.001 |

The clean condition rebuilds the log-mel from the segmented WAV instead of reading the precomputed array, and reproduces the figures of Table 4, which validates the reconstruction.


## Paired contrasts against the clean condition (n = 5 seeds)

| SNR | ΔICBHI | p | ΔSe | p | ΔAUC | p |
|---|---|---|---|---|---|---|
| 20 dB | +0.023 | 0.025 | −0.296 | 0.008 | +0.015 | 0.517 |
| 10 dB | +0.026 | 0.016 | −0.349 | 0.003 | +0.004 | 0.843 |
| 5 dB | +0.028 | 0.019 | −0.379 | 0.0005 | −0.016 | 0.593 |
| 0 dB | +0.031 | 0.031 | −0.399 | 0.0002 | −0.036 | 0.251 |

Two-sided paired t-test over the five seeds. The rise in the ICBHI score and the
fall in sensitivity are supported at every level; no AUC difference is detected.

Seed 42 reaches Se = 0.000 and Sp = 1.000 at 5 and at 0 dB, that is, it becomes
the always-Normal classifier and obtains an ICBHI score of exactly 0.500, while
its binary AUC at 5 dB, 0.613, is the highest of the five runs.