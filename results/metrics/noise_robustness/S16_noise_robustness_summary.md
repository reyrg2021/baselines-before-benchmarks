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
