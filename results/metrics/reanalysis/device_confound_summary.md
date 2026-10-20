# Recording-device confound in the binary AUC (Script 35)

_2026-10-20T12:04:16 · 2733 cycles · 49 patients · B = 2000 · same-device pairs = 0.4991_

## Device-only baselines (patient-level 95 % CI)

| Baseline | AUC [95 % CI] | Note |
|---|---|---|
| Device prior (train) | 0.494 [0.432, 0.561] |  |
| Device prior (train, n<50 -> global) | 0.581 [0.504, 0.669] |  |
| Indicator: AKGC417L | 0.582 [0.506, 0.669] |  |
| ORACLE: test prevalence per device | 0.583 [0.506, 0.672] | oracle, not citable as a baseline |

## AUC decomposition (mean ± sd over seeds; patient-level 95 % CI)

| Model | Pooled [CI] | Same device [CI] | Cross-device | Device mean only [CI] |
|---|---|---|---|---|
| CNN-RespiratoryNet (FP32) | 0.572 ± 0.026 [0.515, 0.630] | 0.614 ± 0.033 [0.552, 0.673] | 0.529 ± 0.063 | 0.512 ± 0.040 [0.463, 0.566] |
| CNN-RespiratoryNet (INT8, FP32 classifier) | 0.572 ± 0.026 [0.517, 0.630] | 0.613 ± 0.033 [0.552, 0.671] | 0.531 ± 0.062 | 0.512 ± 0.040 [0.463, 0.566] |
| MobileNetV3-Small (ImageNet) | 0.543 ± 0.034 [0.477, 0.596] | 0.600 ± 0.024 [0.520, 0.653] | 0.487 ± 0.049 | 0.463 ± 0.042 [0.410, 0.530] |
| MobileNetV3-Small (scratch) | 0.528 ± 0.045 [0.484, 0.574] | 0.555 ± 0.035 [0.505, 0.602] | 0.501 ± 0.060 | 0.496 ± 0.058 [0.444, 0.543] |
| PANNs CNN6 (AudioSet) | 0.546 ± 0.023 [0.488, 0.611] | 0.575 ± 0.033 [0.515, 0.639] | 0.518 ± 0.015 | 0.494 ± 0.000 [0.432, 0.561] |
| PANNs CNN6 (AudioSet, tune) | 0.516 ± 0.027 [0.473, 0.562] | 0.538 ± 0.039 [0.499, 0.579] | 0.494 ± 0.026 | 0.478 ± 0.034 [0.420, 0.540] |
| CfC-Hybrid (mmRNN) | 0.501 ± 0.021 [0.423, 0.586] | 0.540 ± 0.020 [0.451, 0.634] | 0.462 ± 0.030 | 0.494 ± 0.000 [0.427, 0.557] |
| LSTM-Hybrid (h=103) | 0.479 ± 0.018 [0.401, 0.561] | 0.519 ± 0.016 [0.437, 0.608] | 0.438 ± 0.021 | 0.478 ± 0.034 [0.410, 0.540] |
| GRU-Hybrid (h=134) | 0.455 ± 0.018 [0.369, 0.546] | 0.488 ± 0.011 [0.404, 0.582] | 0.422 ± 0.027 | 0.432 ± 0.034 [0.366, 0.535] |
| CfC without mixed memory | 0.498 ± 0.001 [0.494, 0.500] | 0.499 ± 0.001 [0.497, 0.500] | 0.497 ± 0.002 | 0.537 ± 0.042 [0.488, 0.590] |
| Logistic regression (C selected by AUC) | 0.556 [0.493, 0.629] | 0.595 [0.521, 0.669] | 0.518 | 0.494 [0.432, 0.561] |
| Linear SVM (C selected by AUC) | 0.560 [0.495, 0.630] | 0.592 [0.517, 0.669] | 0.528 | 0.494 [0.432, 0.561] |

## Paired contrasts (same patient replicate)

| Contrast | Δ [95 % CI] | p boot |
|---|---|---|
| CNN pooled - B3 indicator | -0.0103 [-0.1321, +0.0982] | 0.846 |
| CNN pooled - B2 shrunk prior | -0.0090 [-0.1265, +0.0960] | 0.879 |
| CNN within - 0.5 | +0.1144 [+0.0517, +0.1731] | 0.001 |
| CNN within - CNN_INT8 within | +0.0016 [-0.0012, +0.0034] | 0.231 |
| CNN within - MNv3_IN within | +0.0147 [-0.0360, +0.0772] | 0.504 |
| CNN within - MNv3_SC within | +0.0599 [+0.0234, +0.0948] | 0.000 |
| CNN within - PANNs within | +0.0398 [-0.0228, +0.0910] | 0.220 |
| CNN within - PANNs_tune within | +0.0766 [+0.0239, +0.1192] | 0.001 |
| CNN within - CfC within | +0.0743 [-0.0135, +0.1505] | 0.111 |
| CNN within - LSTM within | +0.0957 [+0.0077, +0.1708] | 0.028 |
| CNN within - GRU within | +0.1261 [+0.0285, +0.2237] | 0.006 |
| CNN within - CfC_pure within | +0.1150 [+0.0537, +0.1738] | 0.001 |
| CNN pooled - logreg pooled | +0.0155 [-0.0315, +0.0603] | 0.532 |
| CNN within - logreg within | +0.0195 [-0.0359, +0.0743] | 0.492 |
| CNN pooled - linsvm pooled | +0.0118 [-0.0386, +0.0616] | 0.674 |
| CNN within - linsvm within | +0.0227 [-0.0352, +0.0808] | 0.456 |

## Same-device AUC: per-seed paired t-test and TOST (delta = 0.02, alpha = 0.05)

| Comparison | Mean Δ | 90 % CI | t p | TOST p | Equivalent |
|---|---|---|---|---|---|
| CNN - CNN_INT8 | +0.0016 | [-0.0006, +0.0038] | 0.194 | 0.000 | yes |
| CNN - MNv3_IN | +0.0147 | [-0.0180, +0.0473] | 0.392 | 0.373 | no |
| CNN - MNv3_SC | +0.0599 | [+0.0068, +0.1130] | 0.074 | 0.908 | no |
| CNN - PANNs | +0.0398 | [-0.0203, +0.0999] | 0.231 | 0.739 | no |
| CNN - PANNs_tune | +0.0766 | [+0.0126, +0.1407] | 0.063 | 0.934 | no |
| CNN - CfC | +0.0743 | [+0.0369, +0.1118] | 0.013 | 0.982 | no |
| CNN - LSTM | +0.0957 | [+0.0561, +0.1352] | 0.007 | 0.992 | no |
| CNN - GRU | +0.1261 | [+0.0972, +0.1551] | 0.001 | 0.999 | no |
| CNN - CfC_pure | +0.1150 | [+0.0829, +0.1471] | 0.002 | 0.998 | no |

_Post hoc analysis: report it as such._
