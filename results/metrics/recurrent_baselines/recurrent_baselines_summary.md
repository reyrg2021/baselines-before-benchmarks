# Recurrent baselines (Script 31)

_2026-09-17T11:25:26 · target 491,492 parameters_


| Cell | Params | h | Val ICBHI | Val AUC | Test ICBHI | Se off. | Sp off. | Test AUC | Se Wheeze | Se Both | Best ep. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cfcpure | 212,708 | 64 | 0.500 ± 0.000 | 0.503 ± 0.007 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.498 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 9.0 ± 6.2 |
| gru | 490,380 | 134 | 0.485 ± 0.049 | 0.665 ± 0.039 | 0.295 ± 0.029 | 0.374 ± 0.057 | 0.215 ± 0.094 | 0.455 ± 0.018 | 0.320 ± 0.039 | 0.220 ± 0.050 | 60.0 ± 7.5 |
| lstm | 489,084 | 103 | 0.517 ± 0.014 | 0.664 ± 0.029 | 0.353 ± 0.033 | 0.333 ± 0.070 | 0.372 ± 0.107 | 0.479 ± 0.018 | 0.325 ± 0.054 | 0.186 ± 0.046 | 55.0 ± 4.4 |
| _CfC-Hybrid (mmRNN)_ | 491,492 | | | | 0.351 ± 0.030 | | | 0.501 ± 0.021 | | | |
| _CNN-RespiratoryNet_ | 556,500 | | | | 0.460 ± 0.017 | | | 0.572 ± 0.026 | | | |

**Paired per-seed differences (cell − reference):**

| Cell | vs | Metric | Δ | CI 90 % | t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| cfcpure | CfC-Hybrid (mmRNN) | icbhi | +0.1488 | [+0.1199, +0.1777] | 0.000 | 0.062 |
| cfcpure | CfC-Hybrid (mmRNN) | auc | -0.0025 | [-0.0224, +0.0174] | 0.802 | 1.000 |
| cfcpure | CNN-RespiratoryNet | icbhi | +0.0402 | [+0.0240, +0.0564] | 0.006 | 0.062 |
| cfcpure | CNN-RespiratoryNet | auc | -0.0733 | [-0.0994, -0.0472] | 0.004 | 0.062 |
| gru | CfC-Hybrid (mmRNN) | icbhi | -0.0564 | [-0.0886, -0.0242] | 0.020 | 0.062 |
| gru | CfC-Hybrid (mmRNN) | auc | -0.0457 | [-0.0780, -0.0135] | 0.039 | 0.125 |
| gru | CNN-RespiratoryNet | icbhi | -0.1650 | [-0.1851, -0.1448] | 0.000 | 0.062 |
| gru | CNN-RespiratoryNet | auc | -0.1166 | [-0.1481, -0.0850] | 0.001 | 0.062 |
| lstm | CfC-Hybrid (mmRNN) | icbhi | +0.0016 | [-0.0052, +0.0084] | 0.643 | 1.000 |
| lstm | CfC-Hybrid (mmRNN) | auc | -0.0224 | [-0.0295, -0.0153] | 0.003 | 0.062 |
| lstm | CNN-RespiratoryNet | icbhi | -0.1070 | [-0.1407, -0.0733] | 0.002 | 0.062 |
| lstm | CNN-RespiratoryNet | auc | -0.0932 | [-0.1167, -0.0697] | 0.001 | 0.062 |
