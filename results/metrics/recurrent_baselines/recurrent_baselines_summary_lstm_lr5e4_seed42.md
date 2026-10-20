# Recurrent baselines (Script 31)

_2026-09-07T07:09:43 · target 491,492 parameters_


| Cell | Params | h | Val ICBHI | Val AUC | Test ICBHI | Se off. | Sp off. | Test AUC | Se Wheeze | Se Both | Best ep. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm | 489,084 | 103 | 0.508 ± 0.000 | 0.728 ± 0.000 | 0.362 ± 0.000 | 0.216 ± 0.000 | 0.509 ± 0.000 | 0.479 ± 0.000 | 0.350 ± 0.000 | 0.187 ± 0.000 | 52.0 ± 0.0 |
| _CfC-Hybrid (mmRNN)_ | 491,492 | | | | 0.351 ± 0.030 | | | 0.501 ± 0.021 | | | |
| _CNN-RespiratoryNet_ | 556,500 | | | | 0.460 ± 0.017 | | | 0.572 ± 0.026 | | | |
