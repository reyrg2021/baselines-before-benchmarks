# Multi-Seed Results — FONIS MVP

Generated: 2026-06-08T10:51:30.087182

## Main table (test set)

| Model | Params | n seeds | ICBHI | Sensitivity | Specificity | Accuracy |
|---|---:|---:|---|---|---|---|
| **CNN-RespiratoryNet** | 556,500 | 5 | 0.5805 ± 0.0121 | 0.3712 ± 0.0258 | 0.7898 ± 0.0065 | 46.73% ± 2.14% |
| **CfC-Hybrid** | 491,492 | 5 | 0.5340 ± 0.0179 | 0.3154 ± 0.0277 | 0.7526 ± 0.0087 | 35.81% ± 3.45% |

## Sensitivity per class

| Model | Normal | Crackle | Wheeze | Both |
|---|---|---|---|---|
| CNN-RespiratoryNet | 0.511 ± 0.071 | 0.570 ± 0.188 | 0.213 ± 0.116 | 0.191 ± 0.127 | 
| CfC-Hybrid | 0.398 ± 0.068 | 0.313 ± 0.062 | 0.315 ± 0.066 | 0.236 ± 0.030 | 

## Statistical tests

### CNN vs CfC-Hybrid

- Seeds: [7, 42, 99, 123, 314]
- ICBHI CNN: [0.5834142405896523, 0.5611925579552464, 0.5944442589644194, 0.580344646013748, 0.5831772856988369]
- ICBHI CfC-Hybrid: [0.5162503671874696, 0.526290616258381, 0.5310945894975523, 0.5639783898456846, 0.5322396011375506]
- Mean diff (b − a): -0.0465
- **Paired t-test:** t=4.943, p=0.0078 (significant at α=0.05)
- **Wilcoxon signed-rank:** W=0.000, p=0.0625

