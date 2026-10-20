# Full Re-Evaluation Multi-Seed — Results for the paper

_Generated: 2026-06-10 23:39:43_  
_Device: cuda · PyTorch 2.10.0+cu126_

## 0. ICBHI Score — OFFICIAL vs Macro (the paper number)

The **official** score (ICBHI 2017) is the one comparable with the literature (RespireNet, Pham 2022, etc.). The **macro** score is the old number from scripts 09/11a/14/16; it is shown only to measure the inflation. The official score here is computed from fresh inference and must match that of Script 17 (which computes it from the saved counts).

| Model | ICBHI OFFICIAL | Official Se | Official Sp | ICBHI Macro | Inflation (macro−official) |
|---|---|---|---|---|---|
| CNN-RespiratoryNet | **0.460 ± 0.017** | 0.408 ± 0.063 | 0.511 ± 0.071 | 0.581 ± 0.012 | +0.121 |
| CfC-Hybrid | **0.351 ± 0.030** | 0.304 ± 0.042 | 0.398 ± 0.068 | 0.534 ± 0.018 | +0.183 |

> **CRITICAL for the paper:** use the OFFICIAL column in the literature comparison table, NOT the macro. The inflation quantifies how much the score rose because of the one-vs-rest average over minority classes.

## 1. Per-class metrics (4-class) — mean ± std

| Model | Class | Sensitivity | Specificity | Precision (PPV) | F1 |
|---|---|---|---|---|---|
| CNN-RespiratoryNet | Normal | 0.511 ± 0.071 | 0.622 ± 0.094 | 0.651 ± 0.040 | 0.569 ± 0.040 |
| CNN-RespiratoryNet | Crackle | 0.570 ± 0.188 | 0.689 ± 0.155 | 0.371 ± 0.033 | 0.436 ± 0.045 |
| CNN-RespiratoryNet | Wheeze | 0.213 ± 0.116 | 0.892 ± 0.054 | 0.223 ± 0.043 | 0.207 ± 0.099 |
| CNN-RespiratoryNet | Both | 0.191 ± 0.127 | 0.957 ± 0.031 | 0.196 ± 0.018 | 0.170 ± 0.087 |
| CfC-Hybrid | Normal | 0.398 ± 0.068 | 0.566 ± 0.073 | 0.554 ± 0.018 | 0.460 ± 0.052 |
| CfC-Hybrid | Crackle | 0.313 ± 0.062 | 0.708 ± 0.063 | 0.250 ± 0.034 | 0.276 ± 0.038 |
| CfC-Hybrid | Wheeze | 0.315 ± 0.066 | 0.817 ± 0.015 | 0.214 ± 0.026 | 0.254 ± 0.040 |
| CfC-Hybrid | Both | 0.236 ± 0.030 | 0.919 ± 0.020 | 0.139 ± 0.023 | 0.173 ± 0.020 |

> **Note for Discussion:** the Precision (PPV) column is the one that was missing. Check the Crackle trade-off: if sensitivity is high but precision is low, the model over-predicts that class. This is defensible in triage screening (over-referral is preferable), but it must be reported.

## 2. Binary Normal-vs-Abnormal screening — mean ± std

| Model | Se | Sp | PPV | NPV | F1 | Accuracy | AUC-ROC |
|---|---|---|---|---|---|---|---|
| CNN-RespiratoryNet | 0.622 ± 0.094 | 0.511 ± 0.071 | 0.484 ± 0.019 | 0.651 ± 0.040 | 0.542 ± 0.040 | 0.558 ± 0.022 | 0.572 ± 0.026 |
| CfC-Hybrid | 0.566 ± 0.073 | 0.398 ± 0.068 | 0.409 ± 0.014 | 0.554 ± 0.018 | 0.474 ± 0.031 | 0.469 ± 0.017 | 0.501 ± 0.021 |

> Binary screening is the actual clinical decision of rural triage (normal vs refer). Binary sensitivity is the metric that weighs most: minimize false negatives (do not miss a patient who should be referred).

## 3. Paired McNemar: CNN vs CfC-Hybrid

Compares hits/errors sample by sample (more powerful than comparing ICBHI means with n=5 seeds).

**Aggregate (discordant pairs summed over all seeds):**

- b (only CNN correct): 3287
- c (only Hybrid correct): 1794
- χ² (Edwards): 438.1153316276324
- p-value: 0.0000 (SIGNIFICANT)

**Per seed:**

| Seed | b (CNN) | c (Hybrid) | χ² | p-value |
|---|---|---|---|---|
| 7 | 635 | 372 | 68.167 | 0.0000 |
| 42 | 722 | 337 | 139.241 | 0.0000 |
| 99 | 664 | 353 | 94.494 | 0.0000 |
| 123 | 496 | 323 | 36.122 | 0.0000 |
| 314 | 770 | 409 | 109.924 | 0.0000 |
