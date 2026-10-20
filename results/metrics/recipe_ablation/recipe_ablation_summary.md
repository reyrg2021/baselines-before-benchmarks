# Training-recipe ablation (Script 34)

_2026-09-16T10:48:56 · Arm A preset `moderate` · Arm B without augmentation or mixup_


| Configuration | ICBHI | Se off. | Sp off. | AUC | Se Wheeze | Se Both |
|---|---|---|---|---|---|---|
| CNN-RespiratoryNet + augmentation | 0.247 ± 0.152 | 0.204 ± 0.178 | 0.290 ± 0.256 | 0.522 ± 0.063 | 0.169 ± 0.287 | 0.371 ± 0.406 |
| MobileNetV3-ImageNet without augmentation or mixup | 0.439 ± 0.049 | 0.076 ± 0.033 | 0.802 ± 0.123 | 0.512 ± 0.033 | 0.114 ± 0.076 | 0.106 ± 0.082 |

**Paired per-seed comparison against the original recipe:**

| Arm | Reference | Metric | Reference | New | Δ | CI 90 % | t p | TOST p | Equivalent |
|---|---|---|---|---|---|---|---|---|---|
| cnn_aug | CNN-RespiratoryNet (FP32) | icbhi | 0.460 | 0.247 | -0.2126 | [-0.3626, -0.0627] | 0.039 | 0.974 | no |
| cnn_aug | CNN-RespiratoryNet (FP32) | auc | 0.572 | 0.522 | -0.0494 | [-0.1289, +0.0302] | 0.256 | 0.762 | no |
| mnv3_noaug | MobileNetV3-Small (ImageNet) | icbhi | 0.462 | 0.439 | -0.0232 | [-0.0628, +0.0164] | 0.280 | 0.564 | no |
| mnv3_noaug | MobileNetV3-Small (ImageNet) | auc | 0.543 | 0.512 | -0.0310 | [-0.0588, -0.0031] | 0.077 | 0.775 | no |
