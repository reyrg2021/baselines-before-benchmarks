# Per-seed fit to the training set (Script 38)

_2026-10-20T12:05:44 · degenerate-run threshold: training accuracy <= 25 %_

## Per run

| Model | Seed | Source | Verified | Epochs | Selected ep. | Final train acc | Train acc at selected ep. | Final train loss | Val AUC | Test ICBHI | Test AUC | Degenerate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN-RespiratoryNet (FP32) | 42 | log | yes | 17 | 2 | 23.8 % | 28.3 % | 1.385 | 0.595 | 0.452 | 0.605 | yes |
| CNN-RespiratoryNet (FP32) | 123 | log | yes | 40 | 25 | 70.6 % | 57.1 % | 0.597 | 0.710 | 0.443 | 0.545 | no |
| CNN-RespiratoryNet (FP32) | 7 | log | yes | 80 | 69 | 69.2 % | 66.2 % | 0.664 | 0.717 | 0.447 | 0.548 | no |
| CNN-RespiratoryNet (FP32) | 99 | log | yes | 43 | 28 | 76.9 % | 61.7 % | 0.510 | 0.686 | 0.478 | 0.592 | no |
| CNN-RespiratoryNet (FP32) | 314 | log | yes | 64 | 49 | 80.2 % | 59.3 % | 0.481 | 0.743 | 0.477 | 0.569 | no |
| CfC-Hybrid | 42 | log | yes | 59 | 39 | 44.8 % | 42.9 % | 1.201 | 0.655 | 0.311 | 0.526 | no |
| CfC-Hybrid | 123 | log | yes | 80 | 66 | 60.2 % | 60.3 % | 0.907 | 0.721 | 0.393 | 0.515 | no |
| CfC-Hybrid | 7 | log | yes | 53 | 33 | 48.0 % | 40.9 % | 1.136 | 0.658 | 0.339 | 0.481 | no |
| CfC-Hybrid | 99 | log | yes | 64 | 44 | 54.6 % | 49.6 % | 1.016 | 0.699 | 0.365 | 0.505 | no |
| CfC-Hybrid | 314 | log | yes | 80 | 66 | 47.0 % | 46.1 % | 1.174 | 0.700 | 0.348 | 0.478 | no |
| LSTM-Hybrid | 42 | log | yes | 72 | 52 | 45.9 % | 41.6 % | 1.208 | n/a | n/a | n/a | no |
| LSTM-Hybrid | 123 | log | yes | 76 | 56 | 52.8 % | 49.5 % | 1.026 | n/a | n/a | n/a | no |
| LSTM-Hybrid | 7 | log | yes | 71 | 51 | 49.7 % | 43.9 % | 1.142 | n/a | n/a | n/a | no |
| LSTM-Hybrid | 99 | log | yes | 74 | 54 | 49.0 % | 45.3 % | 1.121 | n/a | n/a | n/a | no |
| LSTM-Hybrid | 314 | log | yes | 80 | 62 | 46.1 % | 46.2 % | 1.169 | n/a | n/a | n/a | no |
| GRU-Hybrid | 42 | log | yes | 70 | 50 | 48.5 % | 44.3 % | 1.117 | n/a | n/a | n/a | no |
| GRU-Hybrid | 123 | log | yes | 80 | 64 | 47.8 % | 45.4 % | 1.043 | n/a | n/a | n/a | no |
| GRU-Hybrid | 7 | log | yes | 80 | 70 | 46.9 % | 46.4 % | 1.120 | n/a | n/a | n/a | no |
| GRU-Hybrid | 99 | log | yes | 77 | 57 | 57.3 % | 54.0 % | 0.959 | n/a | n/a | n/a | no |
| GRU-Hybrid | 314 | log | yes | 79 | 59 | 44.1 % | 44.1 % | 1.189 | n/a | n/a | n/a | no |
| CfC without mixed memory | 42 | log | yes | 25 | 5 | 25.5 % | 24.4 % | 1.383 | n/a | n/a | n/a | no |
| CfC without mixed memory | 123 | log | yes | 21 | 1 | 26.7 % | 24.8 % | 1.384 | n/a | n/a | n/a | no |
| CfC without mixed memory | 7 | log | yes | 36 | 16 | 23.2 % | 25.3 % | 1.386 | n/a | n/a | n/a | yes |
| CfC without mixed memory | 99 | log | yes | 34 | 14 | 23.3 % | 24.0 % | 1.386 | n/a | n/a | n/a | yes |
| CfC without mixed memory | 314 | log | yes | 29 | 9 | 24.2 % | 25.4 % | 1.382 | n/a | n/a | n/a | yes |
| CNN-RespiratoryNet with augmentation | 42 | csv | yes | 16 | 1 | 25.2 % | 27.0 % | 1.387 | n/a | n/a | n/a | no |
| CNN-RespiratoryNet with augmentation | 123 | csv | yes | 16 | 1 | 30.0 % | 26.4 % | 1.336 | n/a | n/a | n/a | no |
| CNN-RespiratoryNet with augmentation | 7 | csv | yes | 26 | 11 | 24.0 % | 26.4 % | 1.386 | n/a | n/a | n/a | yes |
| CNN-RespiratoryNet with augmentation | 99 | csv | yes | 17 | 2 | 26.3 % | 27.8 % | 1.379 | n/a | n/a | n/a | no |
| CNN-RespiratoryNet with augmentation | 314 | csv | yes | 32 | 17 | 29.3 % | 26.3 % | 1.354 | n/a | n/a | n/a | no |
| MobileNetV3-ImageNet without augmentation | 42 | log | yes | 19 | 4 | 98.4 % | 77.8 % | 0.053 | n/a | n/a | n/a | no |

## Per model

| Model | Verified runs | Epochs | Selected ep. | Train acc | Test AUC | Pearson (5 seeds) | Pearson (without degenerate) | Degenerate |
|---|---|---|---|---|---|---|---|---|
| CNN-RespiratoryNet (FP32) | 5/5 | 17-80 | 2-69 | 23.8-80.2 % | 0.545-0.605 | -0.58 (p 0.31, n=5) | 0.74 (p 0.26, n=4) | 1 |
| CfC-Hybrid | 5/5 | 53-80 | 33-66 | 44.8-60.2 % | 0.478-0.526 | 0.27 (p 0.66, n=5) | 0.27 (p 0.66, n=5) | 0 |
| LSTM-Hybrid | 5/5 | 71-80 | 51-62 | 45.9-52.8 % | n/a-n/a | n/a (p n/a, n=0) | n/a (p n/a, n=0) | 0 |
| GRU-Hybrid | 5/5 | 70-80 | 50-70 | 44.1-57.3 % | n/a-n/a | n/a (p n/a, n=0) | n/a (p n/a, n=0) | 0 |
| CfC without mixed memory | 5/5 | 21-36 | 1-16 | 23.2-26.7 % | n/a-n/a | n/a (p n/a, n=0) | n/a (p n/a, n=0) | 3 |
| CNN-RespiratoryNet with augmentation | 5/5 | 16-32 | 1-17 | 24.0-30.0 % | n/a-n/a | n/a (p n/a, n=0) | n/a (p n/a, n=0) | 1 |
| MobileNetV3-ImageNet without augmentation | 1/1 | 19-19 | 4-4 | 98.4-98.4 % | n/a-n/a | n/a (p n/a, n=0) | n/a (p n/a, n=0) | 0 |

_No source: MNv3_IN/123, MNv3_IN/7, MNv3_IN/99, MNv3_IN/314, MNv3_SC/42, MNv3_SC/123, MNv3_SC/7, MNv3_SC/99, MNv3_SC/314, PANNs/123, PANNs/7, PANNs/99, PANNs/314, PANNs_tune/42, PANNs_tune/123, PANNs_tune/7, PANNs_tune/99, PANNs_tune/314._

_Runs whose source does not reproduce the checkpoint epoch:_

- MNv3_IN/42: no readable source
- PANNs/42: no readable source
- MNv3_noaug/123: no readable source
- MNv3_noaug/7: no readable source
- MNv3_noaug/99: no readable source
- MNv3_noaug/314: no readable source

_No run is excluded from the reported averages; the flag is descriptive._
