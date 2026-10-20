# Third-party notices

This repository includes two files derived from third-party code. They are distributed
under their original licenses, whose full texts are in `LICENSES/`.

## torch_cfc.py (Apache License 2.0)

- Files: `scripts/torch_cfc.py` and `events/torch_cfc.py` (identical copies; each folder
  imports it from its own location).
- Source: <https://github.com/raminmh/CfC>, file `torch_cfc.py`, by Ramin Hasani,
  Mathias Lechner, and co-authors.
- License: Apache License 2.0, `LICENSES/Apache-2.0_raminmh-CfC.txt`.
- Modifications: none. The file is byte-identical to the upstream `torch_cfc.py`
  (checked on 2026-10-20).
- Reference: Hasani R, et al. Closed-form continuous-time neural networks.
  *Nature Machine Intelligence* 4, 992-1003 (2022).

## panns_cnn6.py (MIT License)

- File: `scripts/panns_cnn6.py`.
- Source: the `Cnn6` class and its initializers from
  <https://github.com/qiuqiangkong/audioset_tagging_cnn> (`pytorch/models.py`),
  Copyright (c) 2018-2020 Qiuqiang Kong.
- License: MIT, `LICENSES/MIT_qiuqiangkong-audioset_tagging_cnn.txt`.
- Modifications: the architecture is reproduced so that the AudioSet checkpoint loads
  with matching layer names; comments were added, and a transfer-learning wrapper
  (`Transfer_Cnn6`) with a new classification head was written for this study.
- Pretrained weights (not included): `Cnn6_mAP=0.343.pth`,
  <https://doi.org/10.5281/zenodo.3987831>, CC BY 4.0.
- Reference: Kong Q, et al. PANNs: Large-scale pretrained audio neural networks for audio
  pattern recognition. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*
  28, 2880-2894 (2020).
