# -*- coding: utf-8 -*-
"""
PANNs CNN6 architecture (vendored) and transfer-learning wrapper.

Adapted from PANNs (Kong et al., 2020), https://github.com/qiuqiangkong/audioset_tagging_cnn (MIT license).

The Cnn6 class is copied architecturally verbatim from pytorch/models.py of that
repository so that the state_dict matches the AudioSet checkpoint (Cnn6_mAP=0.343.pth)
exactly by layer name and shape; it is vendored rather than installed to keep full
control during fine-tuning. The model takes raw 32 kHz audio (batch_size, data_length)
and computes its own 64-band log-mel front-end with the pretraining parameters
(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000),
which must not be changed. Requires torchlibrosa.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


# Initializers (identical to the original repository)
def init_layer(layer):
    """Initialize a Linear or Conv layer (Xavier uniform, zero bias)."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a BatchNorm layer (zero bias, unit weight)."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def do_mixup(x, mixup_lambda):
    """
    Tensor-level mixup over consecutive batch pairs (identical to the original
    repository). Not used by default here (mixup is applied to the waveform in
    the training loop); kept for fidelity.
    """
    out = (x[0::2].transpose(0, -1) * mixup_lambda[0::2] +
           x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)
    return out


class ConvBlock5x5(nn.Module):
    """5x5 convolutional block used by CNN6 (one conv per block)."""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")
        return x


class Cnn6(nn.Module):
    """CNN6 from the original PANNs repository."""

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num):
        super(Cnn6, self).__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window=window, center=center, pad_mode=pad_mode, freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin,
            fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)

        # SpecAugment (applied only in training)
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """Input: raw audio (batch_size, data_length)."""
        x = self.spectrogram_extractor(input)   # (B, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)            # (B, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)                # mean over frequency

        (x1, _) = torch.max(x, dim=2)           # max over time
        x2 = torch.mean(x, dim=2)               # mean over time
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class Transfer_Cnn6(nn.Module):
    """
    AudioSet-pretrained Cnn6 (527 classes) with a new 'fc_transfer' head of
    `classes_num` outputs, following the official PANNs finetune_template.py.
    forward returns raw logits (not log_softmax) for nn.CrossEntropyLoss and a
    direct softmax at evaluation. Front-end parameters are fixed to the checkpoint.
    """
    # Constants of the official Cnn6_mAP=0.343.pth checkpoint; do not change
    SAMPLE_RATE = 32000
    WINDOW_SIZE = 1024
    HOP_SIZE = 320
    MEL_BINS = 64
    FMIN = 50
    FMAX = 14000
    AUDIOSET_CLASSES = 527
    EMBEDDING_DIM = 512

    def __init__(self, classes_num=4, freeze_base=False):
        super(Transfer_Cnn6, self).__init__()
        self.base = Cnn6(
            sample_rate=self.SAMPLE_RATE, window_size=self.WINDOW_SIZE,
            hop_size=self.HOP_SIZE, mel_bins=self.MEL_BINS,
            fmin=self.FMIN, fmax=self.FMAX, classes_num=self.AUDIOSET_CLASSES)

        # New head for the N-class task
        self.fc_transfer = nn.Linear(self.EMBEDDING_DIM, classes_num, bias=True)

        self.freeze_base = freeze_base
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        self.model_name = f"PANNs_Cnn6_transfer_{classes_num}cls"
        self._init_transfer()

    def _init_transfer(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path, logger=None):
        """
        Load the AudioSet weights into self.base. The official checkpoint stores
        the state_dict under the key 'model'; a plain state_dict is also accepted.
        """
        ckpt = torch.load(pretrained_checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = self.base.load_state_dict(state, strict=False)
        msg = (f"[PANNs] AudioSet weights loaded into base. "
               f"missing={len(missing)} unexpected={len(unexpected)}")
        if logger is not None:
            logger.info(msg)
            if missing:
                logger.info(f"[PANNs]   missing keys (examples): {missing[:4]}")
            if unexpected:
                logger.info(f"[PANNs]   unexpected keys (examples): {unexpected[:4]}")
        return missing, unexpected

    def set_base_frozen(self, frozen: bool):
        for p in self.base.parameters():
            p.requires_grad = not frozen
        self.freeze_base = frozen

    def base_params(self):
        return self.base.parameters()

    def head_params(self):
        return self.fc_transfer.parameters()

    def forward(self, x, mixup_lambda=None):
        """x: raw 32 kHz audio (batch_size, data_length). Returns logits (batch_size, classes_num)."""
        out = self.base(x, mixup_lambda)
        embedding = out["embedding"]               # (B, 512)
        logits = self.fc_transfer(embedding)       # (B, classes_num), raw logits
        return logits
