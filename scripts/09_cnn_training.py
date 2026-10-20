#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 09: CNN-RespiratoryNet architecture and training.

Defines the compact convolutional baseline (four double-convolution blocks with
16-32-64-128 filters, adaptive average pooling to 4x4, and a two-layer classifier
with dropout 0.5) and trains it from scratch on log-mel spectrograms of ICBHI 2017
respiratory cycles. A small network is used because the inputs are single-channel
spectrograms, the training set is small, and the model must run on low-cost edge
hardware. The official ICBHI score (Se, Sp) is logged every epoch to detect collapse
onto one class; the checkpoint is selected by a configurable validation metric
(macro-F1 by default). Optional tools (torchinfo, TensorBoard, torchview, torchviz,
ONNX) are used when installed.

Inputs:  scripts/08_dataset_dataloader.py (data splits), scripts/focal_loss.py
Outputs: models/checkpoints/{run_tag}.pth, {run_tag}_final.pth, training_config.json,
         model_architecture.onnx, best_model.onnx
         results/metrics/{run_tag}_history.csv, model_summary_torchinfo.txt
         results/figures/training_curves.png, overfitting_gap.png, architecture graphs
         results/tensorboard/run_<timestamp>/
         {run_tag}_log.txt (per-epoch log, parsed by script 38)
Usage:
    python scripts/09_cnn_training.py [--seed 42] [--loss focal] [--focal_gamma 2.0]
        [--focal_alpha balanced] [--sampler natural] [--select_metric macro_f1]
        [--epochs 80] [--lr 3e-4] [--tag TAG]
"""

import os
import sys
import json
import time
import copy
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# Silence ONNX export warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.onnx")
# dynamic_axes vs dynamo export warning
warnings.filterwarnings("ignore", message=".*dynamic_axes.*")


# Optional visualization tools: the script runs with whichever are installed.
try:
    from torchinfo import summary as torchinfo_summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    from torchview import draw_graph
    HAS_TORCHVIEW = True
except ImportError:
    HAS_TORCHVIEW = False

try:
    from torchviz import make_dot
    HAS_TORCHVIZ = True
except ImportError:
    HAS_TORCHVIZ = False

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_log.txt', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
CHECKPOINTS_DIR = BASE_DIR / "models" / "checkpoints"
FIGURES_DIR = BASE_DIR / "results" / "figures"
METRICS_DIR = BASE_DIR / "results" / "metrics"
TENSORBOARD_DIR = BASE_DIR / "results" / "tensorboard"


def _import_focal_loss():
    """Import the shared focal_loss.py module (loss builder, official ICBHI score, selection metrics)."""
    from importlib.util import spec_from_file_location, module_from_spec
    here = Path(__file__).resolve().parent
    candidates = [here / "focal_loss.py", SCRIPTS_DIR / "focal_loss.py",
                  BASE_DIR / "focal_loss.py", BASE_DIR / "scripts" / "focal_loss.py"]
    for path in candidates:
        if path.exists():
            spec = spec_from_file_location("focal_loss", str(path))
            m = module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError("focal_loss.py not found. Copy it into the scripts/ folder.")



TRAINING_CONFIG = {
    # Architecture
    "model_name": "RespiratoryNet",
    "n_classes": 4,                # 4class: Normal/Crackle/Wheeze/Both
    "input_channels": 1,           # Mel-spectrogram = 1 channel
    "base_filters": 16,            # Filters in the first conv layer
    "n_conv_blocks": 4,            # Convolutional blocks
    "fc_hidden_dim": 128,          # Units in the fully connected layer
    "dropout_rate": 0.5,           # Classifier dropout
    
    # Optimization
    "learning_rate": 3e-4,         # Initial LR
    "weight_decay": 1e-4,          # L2 regularization
    "optimizer": "Adam",           # Adam is robust on small datasets
    
    # Training
    "n_epochs": 80,                # Maximum epochs
    "batch_size": 32,              # Batch size
    
    # Early stopping: stop if val_loss does not improve for 'patience' consecutive epochs
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 1e-4,  # Minimum improvement that counts
    
    # LR scheduler: halve the LR after 7 epochs without val_loss improvement
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 7,
    "scheduler_min_lr": 1e-6,
    
    # Reproducibility
    "random_seed": 42,
    
    # Data
    "split_strategy": "strategy2_custom",  # Zero leakage
    "class_mode": "4class",
    "augment_preset": "none",              # augmentation is evaluated in a separate ablation
    "use_weighted_sampling": False,        # no sampler: focal loss handles class imbalance
    "use_weighted_loss": False,

    # Loss and checkpoint-selection criterion. Defaults: focal loss without sampler
    # (Pham et al. 2022 style). Previous baseline: --loss ce --sampler weighted --select_metric val_loss
    "loss": "focal",             # 'ce' (previous baseline) | 'focal' (recommended)
    "focal_gamma": 2.0,          # screened over {1, 2, 3}
    "focal_alpha": "balanced",   # 'none' | 'balanced' (inverse-freq) | 'effective'
    "select_metric": "macro_f1", # robust to collapse. Alt: 'val_loss','val_official','val_macro'
    
    # Visualization
    "use_torchinfo": True,
    "use_tensorboard": True,
    "use_torchview": True,
    "use_torchviz": True,
    "export_onnx": True,                   # for Netron
    "log_histograms_every_n_epochs": 10,   # weight histograms in TensorBoard
}


class ConvBlock(nn.Module):
    """
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> MaxPool 2x2.

    Two stacked 3x3 convolutions before pooling give a 5x5 receptive field with fewer
    parameters and preserve time resolution for brief events such as crackles, which
    early pooling would blur. The convolution bias is omitted because BatchNorm has one.
    """
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False  # BatchNorm provides the bias
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class RespiratoryNet(nn.Module):
    """
    CNN-RespiratoryNet for respiratory cycle classification.

    (B, 1, 128, 251) -> 4 ConvBlocks (16, 32, 64, 128 filters) -> (B, 128, 8, 15)
    -> AdaptiveAvgPool (4, 4) -> Flatten (2048) -> FC 128 -> ReLU -> Dropout -> FC n_classes.
    Adaptive pooling fixes the classifier input size regardless of spectrogram length and
    keeps the first FC layer small (2048 instead of 15360 inputs). Returns logits.
    """
    
    def __init__(
        self,
        n_classes=4,
        input_channels=1,
        base_filters=16,
        n_conv_blocks=4,
        fc_hidden_dim=128,
        dropout_rate=0.5,
    ):
        super(RespiratoryNet, self).__init__()
        
        self.n_classes = n_classes
        self.model_name = "RespiratoryNet"
        
        # Convolutional blocks: the number of filters doubles at each block
        conv_blocks = []
        in_ch = input_channels
        
        for i in range(n_conv_blocks):
            out_ch = base_filters * (2 ** i)  # 16, 32, 64, 128
            conv_blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        
        self.features = nn.Sequential(*conv_blocks)
        
        # Fixed-size (4, 4) pooling, independent of the input length
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier; in_ch is the last block's filter count (128 by default)
        fc_input_dim = in_ch * 4 * 4  # 128 * 4 * 4 = 2048
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_hidden_dim, n_classes),
            # No Softmax: the loss operates on logits
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization (He et al., 2015), suited to ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Map (B, 1, n_mels, time) spectrograms to (B, n_classes) logits."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def visualize_model_architecture(model, input_shape, device, config):
    """
    Run the available architecture tools once before training: torchinfo summary,
    torchview diagram, torchviz gradient graph, and ONNX export for Netron.
    """
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    if HAS_TORCHINFO and config.get("use_torchinfo", True):
        logger.info("\n" + "═" * 70)
        logger.info("TORCHINFO - Architecture summary")
        logger.info("(equivalent to MATLAB analyzeNetwork())")
        logger.info("═" * 70)
        
        try:
            model_summary = torchinfo_summary(
                model,
                input_size=(1, *input_shape),  # (batch, C, H, W)
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                    "mult_adds",
                ],
                col_width=18,
                row_settings=["var_names"],
                verbose=2,
            )
            
            summary_path = METRICS_DIR / "model_summary_torchinfo.txt"
            METRICS_DIR.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(str(model_summary))
            logger.info(f"\n  Summary saved: {summary_path}")
            
        except Exception as e:
            logger.warning(f"  torchinfo error: {e}")
    else:
        if not HAS_TORCHINFO:
            logger.info("\n  torchinfo not installed -> pip install torchinfo")
    
    if HAS_TORCHVIEW and config.get("use_torchview", True):
        logger.info("\n" + "═" * 70)
        logger.info("TORCHVIEW - Architecture diagram")
        logger.info("(equivalent to MATLAB Deep Network Designer)")
        logger.info("═" * 70)
        
        try:
            model_graph = draw_graph(
                model,
                input_size=(1, *input_shape),
                device=device,
                expand_nested=True,
                graph_name="RespiratoryNet",
                save_graph=True,
                filename="model_architecture_torchview",
                directory=str(FIGURES_DIR),
            )
            logger.info(f"  Diagram saved in: {FIGURES_DIR}")
            
        except Exception as e:
            logger.warning(f"  torchview error: {e}")
            logger.warning("  Possible cause: Graphviz not installed")
            logger.warning("  Install: https://graphviz.org/download/")
    else:
        if not HAS_TORCHVIEW:
            logger.info("\n  torchview not installed -> pip install torchview")
    
    if HAS_TORCHVIZ and config.get("use_torchviz", True):
        logger.info("\n" + "═" * 70)
        logger.info("TORCHVIZ - Computational graph with gradients")
        logger.info("(useful to check that gradients flow)")
        logger.info("═" * 70)
        
        try:
            model.eval()
            output = model(dummy_input)
            
            dot = make_dot(
                output,
                params=dict(model.named_parameters()),
                show_attrs=False,
                show_saved=False,
            )
            
            graph_path = FIGURES_DIR / "model_gradient_graph"
            dot.render(str(graph_path), format='png', cleanup=True)
            logger.info(f"  Graph saved: {graph_path}.png")
            
            model.train()
            
        except Exception as e:
            logger.warning(f"  torchviz error: {e}")
            logger.warning("  Possible cause: Graphviz not installed")
    else:
        if not HAS_TORCHVIZ:
            logger.info("\n  torchviz not installed -> pip install torchviz")
    
    if config.get("export_onnx", True):
        logger.info("\n" + "═" * 70)
        logger.info("ONNX - Export for Netron")
        logger.info("(open with Netron for an interactive view)")
        logger.info("═" * 70)
        
        try:
            onnx_path = CHECKPOINTS_DIR / "model_architecture.onnx"
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            
            model.eval()
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['mel_spectrogram'],
                output_names=['class_logits'],
                dynamic_axes={
                    'mel_spectrogram': {0: 'batch_size'},
                    'class_logits': {0: 'batch_size'},
                },
            )
            model.train()
            
            # Validate the exported graph if onnx is installed
            if HAS_ONNX:
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                logger.info(f"  Valid ONNX model: {onnx_path}")
            else:
                logger.info(f"  Model exported: {onnx_path}")
            
            logger.info(f"  Interactive view:")
            logger.info(f"    pip install netron")
            logger.info(f"    netron \"{onnx_path}\"")
            logger.info(f"  Or open it in the Netron app: https://netron.app")
            
        except Exception as e:
            logger.warning(f"  ONNX export error: {e}")
    
    logger.info("\n" + "─" * 70)
    logger.info("VISUALIZATION TOOLS SUMMARY")
    logger.info("─" * 70)
    tools = [
        ("torchinfo", HAS_TORCHINFO, "Tabular summary"),
        ("TensorBoard", HAS_TENSORBOARD, "Real-time monitoring"),
        ("torchview", HAS_TORCHVIEW, "Architecture diagram"),
        ("torchviz", HAS_TORCHVIZ, "Computational graph"),
        ("ONNX/Netron", HAS_ONNX, "Interactive view"),
    ]
    for name, available, desc in tools:
        status = "Available" if available else "Not installed"
        logger.info(f"  {name:15s}: {status:20s} - {desc}")
    logger.info("─" * 70)


class EarlyStopping:
    """
    Stop training when val_loss has not improved by more than min_delta for
    'patience' consecutive epochs (checkpoint selection is handled separately).
    """
    
    def __init__(self, patience=15, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"  EarlyStopping: {self.counter}/{self.patience} "
                    f"(best val_loss: {self.best_loss:.6f} at epoch {self.best_epoch+1})"
                )
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch. Returns (mean loss, accuracy %, predictions, labels)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"  Train Epoch {epoch+1}", leave=False, ncols=100)
    
    for batch_idx, (specs, labels) in enumerate(pbar):
        specs = specs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping against exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        running_loss += loss.item() * specs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def validate(model, loader, criterion, device, epoch, split_name="Val"):
    """
    Evaluate without gradients.

    Returns (mean loss, accuracy %, predictions, labels, softmax probabilities).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"  {split_name} Epoch {epoch+1}", leave=False, ncols=100)
        
        for specs, labels in pbar:
            specs = specs.to(device)
            labels = labels.to(device)
            
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * specs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
    
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_model(model, train_loader, val_loader, config, device, class_weights=None):
    """
    Train with focal or cross-entropy loss, Adam with weight decay, ReduceLROnPlateau
    on val_loss, and early stopping. The checkpoint is chosen by config["select_metric"]
    and saved as {run_tag}.pth; the per-epoch history goes to {run_tag}_history.csv.

    Returns a dict with model (best weights), history (DataFrame), best_epoch,
    best_val_loss, best_val_acc and total_time (seconds).
    """
    
    n_epochs = config["n_epochs"]
    lr = config["learning_rate"]
    seed = config["random_seed"]
    
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Loss: cross-entropy or focal loss via focal_loss.build_criterion
    fl_mod = _import_focal_loss()
    # Per-class training counts (for the focal alpha weights)
    try:
        train_labels = np.array(train_loader.dataset.labels)
        class_counts = [int(np.sum(train_labels == c)) for c in range(config.get("n_classes", 4))]
    except AttributeError:
        class_counts = None
    logger.info(f"  Class counts (train): {class_counts}")

    if config.get("loss", "ce") == "focal":
        if config.get("use_weighted_sampling", False) and \
                str(config.get("focal_alpha", "none")).lower() in ("balanced", "inverse", "effective"):
            logger.warning("  [WARNING] focal_alpha active WITH weighted sampler: possible double "
                           "compensation. Consider the natural sampler (use_weighted_sampling=False).")
        criterion = fl_mod.build_criterion(config, class_counts=class_counts,
                                           device=device, logger=logger)
    elif config.get("use_weighted_loss", True) and class_weights is not None:
        # Previous baseline: class-weighted CrossEntropyLoss
        weight_tensor = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info(f"  Loss: CrossEntropyLoss with weights: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info(f"  Loss: CrossEntropyLoss without weights")

    select_metric = config.get("select_metric", "val_loss")
    logger.info(f"  Selection criterion: {select_metric}")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=config["weight_decay"],
    )
    logger.info(f"  Optimizer: Adam (lr={lr}, weight_decay={config['weight_decay']})")
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        min_lr=config["scheduler_min_lr"],
    )
    
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
    )
    
    writer = None
    if HAS_TENSORBOARD and config.get("use_tensorboard", True):
        run_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        tb_dir = TENSORBOARD_DIR / run_name
        writer = SummaryWriter(str(tb_dir))
        logger.info(f"  TensorBoard: {tb_dir}")
        logger.info(f"  To monitor: tensorboard --logdir \"{TENSORBOARD_DIR}\"")
        
        try:
            dummy_input = torch.randn(1, *config["input_shape"]).to(device)
            writer.add_graph(model, dummy_input)
            logger.info(f"  Model graph logged to TensorBoard")
        except Exception as e:
            logger.warning(f"  Could not log the model graph to TensorBoard: {e}")
    
    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": [],
    }
    
    best_model_state = None
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_select_score = float('-inf')   # configurable selection score (higher is better)
    best_epoch = 0
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING START")
    logger.info("=" * 70)
    
    total_start = time.time()
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss, train_acc, train_preds, train_labels = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device, epoch
        )

        # Official ICBHI validation metrics every epoch, needed to detect collapse:
        # an all-Normal model gives Se=0, Sp=1 and official=0.5 (high but useless).
        val_off, val_off_se, val_off_sp = fl_mod.official_icbhi(val_labels, val_preds)
        val_mf1, _ = fl_mod.macro_f1(val_labels, val_preds)
        
        scheduler.step(val_loss)
        
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"\n  Epoch {epoch+1:3d}/{n_epochs} ({epoch_time:.1f}s) | "
            f"LR: {current_lr:.2e} | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.1f}% | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.1f}% | "
            f"OFFICIAL={val_off:.4f} (Se={val_off_se:.3f} Sp={val_off_sp:.3f}) | "
            f"macroF1={val_mf1:.4f}"
        )
        # Collapse warning (either direction)
        if val_off_se < 0.02 and val_off_sp > 0.95:
            logger.warning("    [COLLAPSE->Normal] Se~0 with Sp~1: predicts almost everything as Normal. "
                           "The official 0.5 is misleading (no abnormal cycles detected).")
        elif val_off_sp < 0.05:
            logger.warning("    [COLLAPSE->Abnormal] Sp~0: predicts almost everything as abnormal "
                           "(no Normal cycles identified). Configuration too aggressive.")
        
        # TensorBoard logging
        if writer:
            writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss,
            }, epoch)
            
            writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc,
            }, epoch)
            
            writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # Weight and gradient histograms every N epochs
            if (epoch + 1) % config.get("log_histograms_every_n_epochs", 10) == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f'weights/{name}', param.data, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'gradients/{name}', param.grad, epoch)
        
        # Keep the best model according to select_metric
        cur_score, score_name = fl_mod.select_score(
            select_metric, val_loss, y_true=val_labels, y_pred=val_preds
        )
        if cur_score > best_select_score:
            best_select_score = cur_score
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": best_model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "select_metric": select_metric,
                "select_score": cur_score,
                "config": config,
            }
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, CHECKPOINTS_DIR / f"{config.get('run_tag', 'best_model')}.pth")
            logger.info(f"  New best model saved "
                        f"({score_name}={cur_score:.4f}, val_loss={best_val_loss:.6f})")
        
        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            logger.info(f"\n  EARLY STOPPING at epoch {epoch+1}")
            logger.info(f"    Best model: epoch {best_epoch+1} "
                       f"(val_loss: {best_val_loss:.6f}, val_acc: {best_val_acc:.1f}%)")
            break
    
    total_time = time.time() - total_start
    
    model.load_state_dict(best_model_state)
    
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": best_model_state,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "config": config,
    }, CHECKPOINTS_DIR / f"{config.get('run_tag', 'model')}_final.pth")
    
    if writer:
        writer.close()
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(METRICS_DIR / f"{config.get('run_tag', 'training')}_history.csv", index=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info(f"  Total time: {total_time/60:.1f} minutes")
    logger.info(f"  Epochs run: {epoch+1}/{n_epochs}")
    logger.info(f"  Best epoch: {best_epoch+1}")
    logger.info(f"  Best val_loss: {best_val_loss:.6f}")
    logger.info(f"  Best val_acc: {best_val_acc:.1f}%")
    logger.info(f"  Model saved: {CHECKPOINTS_DIR / 'final_model.pth'}")
    logger.info("=" * 70)
    
    return {
        "model": model,
        "history": history_df,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "total_time": total_time,
    }


def plot_training_curves(history_df, config, save_dir=None):
    """Save static training curves (loss, accuracy, LR) and the train-val accuracy gap figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    save_dir = save_dir or FIGURES_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    epochs = history_df["epoch"].values
    
    # Figure 1: loss, accuracy and learning rate
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Training curves - {config.get('model_name', 'RespiratoryNet')}\n"
        f"Augmentation: {config.get('augment_preset', 'none')} | "
        f"Split: {config.get('split_strategy', 'strategy2')}",
        fontsize=13, fontweight='bold'
    )
    
    # Panel 1: Loss
    ax1 = axes[0]
    ax1.plot(epochs, history_df["train_loss"], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history_df["val_loss"], 'r-', label='Val Loss', linewidth=2)
    
    best_epoch = history_df.loc[history_df["val_loss"].idxmin(), "epoch"]
    best_loss = history_df["val_loss"].min()
    ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
    ax1.scatter([best_epoch], [best_loss], color='red', s=100, zorder=5,
                label=f'Best (epoch {best_epoch})')
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss (CrossEntropy)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Accuracy
    ax2 = axes[1]
    ax2.plot(epochs, history_df["train_acc"], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history_df["val_acc"], 'r-', label='Val Acc', linewidth=2)
    ax2.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
    
    best_acc = history_df.loc[history_df["val_loss"].idxmin(), "val_acc"]
    ax2.scatter([best_epoch], [best_acc], color='red', s=100, zorder=5,
                label=f'Best: {best_acc:.1f}%')
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Learning Rate
    ax3 = axes[2]
    ax3.plot(epochs, history_df["lr"], 'g-', linewidth=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = Path(save_dir) / "training_curves.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Training curves: {fig_path}")
    
    # Figure 2: overfitting gap
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    gap = np.array(history_df["train_acc"]) - np.array(history_df["val_acc"])
    ax.fill_between(epochs, 0, gap, alpha=0.3, color='orange', label='Overfitting Gap')
    ax.plot(epochs, gap, 'orange', linewidth=2)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7,
               label=f'Best epoch ({best_epoch})')
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Acc - Val Acc (%)")
    ax.set_title(
        "Overfitting gap (Train Acc - Val Acc)\n"
        "A growing gap indicates memorization rather than generalization"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig2_path = Path(save_dir) / "overfitting_gap.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Overfitting gap: {fig2_path}")
    
    return fig_path, fig2_path


def export_trained_model_onnx(model, input_shape, device, save_path=None):
    """Export the trained model (with learned weights) to ONNX for inspection in Netron."""
    save_path = save_path or (CHECKPOINTS_DIR / "best_model.onnx")
    
    model.eval()
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['mel_spectrogram'],
            output_names=['class_logits'],
            dynamic_axes={
                'mel_spectrogram': {0: 'batch_size'},
                'class_logits': {0: 'batch_size'},
            },
        )
        
        if HAS_ONNX:
            onnx_model = onnx.load(str(save_path))
            onnx.checker.check_model(onnx_model)
        
        logger.info(f"  Trained model exported to ONNX: {save_path}")
        logger.info(f"  Open with: netron \"{save_path}\"")
        
    except Exception as e:
        logger.warning(f"  ONNX export error: {e}")


def parse_args():
    """Command-line options; a None default keeps the TRAINING_CONFIG value."""
    p = argparse.ArgumentParser(description="Baseline CNN training with focal loss.")
    p.add_argument("--seed", type=int, default=None, help="Random seed (random_seed).")
    p.add_argument("--loss", type=str, default=None, choices=["ce", "focal"])
    p.add_argument("--focal_gamma", type=float, default=None)
    p.add_argument("--focal_alpha", type=str, default=None,
                   help="'none' | 'balanced' | 'effective'")
    p.add_argument("--sampler", type=str, default=None, choices=["weighted", "natural"],
                   help="'weighted' (previous baseline) or 'natural' (no sampler). With focal_alpha "
                        "balanced, use 'natural' to avoid double compensation.")
    p.add_argument("--select_metric", type=str, default=None,
                   choices=["val_loss", "macro_f1", "val_official", "val_macro"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tag", type=str, default=None,
                   help="Tag for the output files. If omitted, it is derived from the configuration.")
    return p.parse_args()


def build_run_tag(config, explicit_tag=None):
    """Unique name for checkpoint/log/CSV files, so that runs do not overwrite each other."""
    if explicit_tag:
        return explicit_tag
    seed = config.get("random_seed", 42)
    if config.get("loss", "ce") == "focal":
        g = config.get("focal_gamma", 2.0)
        a = str(config.get("focal_alpha", "none"))
        sm = config.get("select_metric", "val_loss")
        samp = "samp" if config.get("use_weighted_sampling", False) else "nosamp"
        return f"cnn_focal_g{g:g}_{a}_{samp}_{sm}_seed{seed}"
    return f"cnn_ce_seed{seed}"


def reconfigure_logging(run_tag):
    """Redirect the file log to {run_tag}_log.txt instead of overwriting training_log.txt."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h); h.close()
    fh = logging.FileHandler(f"{run_tag}_log.txt", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)


def main():
    """Full pipeline: build the model, visualize it, train, and export."""
    args = parse_args()

    # Base config plus CLI overrides (None keeps the TRAINING_CONFIG default)
    config = TRAINING_CONFIG.copy()
    if args.seed is not None:          config["random_seed"] = args.seed
    if args.loss is not None:          config["loss"] = args.loss
    if args.focal_gamma is not None:   config["focal_gamma"] = args.focal_gamma
    if args.focal_alpha is not None:   config["focal_alpha"] = args.focal_alpha
    if args.sampler is not None:       config["use_weighted_sampling"] = (args.sampler == "weighted")
    if args.select_metric is not None: config["select_metric"] = args.select_metric
    if args.epochs is not None:        config["n_epochs"] = args.epochs
    if args.lr is not None:            config["learning_rate"] = args.lr

    run_tag = build_run_tag(config, args.tag)
    config["run_tag"] = run_tag
    reconfigure_logging(run_tag)

    logger.info("=" * 70)
    logger.info("BASELINE CNN TRAINING")
    logger.info("=" * 70)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Run tag: {run_tag}")
    logger.info(f"Config:  loss={config['loss']} gamma={config.get('focal_gamma')} "
                f"alpha={config.get('focal_alpha')} sampler={'weighted' if config['use_weighted_sampling'] else 'natural'} "
                f"select={config.get('select_metric')} seed={config['random_seed']}")
    
    for d in [CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, TENSORBOARD_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Import the DataLoader factory from script 08
    sys.path.insert(0, str(SCRIPTS_DIR))
    
    try:
        from importlib.util import spec_from_file_location, module_from_spec
        
        script_08_path = SCRIPTS_DIR / "08_dataset_dataloader.py"
        
        if not script_08_path.exists():
            logger.error(f"Script 08 not found at: {script_08_path}")
            logger.error("Place 08_dataset_dataloader.py in scripts/")
            sys.exit(1)
        
        spec = spec_from_file_location("dataset_module", str(script_08_path))
        dataset_module = module_from_spec(spec)
        spec.loader.exec_module(dataset_module)
        
        create_dataloaders = dataset_module.create_dataloaders
        logger.info(f"Script 08 imported from: {script_08_path}")
        
    except Exception as e:
        logger.error(f"Error importing script 08: {e}")
        logger.error("Make sure 08_dataset_dataloader.py is in scripts/")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("\n" + "═" * 70)
    logger.info("STEP 1: PREPARE DATA")
    logger.info("═" * 70)
    
    data = create_dataloaders(
        strategy=config["split_strategy"],
        class_mode=config["class_mode"],
        augment_preset=config["augment_preset"],
        batch_size=config["batch_size"],
        use_weighted_sampling=config["use_weighted_sampling"],
    )
    
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    class_weights = data["class_weights"]
    config["input_shape"] = data["input_shape"]
    config["class_names"] = data["class_names"]
    
    logger.info(f"  Input shape: {config['input_shape']}")
    logger.info(f"  Classes: {config['class_names']}")
    
    logger.info("\n" + "═" * 70)
    logger.info("STEP 2: BUILD MODEL")
    logger.info("═" * 70)
    
    model = RespiratoryNet(
        n_classes=config["n_classes"],
        input_channels=config["input_channels"],
        base_filters=config["base_filters"],
        n_conv_blocks=config["n_conv_blocks"],
        fc_hidden_dim=config["fc_hidden_dim"],
        dropout_rate=config["dropout_rate"],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model: {model.model_name}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    logger.info("\n" + "═" * 70)
    logger.info("STEP 3: VISUALIZE ARCHITECTURE")
    logger.info("═" * 70)
    
    visualize_model_architecture(model, config["input_shape"], device, config)
    
    logger.info("\n" + "═" * 70)
    logger.info("STEP 4: TRAINING")
    logger.info("═" * 70)
    
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        class_weights=class_weights,
    )
    
    logger.info("\n" + "═" * 70)
    logger.info("STEP 5: POST-TRAINING VISUALIZATION")
    logger.info("═" * 70)
    
    plot_training_curves(result["history"], config)
    
    export_trained_model_onnx(result["model"], config["input_shape"], device)
    
    # Save the full configuration and results
    config_record = {
        "script": "09_cnn_training.py",
        "timestamp": datetime.now().isoformat(),
        "training_config": {
            k: v for k, v in config.items()
            if not isinstance(v, (torch.Tensor,))
        },
        "results": {
            "best_epoch": result["best_epoch"] + 1,
            "best_val_loss": result["best_val_loss"],
            "best_val_acc": result["best_val_acc"],
            "total_time_minutes": result["total_time"] / 60,
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "tools_available": {
            "torchinfo": HAS_TORCHINFO,
            "tensorboard": HAS_TENSORBOARD,
            "torchview": HAS_TORCHVIEW,
            "torchviz": HAS_TORCHVIZ,
            "onnx": HAS_ONNX,
        },
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        },
    }
    
    config_path = CHECKPOINTS_DIR / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_record, f, indent=2, default=str)
    logger.info(f"  Configuration saved: {config_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Model:            {config['model_name']}")
    logger.info(f"  Parameters:       {total_params:,}")
    logger.info(f"  Best epoch:       {result['best_epoch']+1}")
    logger.info(f"  Best val_loss:    {result['best_val_loss']:.6f}")
    logger.info(f"  Best val_acc:     {result['best_val_acc']:.1f}%")
    logger.info(f"  Total time:       {result['total_time']/60:.1f} minutes")
    logger.info(f"")
    logger.info(f"  OUTPUT FILES:")
    logger.info(f"  ├── {CHECKPOINTS_DIR / 'best_model.pth'}")
    logger.info(f"  ├── {CHECKPOINTS_DIR / 'best_model.onnx'}")
    logger.info(f"  ├── {CHECKPOINTS_DIR / 'training_config.json'}")
    logger.info(f"  ├── {METRICS_DIR / 'training_log.csv'}")
    logger.info(f"  ├── {FIGURES_DIR / 'training_curves.png'}")
    logger.info(f"  ├── {FIGURES_DIR / 'overfitting_gap.png'}")
    if HAS_TENSORBOARD:
        logger.info(f"  └── {TENSORBOARD_DIR / '...'}")
    logger.info(f"")
    logger.info(f"  NEXT STEPS:")
    logger.info(f"  1. Inspect the curves in TensorBoard:")
    logger.info(f"     tensorboard --logdir \"{TENSORBOARD_DIR}\"")
    logger.info(f"  2. Explore the model in Netron:")
    logger.info(f"     netron \"{CHECKPOINTS_DIR / 'best_model.onnx'}\"")
    logger.info(f"  3. Run script 10 for the full test-set evaluation")
    logger.info("=" * 70)
    
    return result


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Script 09: CNN training and visualization")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Visualization tool availability
    print("Visualization tools:")
    print(f"  torchinfo:   {'✅' if HAS_TORCHINFO else '❌ pip install torchinfo'}")
    print(f"  TensorBoard: {'✅' if HAS_TENSORBOARD else '❌ pip install tensorboard'}")
    print(f"  torchview:   {'✅' if HAS_TORCHVIEW else '❌ pip install torchview'}")
    print(f"  torchviz:    {'✅' if HAS_TORCHVIZ else '❌ pip install torchviz'}")
    print(f"  ONNX/Netron: {'✅' if HAS_ONNX else '❌ pip install onnx'}")
    print("=" * 70)
    
    script_08 = SCRIPTS_DIR / "08_dataset_dataloader.py"
    if not script_08.exists():
        print(f"\nIMPORTANT: place 08_dataset_dataloader.py in:")
        print(f"   {SCRIPTS_DIR}")
        print(f"   (it is imported automatically)")
    
    print()
    print("CONFIGURATION:")
    print(f"  Max epochs:         {TRAINING_CONFIG['n_epochs']}")
    print(f"  Batch size:         {TRAINING_CONFIG['batch_size']}")
    print(f"  Learning rate:      {TRAINING_CONFIG['learning_rate']}")
    print(f"  Early stopping:     patience={TRAINING_CONFIG['early_stopping_patience']}")
    print(f"  Split:              {TRAINING_CONFIG['split_strategy']}")
    print(f"  Augmentation:       {TRAINING_CONFIG['augment_preset']}")
    print(f"  Weighted loss:      {TRAINING_CONFIG['use_weighted_loss']}")
    print(f"  Weighted sampling:  {TRAINING_CONFIG['use_weighted_sampling']}")
    print("=" * 70 + "\n")
    
    result = main()
