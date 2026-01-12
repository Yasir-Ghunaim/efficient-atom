"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
from jmp.lightning import GradientClippingConfig

from ...modules.dataset.concat_dataset import MTDatasetConfig
from ...modules.ema import EMAConfig
from ...tasks.config import AdamWConfig
from ...tasks.pretrain import PretrainConfig
from ...tasks.pretrain.module import LinearWarmupCosineAnnealingSchedulerConfig
from ...models.gemnet.config import BackboneConfig
from ...models.equiformer_v2.config import EquiformerV2Config

from ...tasks.finetune.base import (
    PrimaryMetricConfig,
    CheckpointBestConfig,
)


def jmp_l_pt_config_(config: PretrainConfig, args: argparse.Namespace):
    # Assign seed for config and pytorch lightning
    config.trainer.seed = args.seed
    config.set_seed(args.seed)

    # Set the model trainer settings for maximum performance
    config.trainer.precision = "16-mixed"
    config.trainer.set_float32_matmul_precision = "medium"
    config.trainer.supports_parameter_hooks = False
    config.trainer.supports_skip_batch_exception = False

    # Batch size
    config.batch_size = args.batch_size
    config.eval_batch_size = 20

    # Epochs
    config.trainer.max_epochs = args.epochs

    # Set backbone config
    # config.backbone = BackboneConfig.large() if args.large else BackboneConfig.small()
    if "gemnet" in config.model_name:
        config.backbone = BackboneConfig.large()
    elif "equiformer_v2" in config.model_name:
        config.backbone = EquiformerV2Config.small()

    # Optimizer settings
    config.optimizer = AdamWConfig(
        lr=args.lr,
        amsgrad=False,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    config.trainer.optimizer.log_grad_norm = True
    config.trainer.optimizer.gradient_clipping = GradientClippingConfig(
        value=1.0,
        algorithm="norm",
    )
    # LR Scheduler settings
    config.lr_scheduler = LinearWarmupCosineAnnealingSchedulerConfig(
        warmup_steps=2000,
        warmup_start_lr_factor=0.2,
        min_lr_factor=0.1,
        max_epochs=args.epochs, # Note that we set max_epochs to None in the pretrain.py as we use max_steps instead.
    )
    # Regularization settings
    config.edge_dropout = 0.1
    
    # EMA settings
    if "gemnet" in config.model_name:
        config.ema = EMAConfig(decay=0.99)
    elif "equiformer_v2" in config.model_name:
        config.ema = None  # Disable EMA for EquiformerV2


    # Set data config
    config.num_workers = args.num_workers

    config.primary_metric = PrimaryMetricConfig(name="loss", mode="min")
    config.ckpt_best = CheckpointBestConfig()

    # Set up the JMP MT dataset config and tasks
    config.mt_dataset = MTDatasetConfig(
        sample_type="temperature",
        sample_temperature=1.0,
    )
