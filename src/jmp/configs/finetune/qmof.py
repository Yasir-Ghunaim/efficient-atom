"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
from pathlib import Path

from ...modules.transforms.normalize import NormalizationConfig as NC
from ...tasks.config import AdamWConfig
from ...tasks.finetune import QMOFConfig
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import PrimaryMetricConfig

STATS: dict[str, NC] = {
    "y": NC(mean=2.1866251527, std=1.175752521125648),
}


def jmp_l_qmof_config_(config: QMOFConfig, base_path: Path, target: str = "y", args: argparse.Namespace = None):
    # Optimizer settings
    # config.optimizer = AdamWConfig(
    #     lr=5.0e-6,
    #     amsgrad=False,
    #     betas=(0.9, 0.95),
    #     weight_decay=0.1,
    # )

    # Set data config
    config.batch_size = 4

    # Set up dataset
    config.train_dataset = DC.qmof_config(base_path, "train", args=args)
    config.val_dataset = DC.qmof_config(base_path, "val", args=args)
    config.test_dataset = DC.qmof_config(base_path, "test", args=args)

    # Set up normalization
    if (normalization_config := STATS.get(target)) is None:
        raise ValueError(f"Normalization for {target} not found")
    config.normalization = {target: normalization_config}

    # QMOF specific settings
    config.primary_metric = PrimaryMetricConfig(name="y_mae", mode="min")

    # Make sure we only optimize for the single target
    config.graph_scalar_targets = [target]
    config.node_vector_targets = []
    config.graph_classification_targets = []
    # config.graph_scalar_reduction = {target: "sum"}
    config.graph_scalar_reduction_default = "mean"
