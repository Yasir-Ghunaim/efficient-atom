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
from ...tasks.finetune import SPICEConfig
from ...tasks.finetune import dataset_config as DC
from ...tasks.finetune.base import PrimaryMetricConfig

STATS: dict[str, dict[str, NC]] = {
    "dipeptides": {
        "y": NC(mean=-31213.615, std=4636.815),
        "force": NC(mean=3.3810358e-07, std=0.5386545),
    },
    "solvated_amino_acids": {
        "y": NC(mean=-60673.68, std=3310.6692),
        "force": NC(mean=2.7950014e-07, std=0.81945145),
    },
}


def jmp_l_spice_config_(config: SPICEConfig, dataset: DC.SPICEDataset, base_path: Path, args: argparse.Namespace = None):
    # Optimizer settings
    # config.optimizer = AdamWConfig(
    #     lr=5.0e-6,
    #     amsgrad=False,
    #     betas=(0.9, 0.95),
    #     weight_decay=0.1,
    # )

    # Set data config
    if args.batch_size:
        config.batch_size = args.batch_size
    else:
        config.batch_size = 1

    # Set up dataset
    config.train_dataset = DC.spice_config(dataset, base_path, "train", args=args)
    config.val_dataset = DC.spice_config(dataset, base_path, "val", args=args)
    config.test_dataset = DC.spice_config(dataset, base_path, "test", args=args)

    # Spice specific settings
    config.dataset = dataset
    config.primary_metric = PrimaryMetricConfig(name="force_mae", mode="min")

    # Gradient forces
    config.model_type = "forces"
    if config.model_name ==  "gemnet":
        config.gradient_forces = True
        config.trainer.inference_mode = False
    elif config.model_name == "equiformer_v2":
        config.backbone.regress_forces = True
        config.backbone.direct_forces = True

    # Set up normalization
    if (normalization_config := STATS.get(dataset)) is None:
        raise ValueError(f"Normalization for {dataset} not found")
    config.normalization = normalization_config
