import torch

from pathlib import Path
import os
import yaml
from jmp.configs.pretrain.jmp_l import jmp_l_pt_config_
from jmp.tasks.pretrain import PretrainConfig
from jmp.tasks.pretrain.module import (
    NormalizationConfig,
    PretrainDatasetConfig,
    TaskConfig,
)

def load_global_config(filename="global_config.yaml"):
    config_path = Path(__file__).resolve().parent.parent / filename
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Configure tasks based on command-line arguments
def configure_tasks(args):
    dataset_path = Path(args.root_path) / "datasets"
    train_samples_limit = args.train_samples_limit
    if args.val_samples_limit > 0:
        val_samples_limit = args.val_samples_limit
    else:
        val_samples_limit = 2500

    dataset_names = args.task.split(",")

    # Here we apply the ratios of temperature = 2 based on the dataset sizes reported in JMP paper
    if hasattr(args, "temperature_sampling") and args.temperature_sampling:
        temperature_limit = {
            "ani1x": int(0.0812536740566486 * train_samples_limit),
            "transition1x": int(0.18168873861227738 * train_samples_limit),
            "oc20": int(0.5745502392177768 * train_samples_limit),
            "oc22": int(0.1625073481132972 * train_samples_limit)
        }
        if sum(temperature_limit.values()) != train_samples_limit:
            temperature_limit["ani1x"] += train_samples_limit - sum(temperature_limit.values())
        
        assert sum(temperature_limit.values()) == train_samples_limit, "Temperature limit does not match train_samples_limit"
    
    elif hasattr(args, "ani1x_ood") and args.ani1x_ood:
        temperature_limit = {
            "ani1x": int(2/3 * train_samples_limit),
            "transition1x": 0,
            "oc20": 0,
            "oc22": int(1/3 * train_samples_limit)
        }
        assert sum(temperature_limit.values()) == train_samples_limit, "Temperature limit does not match train_samples_limit"

    else:
        train_samples_limit = train_samples_limit // len(dataset_names)


    is_custom_ratios = args.temperature_sampling or args.ani1x_ood

    all_tasks = {
        "oc20": TaskConfig(
            name="oc20",
            train_dataset=PretrainDatasetConfig(
                src=dataset_path / "oc20_s2ef/2M/train/",
                metadata_path=dataset_path / "oc20_s2ef/2M/train_metadata.npz",
                lin_ref=dataset_path / "oc20_s2ef/2M/linref.npz",
                max_samples=temperature_limit["oc20"] if is_custom_ratios else train_samples_limit,
                is_train=True,
                args=args
            ),
            val_dataset=PretrainDatasetConfig(
                src=dataset_path / "oc20_s2ef/all/val_id/",
                metadata_path=dataset_path / "oc20_s2ef/all/val_id_metadata.npz",
                lin_ref=dataset_path / "oc20_s2ef/2M/linref.npz",
                max_samples=val_samples_limit,
                is_train=False,
                args=args
            ),
            energy_loss_scale=1.0,
            force_loss_scale=10.0,
            normalization={
                "y": NormalizationConfig(mean=0.0, std=24.901469505465872),
                "force": NormalizationConfig(mean=0.0, std=0.5111534595489502),
            },
        ),
        "oc22": TaskConfig(
            name="oc22",
            train_dataset=PretrainDatasetConfig(
                src=dataset_path / "oc22/s2ef-total/train/",
                metadata_path=dataset_path / "oc22/s2ef-total/train_metadata.npz",
                lin_ref=dataset_path / "oc22/s2ef-total/linref.npz",
                max_samples=temperature_limit["oc22"] if is_custom_ratios else train_samples_limit,
                is_train=True,
                args=args
            ),
            val_dataset=PretrainDatasetConfig(
                src=dataset_path / "oc22/s2ef-total/val_id/",
                metadata_path=dataset_path / "oc22/s2ef-total/val_id_metadata.npz",
                lin_ref=dataset_path / "oc22/s2ef-total/linref.npz",
                max_samples=val_samples_limit,
                is_train=False,
                args=args
            ),
            energy_loss_scale=1.0,
            force_loss_scale=10.0,
            normalization={
                "y": NormalizationConfig(mean=0.0, std=25.229595396538468),
                "force": NormalizationConfig(mean=0.0, std=0.25678861141204834),
            },
        ),
        "ani1x": TaskConfig(
            name="ani1x",
            train_dataset=PretrainDatasetConfig(
                src=dataset_path / "ani1x/lmdb/train/",
                metadata_path=dataset_path / "ani1x/lmdb/train/metadata.npz",
                lin_ref=dataset_path / "ani1x/linref.npz",
                max_samples=temperature_limit["ani1x"] if is_custom_ratios else train_samples_limit,
                is_train=True,
                args=args
            ),
            val_dataset=PretrainDatasetConfig(
                src=dataset_path / "ani1x/lmdb/val/",
                metadata_path=dataset_path / "ani1x/lmdb/val/metadata.npz",
                lin_ref=dataset_path / "ani1x/linref.npz",
                max_samples=val_samples_limit,
                is_train=False,
                args=args
            ),
            energy_loss_scale=1.0,
            force_loss_scale=10.0,
            normalization={
                "y": NormalizationConfig(mean=0.0, std=2.8700712783472118),
                "force": NormalizationConfig(mean=0.0, std=2.131422996520996),
            },
        ),
        "transition1x": TaskConfig(
            name="transition1x",
            train_dataset=PretrainDatasetConfig(
                src=dataset_path / "transition1x/lmdb/train/",
                metadata_path=dataset_path / "transition1x/lmdb/train/metadata.npz",
                lin_ref=dataset_path / "transition1x/linref.npz",
                max_samples=temperature_limit["transition1x"] if is_custom_ratios else train_samples_limit,
                is_train=True,
                args=args
            ),
            val_dataset=PretrainDatasetConfig(
                src=dataset_path / "transition1x/lmdb/val/",
                metadata_path=dataset_path / "transition1x/lmdb/val/metadata.npz",
                lin_ref=dataset_path / "transition1x/linref.npz",
                max_samples=val_samples_limit,
                is_train=False,
                args=args
            ),
            energy_loss_scale=1.0,
            force_loss_scale=10.0,
            normalization={
                "y": NormalizationConfig(mean=0.0, std=1.787466168382901),
                "force": NormalizationConfig(mean=0.0, std=0.3591422140598297),
            },
        ),
        "omat": TaskConfig(
            name="omat",
            train_dataset=PretrainDatasetConfig(
                src=dataset_path / "omat/train/rattled-300-subsampled",  # Folder of .aselmdb files
                metadata_path=dataset_path / "omat/train/rattled-300-subsampled/metadata.npz",
                lin_ref=dataset_path / "omat/train/linref.npz",
                max_samples=train_samples_limit,#int(train_samples_limit*2),
                is_train=True,
                args=args,
            ),
            val_dataset=PretrainDatasetConfig(
                src=dataset_path / "omat/val/rattled-300-subsampled",  # or point to val if you extract it separately
                metadata_path=dataset_path / "omat/val/rattled-300-subsampled/metadata.npz",
                lin_ref=dataset_path / "omat/train/linref.npz",
                max_samples=val_samples_limit,
                is_train=False,
                args=args,
            ),
            energy_loss_scale=20.0,
            force_loss_scale=20.0,
            normalization={
                "y": NormalizationConfig(mean=0.0, std=7.783477783203125),
                "force": NormalizationConfig(mean=0.0, std=4.445453643798828),
            },
        ),

    }

    # Filter tasks based on dataset_names
    tasks = [all_tasks[dataset_name] for dataset_name in dataset_names if dataset_name in all_tasks]
    return tasks



def configure_model(args):
    """Set up the model configuration based on command-line arguments."""
    config = PretrainConfig.draft()
    config.model_name = args.model_name
    jmp_l_pt_config_(config, args)
    config.tasks = configure_tasks(args)
    config.args = args
    config = config.finalize()
    configure_wandb(config, args)
    configure_validation_and_scheduler(config, args)
    
    return config


def configure_wandb(config, args):
    """Configure WandB settings."""
    if args.enable_wandb:
        global_config = load_global_config()
        config.project = global_config.get("wandb").get("pretrain_project")
    else:
        config.trainer.logging.wandb.enabled = False


def configure_validation_and_scheduler(config, args):
    """Set validation and scheduler parameters."""
    # config.trainer.val_check_interval = 0.0002  # Run validation every 0.02% of the epoch (almost every hour)
    # config.lr_scheduler.max_epochs = None
    # config.lr_scheduler.max_steps = 800000

    config.trainer.val_check_interval = 0.05  # Run validation every 5% of the epoch
    config.lr_scheduler.max_epochs = None

    num_gpus = torch.cuda.device_count()
    effective_batch_size = args.batch_size * num_gpus

    max_steps = (args.train_samples_limit // effective_batch_size) * args.epochs
    config.lr_scheduler.max_steps = max_steps
