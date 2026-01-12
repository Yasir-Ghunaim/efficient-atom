import os
import argparse

from jmp.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
from jmp.lightning import Runner, Trainer
from jmp.modules.ema import EMA
from jmp.modules.scaling.util import ensure_fitted

from setup_finetune import load_global_config, get_configs, load_checkpoint, configure_wandb

# Load global config
global_config = load_global_config()

# Set up argument parser
parser = argparse.ArgumentParser(description="Fine-tuning script for JMP-L")
parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb logging")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
parser.add_argument("--target", type=str, required=True, help="Target molecule for the dataset")
parser.add_argument("--fold", type=int, default=0, help="Fold for Matbench dataset")
parser.add_argument("--lr", type=float, default=8.0e-5, help="Learning rate for the optimizer")
parser.add_argument("--scratch", action="store_true", help="Train from scratch")
parser.add_argument("--checkpoint_path", type=str, help="Path of finetune checkpoint to load")
parser.add_argument("--medium", action="store_true", help="Load the medium pre-trained checkpoint")
parser.add_argument("--large", action="store_true", help="Load the large pre-trained checkpoint")
parser.add_argument("--root_path", type=str, help="Root path containing datasets and checkpoints")
parser.add_argument("--model_name", type=str, default="gemnet", help="Model name")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--batch_size", type=int, help="Training batch size")

args = parser.parse_args()
args.root_path = global_config.get("root_path", None)
print("The arguments are:", args)

def build_job_name(args, config):
    """Construct a job name based on the configuration."""
    # Construct the job name
    job_name = f"{args.dataset_name}_{args.model_name}_{args.target}_ep{args.epochs}_seed{args.seed}"

    if args.checkpoint_path:
        job_name += f"_{args.checkpoint_path}" 

    if args.dataset_name == "matbench":
        job_name += f"_fold{args.fold}" 

    if hasattr(config, 'gradient_forces') and not config.gradient_forces:
        job_name += "_direct"

    job_name += f"_LR_{args.lr}"
    if args.scratch:
        job_name += f"_scratch"

    if args.batch_size:
        job_name += f"_bs{args.batch_size}"

    return job_name


def run(config: FinetuneConfigBase, model_cls: type[FinetuneModelBase]) -> None:
    model = model_cls(config)

    # Load the checkpoint
    if not args.scratch:
        load_checkpoint(model, config, args)

    callbacks = []
    if (ema := config.meta.get("ema")) is not None:
        ema = EMA(decay=ema)
        callbacks.append(ema)

    ensure_fitted(model)
    trainer = Trainer(config, callbacks=callbacks)

    # trainer.validate(model)
    trainer.fit(model)
    trainer.test(model)


config, init_model = get_configs(args.dataset_name, args.target, args=args)
config.name = build_job_name(args, config)


if "SLURM_JOB_ID" in os.environ:
    os.environ["SUBMITIT_EXECUTOR"] = "slurm"
configs: list[tuple[FinetuneConfigBase, type[FinetuneModelBase]]] = []
configs.append((config, init_model))

# for config, _ in configs:
#     assert config.backbone.scale_file, f"Scale file not set for {config.name}"

runner = Runner(run)
runner(configs)