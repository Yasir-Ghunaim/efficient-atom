from pathlib import Path
import os
import argparse

from jmp.tasks.pretrain import PretrainConfig, PretrainModel
from jmp.lightning import Runner, Trainer
from jmp.utils.fit_scales import fit_scales_new
from jmp.modules.scaling.util import ensure_fitted

from setup_pretrain import load_global_config, configure_model

# Load global config
global_config = load_global_config()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pre-training script for JMP-L")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers")
    parser.add_argument("--train_samples_limit", type=int, default=1000000, help="Number of training samples to use")
    parser.add_argument("--val_samples_limit", type=int, default=-1, help="Number of validation samples to use")
    parser.add_argument("--large", action="store_true", help="Load the large pre-trained checkpoint")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--root_path", type=str, help="Root path containing datasets and checkpoints")
    parser.add_argument("--model_name", type=str, default="gemnet", help="Model name")
    parser.add_argument("--task", type=str,
                        required=True, help="Name of the pretraining task. Choose from: oc20, oc22, ani1x, transition1x.")
    parser.add_argument("--sampling_strategy", type=str, choices=["random", "low_difficulty", "mid_difficulty", "high_difficulty", "mixed_difficulty"],
                        default="random", help="Sampling strategy to use: 'random', only 'random' is supported for now.")
    parser.add_argument("--temperature_sampling", action="store_true", help="Use temperature sampling equal to 2")
    parser.add_argument("--ani1x_ood", action="store_true", help="Custom sampling for ani1x ood experiment")

    args = parser.parse_args()
    args.root_path = global_config.get("root_path", None)

    return args



def build_job_name(args, config):
    """Construct a job name based on the configuration."""
    job_name = f"PT_{args.task}_lr{args.lr}_train{args.train_samples_limit}_val{args.val_samples_limit}_ep{args.epochs}"
    if args.scratch:
        job_name += f"_scratch"

    # Remove the underscore and capitalize each starting letter if an underscore exists:
    if "_" in args.sampling_strategy:
        formatted_strategy = ''.join(word.capitalize() for word in args.sampling_strategy.split('_'))
    else:
        formatted_strategy = args.sampling_strategy.capitalize()
    job_name += f"_{formatted_strategy}"
    
    if args.temperature_sampling:
        job_name += "_TmpSampling"

    if args.ani1x_ood:
        job_name += "_AniOOD"

    if args.large:
        job_name += "_large"

    if args.model_name == "equiformer_v2":
        job_name += f"_eqv2"

    return job_name

def run_training(
    config: PretrainConfig, 
    model_cls: type[PretrainModel],
    args
    ):
    """Run the training process."""

    print("Creating the Model =================")
    model = model_cls(config)

    if args.model_name == "gemnet":
        print("Fitting GemNet-OC model =================")
        fit_scales_new(
            config=config,
            model=model,
            backbone=lambda m: m.backbone
        )

        ensure_fitted(model)

    trainer = Trainer(config)
    # trainer.validate(model)
    trainer.fit(model)

def main():
    args = parse_args()
    config = configure_model(args)
    config.name = build_job_name(args, config)

    print("The arguments are:", args)
    print(config)

    if "SLURM_JOB_ID" in os.environ:
        os.environ["SUBMITIT_EXECUTOR"] = "slurm"
    runner = Runner(run_training)
    runner([(config, PretrainModel, args)])

if __name__ == "__main__":
    main()
