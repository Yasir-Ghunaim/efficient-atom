from pathlib import Path
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

import torch
from torch_scatter import scatter
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch

from jmp.tasks.finetune.base import FinetuneConfigBase, FinetuneModelBase
from jmp.modules.scaling.util import ensure_fitted
# from jmp.modules.scaling.compat import load_scales_compat
# from jmp.utils.state_dict_helper import update_gemnet_state_dict_keys

from setup_finetune import load_global_config, get_configs, load_checkpoint, configure_wandb


DATASET_TARGET_MAPPING = {
    "rmd17": "aspirin",
    "qm9": "U_0",
    "md22": "Ac-Ala3-NHMe",
    "qmof": "y",
    "spice": "solvated_amino_acids",
    "matbench": "phonons"
}

global_config = load_global_config()

# Set up argument parser
parser = argparse.ArgumentParser(description="Fine-tuning script for JMP-L")
parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb logging")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
parser.add_argument("--target", type=str, help="Target molecule for the dataset")
parser.add_argument("--fold", type=int, default=0, help="Fold for Matbench dataset")
parser.add_argument("--lr", type=float, default=8.0e-5, help="Learning rate for the optimizer")
parser.add_argument("--num_workers", type=int, default=6, help="Number of workers")
parser.add_argument("--scratch", action="store_true", help="Train from scratch")
parser.add_argument("--large", action="store_true", help="Load the large pre-trained checkpoint")
parser.add_argument("--root_path", type=str, help="Root path containing datasets and checkpoints")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
parser.add_argument("--number_of_samples", type=int, default=10000, help="Number of samples to use for feature extraction")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
# parser.add_argument("--sampling_strategy", type=str, choices=["random"],
#                         default="random", help="Sampling strategy to use: 'random'")
parser.add_argument("--model_name", type=str, default="gemnet", help="Model name")
parser.add_argument("--checkpoint_tag", type=str, default="OC20", help="checkpoint tag")


args = parser.parse_args()
args.root_path = global_config.get("root_path", None)
args.extract_features = True

# Error if the user provides a target manually
if args.target:
    raise ValueError("The '--target' argument is not allowed. Targets are automatically defined based on the dataset in extract_finetine.py.")


# Infer the target from the dataset name
if args.dataset_name in DATASET_TARGET_MAPPING:
    args.target = DATASET_TARGET_MAPPING[args.dataset_name]
    print(f"Assigned target for dataset '{args.dataset_name}': {args.target}")
else:
    raise ValueError(f"No default target defined for dataset '{args.dataset_name}'. Please update the DATASET_TARGET_MAPPING.")


print("The arguments are:", args)

def extract_features(model, args, use_mean_aggregation=False, aggregate_by_atoms=False):
    def collate_fn(data_list):
        return Batch.from_data_list(data_list)

    max_samples = args.number_of_samples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataset = model.train_dataset()
    num_workers = args.num_workers
    dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=collate_fn, batch_size=1, shuffle=True)

    file_suffix = ""
    file_suffix = args.model_name
    if args.checkpoint_tag:
        file_suffix += f"_{args.checkpoint_tag}"

    save_folder = Path(f"dataset_features_{file_suffix}")
    save_folder.mkdir(parents=True, exist_ok=True)
    sample_idx = 0

    extracted_data = []

    for sample in tqdm(dataloader):
        # Skip samples with fewer than 4 atoms as they don't work with GemNet-OC
        if sample.atomic_numbers.shape[0] < 4:
            print(f"Skipping sample with {sample.atomic_numbers.shape[0]} atoms.")
            continue

        sample = sample.to(device)
        features_dict = model(sample)

        # Node-level features
        if features_dict:
            node_features = features_dict['node'].detach().cpu()

            if 'edge' in features_dict:
                edge_features = features_dict['edge'].detach().cpu()
                edge_to_node_mapping = features_dict['idx_t'].detach().cpu()
                collapsed_edge_features = scatter(edge_features, edge_to_node_mapping, dim=0, dim_size=node_features.size(0), reduce="mean")
            else:
                collapsed_edge_features = None

            if collapsed_edge_features is not None:
                extracted_data.append({
                    "node_features": node_features.numpy(),
                    "edge_features": collapsed_edge_features.numpy()
                })
            else:
                extracted_data.append({
                    "node_features": node_features.numpy()
                })

            sample_idx += 1
            if sample_idx == max_samples:
                break


    task_name = args.dataset_name
    if task_name == "matbench":
        task_name = f"{task_name}_fold{args.fold}"
    seed = args.seed
    sampling = "random"

    output_file = f"./{save_folder}/{task_name}_Node_Seed{seed}_Sampling{sampling}.pt"
    torch.save(extracted_data, output_file)
    print(f"Saved features with metadata to {output_file}")


config, model_cls = get_configs(args.dataset_name, args.target, args, extract_features=True)
config.name = "extract"

config.backbone.regress_forces = True
config.backbone.direct_forces = True
if args.checkpoint_tag == "ODAC":
    config.backbone.max_num_elements = 100
elif args.checkpoint_tag == "MP":
    config.backbone.max_num_elements = 96

model = model_cls(config)

# Load the checkpoint
if not args.scratch:
    load_checkpoint(model, config, args)
ensure_fitted(model)

extract_features(model, args)
