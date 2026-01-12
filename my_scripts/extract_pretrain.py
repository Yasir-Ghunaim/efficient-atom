from pathlib import Path
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

import torch
from torch_scatter import scatter
from torch.utils.data import DataLoader
from torch_geometric.data.batch import Batch
from jmp.modules.scaling.util import ensure_fitted
from jmp.modules.scaling.compat import load_scales_compat
from jmp.tasks.pretrain import PretrainConfig, PretrainModel, PretrainModelWithFeatureExtraction
# from jmp.utils.state_dict_helper import update_gemnet_state_dict_keys
from setup_pretrain import load_global_config, configure_model

from jmp.utils.finetune_state_dict import filter_state_dict

# Load global config
global_config = load_global_config()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pre-training script for JMP-L")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of workers")
    parser.add_argument("--train_samples_limit", type=int, default=10000, help="Number of training samples to use")
    parser.add_argument("--val_samples_limit", type=int, default=-1, help="Number of validation samples to use")
    parser.add_argument("--large", action="store_true", help="Load the large pre-trained checkpoint")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--root_path", type=str, help="Root path containing datasets and checkpoints")
    parser.add_argument("--task", type=str, choices=["oc20", "oc22", "ani1x", "transition1x"],
                        required=True, help="Name of the pretraining task. Choose from: oc20, oc22, ani1x, transition1x.")
    parser.add_argument("--sampling_strategy", type=str, choices=["random", "balanced", "balancedNoRep", "stratified", "uniform"],
                        default="random", help="Sampling strategy to use: 'random', 'balanced', 'balancedNoRep', 'stratified', or 'uniform'")
    parser.add_argument("--model_name", type=str, default="gemnet", help="Model name")
    parser.add_argument("--temperature_sampling", action="store_true", help="Use temperature sampling equal to 2")
    parser.add_argument("--ani1x_ood", action="store_true", help="Custom sampling for ani1x ood experiment")
    parser.add_argument("--compute_sample_difficulty", action="store_true",
                        help="If set, compute per-sample difficulty using model loss")
    parser.add_argument("--checkpoint_tag", type=str, default="OC20", help="checkpoint tag")

    args = parser.parse_args()
    args.root_path = global_config.get("root_path", None)
    return args




def get_dataset_and_model(config, args):
    """Returns the dataset and the model."""
    if args.compute_sample_difficulty:
        model = PretrainModel(config)
    
    else:
        model = PretrainModelWithFeatureExtraction(config)
        
    if config.model_name == "gemnet":
        checkpoint_path = Path(args.root_path) / "checkpoints/GemNet"
        if args.large:
            path = checkpoint_path / "jmp-l.pt"
            state_dict = torch.load(path)['state_dict']
        else:
            path = checkpoint_path / "jmp-s.pt"
            state_dict = torch.load(path)['state_dict']
        
        print("Loading checkpoint path:", path)
        
        model.load_state_dict(state_dict, strict=False)
        dataset = model.train_dataset()
        return dataset, model


    elif config.model_name == "equiformer_v2":
        checkpoint_path = Path(args.root_path) / "checkpoints/EquiformerV2"
        full_path = checkpoint_path / "eq2_31M_ec4_allmd.pt"
        if args.checkpoint_tag == "ODAC":
            full_path = checkpoint_path / "eqv2_31M_odac_new.pt"
        elif args.checkpoint_tag == "MP":
            full_path = checkpoint_path / "eqV2_31M_mp.pt"

        print("Loading checkpoint:", full_path)
        full_state_dict = torch.load(full_path)['state_dict']

    # Fix the keys by replacing "module.module" with "backbone"
    full_state_dict = {
        key.replace("module.module", "backbone"): value
        for key, value in full_state_dict.items()
    }

    if args.checkpoint_tag == "MP":
        full_state_dict = {
            key.replace("module.", ""): value
            for key, value in full_state_dict.items()
        }
    

    if args.compute_sample_difficulty:
        # Replace energy_block keys
        state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith("backbone.energy_block"):
                # Convert to: output.energy_heads.0.energy_block...
                new_key = key.replace("backbone.energy_block", "output.energy_heads.0.energy_block")
                state_dict[new_key] = value
            elif key.startswith("backbone.force_block"):
                # Convert to: output.force_heads.0.force_block...
                new_key = key.replace("backbone.force_block", "output.force_heads.0.force_block")
                state_dict[new_key] = value
            else:
                # Keep all other keys unchanged
                state_dict[key] = value
    else:
        # Remove keys starting with "backbone.energy_block" and "backbone.force_block"
        state_dict = {
            key: value
            for key, value in full_state_dict.items()
            if not (key.startswith("backbone.energy_block") or key.startswith("backbone.force_block"))
        }

    if config.model_name == "equiformer_v2":
        backbone_state_dict = filter_state_dict(state_dict, "backbone.")
        model.backbone.load_state_dict(backbone_state_dict)
        print("======= Loaded checkpoint for equiformer_v2")
    else:
        model.load_state_dict(state_dict, strict=False)
        print("======= Loaded checkpoint for gemnet")

    # if config.model_name == "gemnet":
    #     load_scales_compat(model.backbone, scale_file)

    dataset = model.train_dataset()
    return dataset, model


# Feature extraction function
def extract_features(model, dataset, config, args, use_mean_aggregation=False, aggregate_by_atoms=False):
    def collate_fn(data_list):
        return Batch.from_data_list(data_list)

    max_samples = args.train_samples_limit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("Model name:", model.backbone.__class__.__name__)
    num_parameters = sum(p.numel() for p in model.parameters())  # Count total parameters
    print(f"Number of parameters: {num_parameters}")
    
    model.eval()

    num_workers = args.num_workers
    dataloader = DataLoader(dataset, num_workers=num_workers, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)

    # Create a dictionary to count instances of each molecule name
    molecule_counts = defaultdict(int)    


    file_suffix = ""
    file_suffix = args.model_name
    if args.checkpoint_tag:
        file_suffix += f"_{args.checkpoint_tag}"

    save_folder = Path(f"dataset_features_{file_suffix}")
    save_folder.mkdir(parents=True, exist_ok=True)

    extracted_data = []

    for sample in tqdm(dataloader): 

        for molecule_name in sample["molecule_name"]:
            molecule_counts[molecule_name] += 1

        sample = sample.to(device)
        features_dict = model(sample)

        node_features = features_dict['node'].detach().cpu()  # Node features

        batch_indices = sample.batch.detach().cpu()  # Maps each node to a graph index
        sids = sample.sid.detach().cpu()  # List of graph SIDs
        fids = sample.fid.detach().cpu()
        lmdb_idx = sample.lmdb_idx.detach().cpu()


        if 'edge' in features_dict:
            edge_features = features_dict['edge'].detach().cpu()  # Edge features
            edge_to_node_mapping = features_dict['idx_t'].detach().cpu()  # Edge-to-node mapping
            collapsed_edge_features = scatter(edge_features, edge_to_node_mapping, dim=0, dim_size=node_features.size(0), reduce="mean")

        # Map features to their graph indices
        for graph_idx in range(len(sids)):
            node_indices = (batch_indices == graph_idx).nonzero(as_tuple=True)[0]
            # edge_indices = torch.where(torch.isin(edge_to_node_mapping, node_indices))[0]

            graph_features = node_features[node_indices]  # All node features for this graph
            # graph_edge_features = edge_features[edge_indices]
            if 'edge' in features_dict:
                graph_edge_features = collapsed_edge_features[node_indices]
            else:
                graph_edge_features = None

            if graph_edge_features is not None:
                extracted_data.append({
                    "index": lmdb_idx[graph_idx].item(),
                    "sid": sids[graph_idx].item(),
                    "fid": fids[graph_idx].item(),
                    "node_features": graph_features.numpy(),  # Node features for this graph
                    "edge_features": graph_edge_features.numpy(),  # edge features for this graph
                })
            else:
                extracted_data.append({
                    "index": lmdb_idx[graph_idx].item(),
                    "sid": sids[graph_idx].item(),
                    "fid": fids[graph_idx].item(),
                    "node_features": graph_features.numpy(),  # Node features for this graph
                })

    
    # Sort molecule counts by count in descending order
    sorted_molecule_counts = sorted(molecule_counts.items(), key=lambda x: x[1], reverse=True)
    print("Length of counts are:", len(molecule_counts))


    task_name = args.task
    seed = args.seed
    sampling = args.sampling_strategy

    output_file = f"./{save_folder}/{task_name}_Node_Seed{seed}_Sampling{sampling}.pt"
    torch.save(extracted_data, output_file)
    print(f"Saved features with metadata to {output_file}")


def compute_sample_wise_losses(
    model, batch, energy: torch.Tensor, forces: torch.Tensor
) -> torch.Tensor:
    """Returns per-graph total loss (energy + force) without reducing over the batch."""

    # ---- Energy loss ----
    energy_loss, energy_loss_mask = model._energy_loss(batch, energy)  # (b, t)
    energy_loss_per_graph = model._safe_divide(
        energy_loss.sum(dim=1), energy_loss_mask.sum(dim=1)
    )  # (b,)

    # ---- Force loss ----
    force_loss, force_loss_mask = model._force_loss(batch, forces)  # (n, t), (n, t)
    if model.config.structurewise_loss_reduction:
        # Structure-wise loss per graph
        force_loss = scatter(force_loss, batch.batch, dim=0, reduce="sum")  # (b, t)
        force_loss_mask_natoms = scatter(
            force_loss_mask.float(), batch.batch, dim=0, reduce="sum"
        )  # (b, t)
        force_loss = model._safe_divide(force_loss, force_loss_mask_natoms)  # (b, t)
        force_loss_mask = force_loss_mask_natoms > 0.0  # (b, t)

    force_loss_per_graph = model._safe_divide(
        force_loss.sum(dim=1), force_loss_mask.sum(dim=1)
    )  # (b,)

    # ---- Total loss per graph ----
    total_loss = energy_loss_per_graph + force_loss_per_graph  # (b,)

    return total_loss  # (batch_size,)



def compute_sample_difficulty(model, dataset, config, args):
    """Compute per-sample difficulty using loss."""
    def collate_fn(data_list):
        return Batch.from_data_list(data_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=collate_fn, 
        shuffle=False
    )

    sample_difficulties = []

    for batch in tqdm(dataloader, desc="Computing sample difficulty"):
        batch = batch.to(device)

        with torch.no_grad():
            energy, forces = model(batch)
            per_graph_loss = compute_sample_wise_losses(model, batch, energy=energy, forces=forces).detach().cpu()


        for i in range(batch.num_graphs):
            sample_difficulties.append({
                "lmdb_idx": batch.lmdb_idx[i].item(),
                "sid": batch.sid[i].item(),
                "fid": batch.fid[i].item(),
                "loss": per_graph_loss[i].item()
            })


    # Save results
    task_name = args.task
    seed = args.seed
    model_name = args.model_name
    output_path = f"./sample_difficulty_{model_name}/{task_name}_Seed{seed}.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(sample_difficulties, output_path)
    print(f"Saved sample difficulties to {output_path}")

def main():
    args = parse_args()
    args.extract_features = True
    config = configure_model(args)
    config.name = "extract"
    config.trainer.logging.wandb.enabled = False

    print("The arguments are:", args)
    print(config)

    if args.checkpoint_tag == "ODAC":
        config.backbone.max_num_elements = 100
    elif args.checkpoint_tag == "MP":
        config.backbone.max_num_elements = 96

    dataset, model = get_dataset_and_model(config, args)
    ensure_fitted(model)

    if args.compute_sample_difficulty:
        compute_sample_difficulty(model, dataset, config, args)
    else:
        extract_features(model, dataset, config, args)

if __name__ == "__main__":
    main()
