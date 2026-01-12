import os
from pathlib import Path
import torch
import yaml

from jmp.configs.finetune.jmp_l import jmp_l_ft_config_
from jmp.configs.finetune.rmd17 import jmp_l_rmd17_config_
from jmp.configs.finetune.qm9 import jmp_l_qm9_config_
from jmp.configs.finetune.md22 import jmp_l_md22_config_
from jmp.configs.finetune.spice import jmp_l_spice_config_
from jmp.configs.finetune.matbench import jmp_l_matbench_config_
from jmp.configs.finetune.qmof import jmp_l_qmof_config_

from jmp.tasks.finetune.rmd17 import RMD17Config, RMD17Model
from jmp.tasks.finetune.qm9 import QM9Config, QM9Model
from jmp.tasks.finetune.md22 import MD22Config, MD22Model
from jmp.tasks.finetune.spice import SPICEConfig, SPICEModel
from jmp.tasks.finetune.matbench import MatbenchConfig, MatbenchModel
from jmp.tasks.finetune.qmof import QMOFConfig, QMOFModel

from jmp.tasks.finetune.module_extract_features import (
    RMD17ModelFeatureExtraction,
    QM9ModelFeatureExtraction,
    MD22ModelFeatureExtraction,
    SPICEModelFeatureExtraction,
    MatbenchModelFeatureExtraction,
    QMOFModelFeatureExtraction
    )

from jmp.utils.finetune_state_dict import (
    filter_state_dict,
    retreive_state_dict_for_finetuning,
)

def load_global_config(filename="global_config.yaml"):
    config_path = Path(__file__).resolve().parent.parent / filename
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_configs(dataset_name: str, target: str, args, extract_features = False):
    if args.model_name == "gemnet":
        ckpt_path = Path(args.root_path) / "checkpoints/JMP/jmp-s.pt"
        if args.large:
            ckpt_path = Path(args.root_path) / "checkpoints/JMP/jmp-l.pt"
        
        print("Loading checkpoint path:", ckpt_path)
    elif args.model_name == "equiformer_v2":
        ckpt_path = Path(args.root_path) / "checkpoints/EquiformerV2/eq2_31M_ec4_allmd.pt"
        if extract_features:
            if args.checkpoint_tag == "ODAC":
                ckpt_path = Path(args.root_path) / "checkpoints/EquiformerV2/eqv2_31M_odac_new.pt"
            elif args.checkpoint_tag == "MP":
                ckpt_path = Path(args.root_path) / "checkpoints/EquiformerV2/eqV2_31M_mp.pt"
        print("Setting checkpoint path:", ckpt_path)


    dataset_name = dataset_name.lower()
    base_path = Path(args.root_path) / f"datasets/{dataset_name}/"
    
    if dataset_name == "rmd17":
        config = RMD17Config.draft()
        jmp_l_ft_config_(config, ckpt_path, args=args)
        jmp_l_rmd17_config_(config, target, base_path, args=args)
        init_model = RMD17ModelFeatureExtraction if extract_features else RMD17Model

    elif dataset_name == "qm9":
        config = QM9Config.draft()
        jmp_l_ft_config_(config, ckpt_path, args=args)  
        jmp_l_qm9_config_(config, target, base_path, args=args)
        init_model = QM9ModelFeatureExtraction if extract_features else QM9Model

    elif dataset_name == "md22":
        config = MD22Config.draft()
        jmp_l_ft_config_(config, ckpt_path, args=args)  
        jmp_l_md22_config_(config, target, base_path, args=args) 
        init_model = MD22ModelFeatureExtraction if extract_features else MD22Model

    elif dataset_name == "spice":
        config = SPICEConfig.draft()
        jmp_l_ft_config_(config, ckpt_path, args=args)  
        jmp_l_spice_config_(config, target, base_path, args=args)
        init_model = SPICEModelFeatureExtraction if extract_features else SPICEModel

    elif dataset_name == "matbench":
        config = MatbenchConfig.draft()
        jmp_l_ft_config_(config, ckpt_path, args=args)  
        jmp_l_matbench_config_(config, target, args.fold, base_path, args=args)
        init_model = MatbenchModelFeatureExtraction if extract_features else MatbenchModel

    elif dataset_name == "qmof":
        config = QMOFConfig.draft()
        jmp_l_ft_config_(config, ckpt_path, args=args)  
        jmp_l_qmof_config_(config, base_path, target=target, args=args)
        init_model = QMOFModelFeatureExtraction if extract_features else QMOFModel
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config.args = args
    config.model_name = args.model_name
    config = config.finalize()
    configure_wandb(config, args)
    print(config)
    return config, init_model

def remove_dens_blocks(state_dict):
    prefixes = (
        "module.energy_block",
        "module.force_block",
        "module.denoising_pos_block",
        "module.energy_lin_ref",
        "module.force_embedding.weight",
        "module.force_embedding.bias",
        "module.force_embedding.expand_index",
    )
    return {
        k: v for k, v in state_dict.items()
        if not k.startswith(prefixes)
    }

def rename_module_to_backbone(state_dict):
    return {
        (k.replace("module.", "backbone.", 1) if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

def load_checkpoint(model, config, args):
    # Here, we load our fine-tuned model on the same task
    if hasattr(args, "checkpoint_path") and args.checkpoint_path:

        print("Loading custom checkpoint =================")
        root = Path(args.root_path) / "checkpoints/MSI/"
        ckpt_path = root / (args.checkpoint_path + ".ckpt")
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        if args.model_name == "equiformer_v2":

            # Trigger DeNS-specific cleanup ONLY if this key exists
            if any(k.startswith("module.denoising_pos_block") for k in state_dict):
                print("Detected DeNS checkpoint.")
                state_dict = remove_dens_blocks(state_dict)
                state_dict = rename_module_to_backbone(state_dict)

            backbone_state_dict = filter_state_dict(state_dict, "backbone.")
            model.backbone.load_state_dict(backbone_state_dict)
            print("Loaded checkpoint for equiformer_v2")
        else:
            model.load_state_dict(state_dict, strict=False)
        
    
    # Here, we load Meta-AI original foundation model (without the heads)
    else:
        print("Loading pretraining checkpoint =================")
        if (ckpt_path := config.meta.get("ckpt_path")) is None:
            raise ValueError("No checkpoint path provided")

        if config.model_name == "gemnet":
            state_dict = retreive_state_dict_for_finetuning(
                ckpt_path, load_emas=config.meta.get("ema_backbone", False)
            )
            embedding = filter_state_dict(state_dict, "embedding.atom_embedding.")
            backbone = filter_state_dict(state_dict, "backbone.")
            model.load_backbone_state_dict(backbone=backbone, embedding=embedding, strict=True)
        elif config.model_name == "equiformer_v2":
            print("Loading checkpoint:", config.meta.get("ckpt_path"))
            eqv2_state_dict = torch.load(config.meta.get("ckpt_path"))['state_dict']

            # Fix the keys by replacing "module.module" with "backbone"
            eqv2_state_dict = {
                key.replace("module.module", "backbone"): value
                for key, value in eqv2_state_dict.items()
            }

            if hasattr(args, "checkpoint_tag") and args.checkpoint_tag == "MP":
                eqv2_state_dict = {
                    key.replace("module.", ""): value
                    for key, value in eqv2_state_dict.items()
                }

            # Remove keys starting with "backbone.energy_block" and "backbone.force_block"
            state_dict = {
                key: value
                for key, value in eqv2_state_dict.items()
                if not (key.startswith("backbone.energy_block") or key.startswith("backbone.force_block"))
            }

            backbone_state_dict = filter_state_dict(state_dict, "backbone.")
            model.backbone.load_state_dict(backbone_state_dict)
            print("Loaded checkpoint for equiformer_v2")
            # model.load_state_dict(state_dict)


def configure_wandb(config, args):
    """Configure WandB settings."""
    if args.enable_wandb:
        # Define WandB parameters
        global_config = load_global_config()
        config.project = global_config.get("wandb").get("finetune_project")

    else:
        # Disable wandb logging (for debugging)
        config.trainer.logging.wandb.enabled = False