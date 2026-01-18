# On the Importance of Pretraining Data Alignment for Atomic Property Prediction, TMLR 2026

## Introduction
This repository is for our **TMLR (Transactions on Machine Learning Research)** paper "On the Importance of Pretraining Data Alignment for Atomic Property Prediction", which introduces an efficient pretraining paradigm for atomic property prediction. Our work demonstrates that pretraining on a strategically selected, task-relevant dataset can achieve comparable or superior performance to large-scale datasets at a fraction of the computational cost. We also present the Chemical Similarity Index (CSI), inspired by Fréchet Inception Distance, to guide upstream data selection and significantly improve model efficiency.


## Installation
To set up the conda environment, run the following command:
```bash
conda env create -f environment.yml -n efficient_atom
```

Then, activate the environment and install the package:
```bash
conda activate efficient_atom
pip install -e .
```

## Datasets and Checkpoints
For all datasets and checkpoints, the root directory is defined in the `global_config.yaml` file located in the main directory. Update the `root_path` value in this file if needed:
```yaml
# global_config.yaml
root_path: "/path/to/your/root_path"
```

The following subdirectories are expected within the root_path:
- Datasets: `<root_path>/datasets`
- Checkpoints: `<root_path>/checkpoints`


### Pre-training Datasets
Follow the instructions in the [Joint Multi-domain Pre-Training (JMP)](https://github.com/facebookresearch/JMP/blob/main/README.md) repository to install and preprocess the pretraining datasets: OC20, OC22, ANI-1x, and Transition-1x.


### Fine-tuning Datasets
Download the downstream datasets with the following commands:
- rMD17: `python -m jmp.datasets.finetune.rmd17 download --destination <root_path>/datasets/rmd17/`
- QM9: `python -m jmp.datasets.finetune.qm9 download --destination <root_path>/datasets/qm9/`
- MD22: `python -m jmp.datasets.finetune.md22 download --destination <root_path>/datasets/md22/`
- QMOF: `python -m jmp.datasets.finetune.qmof download --destination <root_path>/datasets/qmof/`
- SPICE: `python -m jmp.datasets.finetune.spice download --destination <root_path>/datasets/spice/`
- MatBench: `python -m jmp.datasets.finetune.mat_bench download --destination <root_path>/datasets/matbench/`

## Pre-trained Checkpoints
Download the available pre-trained checkpoints for JMP:
- [JMP-S](https://jmp-iclr-datasets.s3.amazonaws.com/jmp-s.pt)
- [JMP-L](https://jmp-iclr-datasets.s3.amazonaws.com/jmp-l.pt)

Download the checkpoint for EquiformerV2 used for feature extraction:
- [EquiformerV2 - OC20](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt)

These files should be saved under `<root_path>/checkpoints/<model_name>`, where:
- `root_path` is the directory containing datasets and checkpoints.
- `model_name` should be set to `"GemNet"` for JMP checkpoints or `"EquiformerV2"` for the EquiformerV2 model.


## Feature Extraction For CSI 

### Preparing Class Mapping for Upstream Datasets

After downloading the datasets, define the paths to OC20/OC22 in `scripts/generate_structure_mapping.py` within the `"__main__"` section, marked as "TODO". Then, run the following command:
```bash
python scripts/generate_structure_mapping.py
```
This script generates text files that map each training sample to its structure name, which is required for class balancing. Note this script should be called twice, for OC20 and OC22.

### Feature Extraction
To extract features, run the following scripts:

```bash
cd my_scripts/run_analysis

# Extract features for upstream datasets
bash run_extract_pretrain.sh

# Extract features for downstream datasets
bash run_extract_finetune.sh
```

All extracted features will be saved under the folder `"my_scripts/dataset_features_equiformer_v2"`.

### Computing CSI similarity 
To compute the CSI scores, run the following command:
```bash
cd my_scripts/run_analysis
bash compute_CSI_all.sh
```
This will generate the similarity scores for guiding upstream data selection.


## Pretraining Instructions
We provide a starting script to help with reproducing our pretraining experiments:
```bash
cd my_scripts/run_exp
bash run_pretrain.sh
```

- To run pretraining on individual datasets, update the `--task` argument in `run_pretrain.sh`. For example, to pretrain on ANI-1x only, set:
```bash
--task "ani1x"
```

- To run pretraining on all available datasets (mixed pre-training), set:
```bash
--task "oc20,oc22,ani1x,transition1x"
```
You can also specify the `sampling_strategy` as `random` or class-balanced with `balanced` for the mixed pre-training.

Upon completion, the best validation checkpoint will be stored in "my_scripts/lightning_logs/<wandb_run_id>". where `<wandb_run_id>` corresponds to the WandB run ID for the pretraining experiment. 


**Important:**  
Move the best checkpoint to:
```bash
    <root_path>/checkpoints/<checkpoint_name>
```
so it can be accessed later for fine-tuning.

## Fine-Tuning Instructions
We provide dedicated scripts in `my_scripts/run_exp` for fine-tuning on each of the six downstream tasks: rMD17, QM9, MD22, QMOF, SPICE, and MatBench.

To fine-tune on a specific task (e.g., rMD17), run:
```bash
cd my_scripts/run_exp
bash run_finetune_rmd17.sh
```

Example structure for `run_finetune_rmd17.sh`:
```bash
python finetune.py \
    --dataset_name "rmd17" \
    --target "aspirin" \
    --lr 8.0e-5 \
    --epochs 600 \
    --enable_wandb \
    --checkpoint_path "<checkpoint_name>"
```
Note: <checkpoint_name> should match the name of the pretraining checkpoint saved in:
<root_path>/checkpoints/<checkpoint_name>. 

## License
This project follows the same license as JMP:

The majority of JMP is CC-BY-NC licensed, as found in the `LICENSE` file. However, portions of the project are available under separate license terms:

- The ASE library is licensed under the GNU Lesser General Public License v2.1.
- PyTorch Lightning and TorchMetrics are licensed under the Apache License 2.0.
- DeepChem is licensed under the MIT License.
- RDKit is licensed under the BSD 3-Clause License.
- Biopython is licensed under the Biopython License Agreement.
- Pydantic is licensed under the MIT License.
- MatBench is licensed under the MIT License.
- Submitit is licensed under the MIT License.
- Model implementations are based on the Open Catalyst Project and are licensed under the MIT License.
- EMA implementation is based on the NeMo's implementation and is licensed under the Apache License 2.0.

## Citation

If you find this work useful, please cite our TMLR paper:

```bibtex
@article{
ghunaim2026on,
title={On the Importance of Pretraining Data Alignment for Atomic Property Prediction},
author={Yasir M. Ghunaim and Hasan Abed Al Kader Hammoud and Bernard Ghanem},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2026},
url={https://openreview.net/forum?id=jfD9BsrDTb},
note={}
}
```
