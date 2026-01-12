#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name SPICE
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=29:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6

#####
hostname
nvidia-smi
conda activate efficient_atom
##### 

cd ..
CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --dataset_name "spice" \
    --target "solvated_amino_acids" \
    --lr 8.0e-5 \
    --epochs 250 \
    --enable_wandb \
    --checkpoint_path "<checkpoint_name>"