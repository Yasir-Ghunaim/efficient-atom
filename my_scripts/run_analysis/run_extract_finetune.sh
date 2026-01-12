#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name pretrain
#SBATCH -o output_pretrain/gpu.%A.out
#SBATCH -e output_pretrain/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6

#  
#####
hostname
nvidia-smi
conda activate efficient_atom
##### 

cd ..

tasks=("rmd17" "md22" "qm9" "spice" "qmof" "matbench")

# Iterate over all tasks
for task in "${tasks[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python extract_finetune.py \
    --dataset_name $task \
    --batch_size 1 \
    --number_of_samples 10000 \
    --model_name "equiformer_v2" \
    --seed 0
done