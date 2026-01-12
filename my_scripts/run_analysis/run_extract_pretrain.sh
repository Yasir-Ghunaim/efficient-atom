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

# Define arrays for tasks, sampling strategies, and model names
tasks=("ani1x" "transition1x" "oc20" "oc22")

# Iterate over all tasks
for task in "${tasks[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python extract_pretrain.py \
    --task $task \
    --batch_size 5 \
    --train_samples_limit 10000 \
    --sampling_strategy "balanced" \
    --model_name "equiformer_v2" \
    --seed 0
done
