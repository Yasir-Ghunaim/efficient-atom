#!/bin/bash --login
#SBATCH -N 1
#SBATCH --job-name MatBench
#SBATCH -o output/gpu.%A.out
#SBATCH -e output/gpu.%A.err
#SBATCH --mail-type=FAIL
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=6

#####
hostname
nvidia-smi
conda activate efficient_atom
##### 

cd ..
#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --dataset_name "matbench" \
    --target "phonons" \
    --fold 0 \
    --lr 8.0e-5 \
    --epochs 120 \
    --enable_wandb \
    --checkpoint_path "<checkpoint_name>"