#!/bin/bash
#SBATCH --job-name=train_vqvae
#SBATCH --output=./output_dir/slurm_%A.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -p cscc-gpu-p
#SBATCH --time=72:00:00
#SBATCH -q cscc-gpu-qos

export NUM_GPUS=4

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_vqvae.py \
    --batch_size=1 \
    --epochs=400 \
    --distributed \
    --output_dir="./output_dir/vqvae_imagenet_1k" \
    --log_dir="./output_dir/vqvae_imagenet_1k" 