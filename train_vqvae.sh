#!/bin/bash

export NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_vqvae.py \
    --batch_size=8 \
    --epochs=400 \
    --distributed \
    --output_dir="./output_dir/vqvae_cifar10" \
    --log_dir="./output_dir/vqvae_cifar10" 