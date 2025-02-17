#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    --nnodes=1 \
    --node_rank=0 \
    train_vqvae.py \
    --batch_size=16 \
    --num_workers=24 \
    --epochs=400 \
    --distributed \
    --use_mod True \
    --capacity_factor=0.75 \
    --router_aux_loss_coef=0.01 \
    --output_dir="./output_dir/mod_vqvae_roberta_cifar10" \
    --log_dir="./output_dir/mod_vqvae_roberta_cifar10" 