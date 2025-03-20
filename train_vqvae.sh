#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=11111 \
    train_vqvae.py \
    --batch_size=16 \
    --num_workers=8 \
    --epochs=400 \
    --dataset="imagenet" \
    --n_class=1000 \
    --embed_dim=1024 \
    --data_path="/data/drstrange/yaxin/data/imagenet" \
    --output_dir="./output_dir/vqvae_gpt2_codebook_imagenet-stage1" \
    --log_dir="./output_dir/vqvae_gpt2_codebook_imagenet-stage1" 
