#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=11111 \
    train_vqvae.py \
    --batch_size=8 \
    --num_workers=8 \
    --epochs=400 \
    --dataset="mini-imagenet" \
    --n_class=1000 \
    --data_path="/data/drstrange/yaxin/data/mini-imagenet" \
    --output_dir="./output_dir/vqvae_gpt2_mini_imagenet" \
    --log_dir="./output_dir/vqvae_gpt2_mini_imagenet" \
    --resume /data/drstrange/yaxin/Projects/vision_llm/output_dir/vqvae_gpt2_mini_imagenet/vqvae_checkpoint-last.pth