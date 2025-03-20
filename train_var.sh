#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=29500 \
    train_var.py \
    --batch_size=16 \
    --num_workers=8 \
    --epochs=400 \
    --distributed \
    --dataset="imagenet" \
    --n_class=1000 \
    --embed_dim=1024 \
    --data_path="/data/drstrange/yaxin/data/imagenet" \
    --output_dir="./output_dir/pretrain_vision_gpt2-medium_imagenet_1024-stage2" \
    --log_dir="./output_dir/pretrain_vision_gpt2-medium_imagenet_1024-stage2" \
    --resume /data/drstrange/yaxin/Projects/vision_llm/output_dir/vqvae_gpt2_codebook_imagenet-stage1/vqvae_checkpoint-20.pth