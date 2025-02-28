#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=29500 \
    train_var.py \
    --batch_size=8 \
    --num_workers=8 \
    --epochs=400 \
    --distributed \
    --dataset="mini-imagenet" \
    --n_class=1000 \
    --rate_q=1.0 \
    --data_path="/data/drstrange/yaxin/data/mini-imagenet" \
    --output_dir="./output_dir/var_pretrained_gpt2-medium_mini_imagenet-frozen-llm" \
    --log_dir="./output_dir/var_pretrained_gpt2-medium_mini_imagenet-frozen-llm" 