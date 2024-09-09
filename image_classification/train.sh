#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=visionLM
#SBATCH --output=debug.out
torchrun --nproc_per_node 4 main.py --exp_name test_llama --model vqgan_llama_small_patch16_224 \
    --data-path /l/users/zhiqiang.shen/visual_tokenizer/data/ImageNet --output_dir /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/vqgan_llama_classification\
    --num_workers 32 --batch-size 2 --epochs 300 --warmup-epochs 20 --llama_path /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/llama/llama-2-7b\
