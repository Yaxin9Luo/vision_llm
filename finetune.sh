#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --exclusive
#SBATCH --time=72:00:00 
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=finetune
#SBATCH --output logs/train_dataaug_gemma.out

############### multinodes training ###############
# torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node 4 --rdzv_id=1009 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29401 stage_2_training.py \
    --batch_size 256 \
    --image_size 128 \
    --epochs 100 \
    --warmup_epochs 5 \
    --lr 4.5e-4 \
    --n_class 1000 \
    --imagenet_path /l/users/zhiqiang.shen/visual_tokenizer/data/ImageNet \
    --num_workers 16 \
    --rate_q 1 \
    --rate_p 0.1 \
    --n_vision_words 32000 \
    --vq_config_path vqgan_configs/v2l.yaml \
    --output_dir /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/gemma_vqgan_stage2_train \
    --log_dir /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/gemma_vqgan_stage2_train \
    --disc_start 10000 \
    --use_cblinear 0 \
    --embed_dim 768

############### single node training ###############
torchrun --nproc_per_node 4 stage_2_train.py \
    --batch_size 512 \
    --image_size 128 \
    --epochs 300 \
    --warmup_epochs 5 \
    --lr 4.5e-4 \
    --class_num 1000 \
    --imagenet_path /l/users/zhiqiang.shen/visual_tokenizer/data/ImageNet \
    --vqgan_ckt /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/V2LTokenizer/vqgan_checkpoint-last.pth \
    --num_workers 16 \
    --rate_q 1 \
    --rate_p 0.1 \
    --n_vision_words 32000 \
    --vq_config_path vqgan_configs/v2l.yaml \
    --output_dir /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/gemma_vqgan_stage2_train_dataaug \
    --log_dir /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/gemma_vqgan_stage2_train_dataaug \
    --disc_start 10000 \
    --use_cblinear 0 \
    --train_classifier True \
    --embed_dim 768
