#!/bin/bash
#SBATCH --nodes=3 
#SBATCH --exclusive
#SBATCH --time=72:00:00 
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=V2Ltoken


imagenet_path="/l/users/zhiqiang.shen/visual_tokenizer/data/ImageNet"
log_dir="/l/users/zhiqiang.shen/visual_tokenizer/checkpoints/V2LTokenizer"
llama_path="/home/zhiqiang.shen/projects/visual_tokenizer/V2L-Tokenizer/llama_inference/llama-2-7b"

# ####Expand Global Vocabulary Set
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=12247 step1_epanding_vocabulary_set.py \
#     --batch_size 400 \
#     --max_seq_len 64 \
#     --llama_model_path $llama_path

# ####Generate Embedding for Vocabularies
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=12247 step2_generate_codebook_embedding.py

# ####Filtering Vocabularies with Training Images
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=12247 step3_global_codebook_filtering.py \
#     --batch_size 2 \
#     --imagenet_path $imagenet_path \
#     --num_workers 16

####Training V2L Tokenizer
torchrun --nproc_per_node 4 step4_training_v2l_tokenizer.py \
    --batch_size 32 \
    --image_size 128 \
    --epochs 100 \
    --warmup_epochs 5 \
    --lr 4.5e-4 \
    --n_class 1000 \
    --imagenet_path $imagenet_path \
    --num_workers 16 \
    --rate_q 1 \
    --rate_p 0.1 \
    --vq_config_path vqgan_configs/v2l.yaml \
    --output_dir $log_dir \
    --log_dir $log_dir \
    --disc_start 10000 \
    --local_embedding_path /home/zhiqiang.shen/projects/visual_tokenizer/LM4VisualEncoding/image_classification/codebook/local_codebook_embedding.pth \
    --tuning_codebook 0 \
    --use_cblinear 1 \
    --use_crossatt_dec 0 \
    --embed_dim 768

