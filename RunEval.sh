#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --exclusive
#SBATCH --time=72:00:00 
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=V2Ltoken
#SBATCH --output debug_classification.out

##Evaluating Reconstruction
torchrun --nproc_per_node 1 eval.py \
    --batch_size 1 \
    --image_size 128 \
    --class_num 1000 \
    --imagenet_path /l/users/zhiqiang.shen/visual_tokenizer/data/ImageNet \
    --n_vision_words 32000 \
    --llama_model_path /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/llama/llama-2-7b \
    --vq_config_path vqgan_configs/v2l.yaml \
    --output_dir /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/V2LTokenizer \
    --log_dir /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/V2LTokenizer \
    --quantizer_type "org" \
    --stage_1_ckpt "/l/users/zhiqiang.shen/visual_tokenizer/checkpoints/V2LTokenizer/vqgan_checkpoint-last.pth" \
    --embed_dim 768 \
    --use_cblinear 0 \

##Evaluating Few-shot Classification (N-way K-Shot M-Repeat)
# token_nums=(5 21)
# ways=(5 2)
# shots=(3 5 1)
# for (( k = 0 ; k < ${#token_nums[@]} ; k++ ))
# do
#     token_num=${token_nums[$k]}
#     for (( i = 0 ; i < ${#ways[@]} ; i++ ))
#     do
#         way=${ways[$i]}
#         for (( j = 0 ; j < ${#shots[@]} ; j++ ))
#         do
#         shot=${shots[$j]}
#         torchrun --nproc_per_node 1 eval_understanding.py \
#                     --image_size 128 \
#                     --n_class 1000 \
#                     --batch_size 1 \
#                     --max_seq_len 1024 \
#                     --num_workers 0 \
#                     --output_type "next_token_prediction" \
#                     --imagenet_path /l/users/zhiqiang.shen/visual_tokenizer/data/ImageNet \
#                     --vq_config_path vqgan_configs/v2l.yaml \
#                     --output_dir "log_eval_few_shot/7B_"$token_num"_"$way"_"$shot \
#                     --llama_model_path /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/llama/llama-2-7b \
#                     --induction 1 \
#                     --stage_1_ckpt "/l/users/zhiqiang.shen/visual_tokenizer/checkpoints/V2LTokenizer/vqgan_checkpoint-last.pth" \
#                     --embed_dim 768 \
#                     --quantizer_type "org" \
#                     --use_cblinear 0 \
#                     --way $way \
#                     --shot $shot \
#                     --token_num $token_num \
#                     --repeat 0
#         done
#     done
# done

# ##Evaluating Denoising Generation
# tasks=("deblur" "rotation" "shift" "inpainting" "outpainting")
# for (( k = 0 ; k < ${#tasks[@]} ; k++ ))
# do
#     task=${tasks[$k]}
#     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port=12593 eval_denoising_generation.py \
#                             --image_size 128 \
#                             --n_class 1000 \
#                             --max_seq_len 512 \
#                             --num_workers 0 \
#                             --output_type "next_token_prediction" \
#                             --vq_config_path vqgan_configs/v2l.yaml \
#                             --output_dir "log_eval_"$task"/7B_clip_linear" \
#                             --quantizer_type "org" \
#                             --llama_model_path /data/llama2-origin-format/llama-2-7b \
#                             --embed_dim 768 \
#                             --n_vision_words 32000 \
#                             --local_embedding_path "codebooks/local_codebook_embedding.pth" \
#                             --global_embedding_path "codebooks/global_codebook_embedding.pth" \
#                             --stage_1_ckpt "checkpoints/v2l-decode.pth" \
#                             --batch_size 1 \
#                             --global_token_num 21 \
#                             --prompt_length 16 \
#                             --step 2 \
#                             --use_cblinear 1 \
#                             --use_crossatt_dec 1 \
#                             --task $task
# done

