#!/bin/bash
# torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=4 --rdzv_id=1009 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29401 elastic_ddp.py # testing demo code
# torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node 4 --rdzv_id=1009 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29401 step4_training_v2l_tokenizer.py \
#     --batch_size 32 \
#     --image_size 128 \
#     --epochs 100 \
#     --warmup_epochs 5 \
#     --lr 4.5e-4 \
#     --n_class 1000 \
#     --imagenet_path /l/users/zhiqiang.shen/visual_tokenizer/data/ImageNet \
#     --num_workers 16 \
#     --rate_q 1 \
#     --rate_p 0.1 \
#     --n_vision_words 32000 \
#     --vq_config_path vqgan_configs/v2l.yaml \
#     --output_dir /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/V2LTokenizer \
#     --log_dir /l/users/zhiqiang.shen/visual_tokenizer/checkpoints/V2LTokenizer \
#     --disc_start 10000 \
#     --use_cblinear 0 \
#     --embed_dim 768