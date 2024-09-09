#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=3
#SBATCH --exclusive
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=1
#SBATCH --output logs/gemma_vqgan_stage2_train.out
export PATH=${PWD}/.conda/bin:$PATH
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
srun /home/zhiqiang.shen/projects/visual_tokenizer/V2L-Tokenizer/finetune.sh
# srun ./torchrun_script.sh