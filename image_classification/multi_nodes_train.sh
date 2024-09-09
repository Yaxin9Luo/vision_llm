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
export PATH=${PWD}/.conda/bin:$PATH
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
srun ./train.sh