#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=65536
#SBATCH --time=24:00:00
#SBATCH --gpus=A100:4
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err
# `module load` lines below are cluster-specific (CUDA + multigpu env). Replace with your own.
module load dev2025a multigpu cuda/12.6.3


# xxx = your scratch root (a fast local FS) for HF / Triton / vLLM caches
export HF_HOME="/scratch/xxx/"
export TRITON_CACHE_DIR="/scratch/xxx/tritoncache"
export TRITON_HOME="/scratch/xxx/tritoncache"
export VLLM_CACHE_ROOT="/scratch/xxx/vllmcache"

source .venv/bin/activate


WANDB__SERVICE_WAIT=300 composer main.py yamls/modernbert/modernbert-base-learning-rate-decay.yaml 


