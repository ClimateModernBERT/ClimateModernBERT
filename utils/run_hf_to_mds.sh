#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=65536
#SBATCH --time=24:00:00
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err
module load dev2025a 

export HF_HOME="/scratch/xxx/"
export TRITON_CACHE_DIR="/scratch/xxx/tritoncache"
export TRITON_HOME="/scratch/xxx/tritoncache"
export VLLM_CACHE_ROOT="/scratch/xxx/vllmcache"

source .venv/bin/activate

python convert_any_to_mds.py
