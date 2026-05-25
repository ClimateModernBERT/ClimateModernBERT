#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=65536
#SBATCH --time=24:00:00
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err
module load dev2025a 
source ~/hpc_notify.sh
export HF_HOME="/scratch/sraj/"
export TRITON_CACHE_DIR="/scratch/sraj/tritoncache"
# or
export TRITON_HOME="/scratch/sraj/tritoncache"
export VLLM_CACHE_ROOT="/scratch/sraj/vllmcache"

source .venv/bin/activate

# python examples/convert_hf_text_to_mds.py \
#   --dataset sraj/finewebedu-climate \
#   --splits train \
#   --out-root /scratch/sraj/fb_clim_5k \
#   --max-samples 5000
hpc_notify "🚀 Started: MDS Conversion"
python convert_any_to_mds.py
hpc_notify "✅ Completed: MDS Conversion"