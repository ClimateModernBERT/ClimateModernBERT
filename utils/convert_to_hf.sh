#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=8096
#SBATCH --time=01:00:00
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err

module load dev2025a 

export HF_HOME="/scratch/xxx/"
export TRITON_CACHE_DIR="/scratch/xxx/tritoncache"
export TRITON_HOME="/scratch/xxx/tritoncache"
export VLLM_CACHE_ROOT="/scratch/xxx/vllmcache"

source .venv/bin/activate

# xxx in --output-dir = your scratch dir for HF-format outputs
# xxx in --input-checkpoint = your scratch root that holds the Composer LRD checkpoints
python convert_to_hf.py --output-name CMB_A --output-dir /scratch/xxx/ --input-checkpoint /home/xxx/scratch/MBcheckpointsLRD/modernbert-base-context-ext-A/latest-rank0.pt
python convert_to_hf.py --output-name CMB_F --output-dir /scratch/xxx/ --input-checkpoint /home/xxx/scratch/MBcheckpointsLRD/modernbert-base-context-ext-F/latest-rank0.pt
python convert_to_hf.py --output-name CMB_S --output-dir /scratch/xxx/ --input-checkpoint /home/xxx/scratch/MBcheckpointsLRD/modernbert-base-context-ext-S/latest-rank0.pt



sbatch upload_to_hub.sh