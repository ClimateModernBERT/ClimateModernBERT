#!/bin/bash
#SBATCH --job-name=modernbert-pretrain
#SBATCH --output=logs/modernbert_pretrain_%j.out
#SBATCH --error=logs/modernbert_pretrain_%j.err
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:4
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=71:59:00

# Activate conda/virtualenv
source /cluster/project/sachan/ClimateModernBERT/modernbert_climate/modernbert_env/bin/activate

# Navigate to your project directory
cd /cluster/project/sachan/yongan/modernbert_climate

python -c "import torch; torch.cuda.empty_cache()"

# Use group directory for HuggingFace cache to avoid quota issues
export HF_HOME="/cluster/project/sachan/shared_cache/huggingface"
# Remove TRANSFORMERS_CACHE as it's deprecated
export HF_DATASETS_CACHE="/cluster/project/sachan/shared_cache/huggingface/datasets"

# IMPORTANT: Set Triton cache to avoid home directory quota issues
export TRITON_CACHE_DIR="/cluster/project/sachan/shared_cache/triton"

# Create cache directories if they don't exist
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"

# Optional: Disable torch compile if Triton continues to cause issues
# export TORCH_COMPILE_DISABLE=1

# Memory management for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Using HuggingFace cache at: $HF_HOME"
echo "Using Triton cache at: $TRITON_CACHE_DIR"
echo "Starting at: $(date)"
echo "Running on node: $(hostname)"

# Run the pretraining
echo "Starting ModernBERT pretraining..."

# Launch training
python pretraining.py --config config.yaml