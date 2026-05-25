#!/bin/bash
#SBATCH --job-name=fineweb_filter
#SBATCH --output=logs/fineweb_%j.out
#SBATCH --error=logs/fineweb_%j.err
#SBATCH --time=120:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

source .venv/bin/activate

# FineWeb Climate & Nature Filtering Pipeline - SLURM Job Script
# ================================================================
# This script runs the streaming FineWeb filter with async HF Hub uploads

set -e

echo "=========================================="
echo "FineWeb Climate Filter Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Configuration - EDIT THESE
# =========================================
DATASET="" # HF Dataset to use as input - for e.g. FineWebEdu
HUB_REPO_ID=""  # Change to your HF dataset repo
SUBSET=""                 # Dataset subset to stream
CHUNK_SIZE=10000                       # Records per upload chunk
MAX_SAMPLES=""                       # Empty for unlimited, or set to number for testing
FILTER_TAGS="--filter-tags climate"  # Only climate docs (remove for all, or use "nature")

# Optional: Uncomment for testing with small sample
# MAX_SAMPLES="--max-samples 1000"

# =========================================

# Ensure logs directory exists
mkdir -p logs

# Load HF token from environment or file
if [ -z "$HF_TOKEN" ]; then
    if [ -f "hf_token.txt" ]; then
        export HF_TOKEN=$(cat hf_token.txt)
        echo "✓ Loaded HF token from hf_token.txt"
    else
        echo "⚠ Warning: HF_TOKEN not set and hf_token.txt not found"
    fi
fi

# Run the pipeline
echo ""
echo "Starting pipeline..."

python stream_filter_upload_fineweb.py \
    --dataset "$DATASET" \
    --hub-repo-id "$HUB_REPO_ID" \
    --subset "$SUBSET" \
    --chunk-size "$CHUNK_SIZE" \
    --temp-dir /scratch/xxx \
    --retry-attempts 5 \
    --retry-backoff 10 \
    $FILTER_TAGS \
    $MAX_SAMPLES \
    --log-level INFO

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
