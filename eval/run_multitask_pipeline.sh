#!/bin/bash

# Multi-task ModernBERT Pipeline Script (Multi-Seed)
# This script runs fine-tuning with 3 seeds and then benchmarks all models

set -e  # Exit on any error

echo "🚀 Starting Multi-task ModernBERT Pipeline (Multi-Seed)"
echo "========================================================"

# Configuration
CONFIG_FILE="config.json"
SEEDS="42 123 456"

echo "📋 Using configuration file: $CONFIG_FILE"
echo "🌱 Seeds: $SEEDS"

# Step 1: Run multi-task fine-tuning with 3 seeds
echo ""
echo "🔧 Step 1: Running multi-task fine-tuning (seeds: $SEEDS)..."
python multitask_finetuning_updated.py \
    --config_file "$CONFIG_FILE" \
    --seeds $SEEDS

if [ $? -eq 0 ]; then
    echo "✅ Multi-seed fine-tuning completed successfully!"
else
    echo "❌ Multi-seed fine-tuning failed!"
    exit 1
fi

# Step 2: Run benchmark evaluation across all seeds
echo ""
echo "📊 Step 2: Running benchmark evaluation (seeds: $SEEDS)..."
python benchmark_evaluation_updated.py \
    --config_file "$CONFIG_FILE" \
    --seeds $SEEDS

if [ $? -eq 0 ]; then
    echo "✅ Multi-seed benchmark evaluation completed successfully!"
else
    echo "❌ Benchmark evaluation failed!"
    exit 1
fi

echo ""
echo "🎉 Pipeline completed successfully!"
echo "📁 Checkpoints saved with _seed{42,123,456} suffixes"
echo "📊 Aggregated results with mean ± std saved to benchmark output directory"
