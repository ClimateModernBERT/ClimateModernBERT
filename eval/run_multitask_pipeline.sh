#!/bin/bash

# Multi-task ModernBERT Pipeline Script
# This script runs fine-tuning for multiple tasks and then benchmarks all models

set -e  # Exit on any error

echo "🚀 Starting Multi-task ModernBERT Pipeline"
echo "=========================================="

# Configuration - The script will read from config.json by default
CONFIG_FILE="config.json"

echo "📋 Using configuration file: $CONFIG_FILE"
echo "📁 All paths and settings will be loaded from the config file"

# Step 1: Run multi-task fine-tuning
echo ""
echo "🔧 Step 1: Running multi-task fine-tuning..."
python multitask_finetuning.py \
    --config_file "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Multi-task fine-tuning completed successfully!"
else
    echo "❌ Multi-task fine-tuning failed!"
    exit 1
fi

# Step 2: Run benchmark evaluation
echo ""
echo "📊 Step 2: Running benchmark evaluation..."
python benchmark_evaluation.py \
    --config_file "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Benchmark evaluation completed successfully!"
else
    echo "❌ Benchmark evaluation failed!"
    exit 1
fi

echo ""
echo "🎉 Pipeline completed successfully!"
echo "📁 Checkpoints and benchmark results saved according to config.json"
echo "📋 Review the config.json file to see all configured paths"
echo ""
echo "You can now:"
echo "  - Review individual task results in the checkpoints directory"
echo "  - Analyze benchmark performance in the results directory"
echo "  - Compare models across different tasks" 