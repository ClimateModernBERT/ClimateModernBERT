# Multi-task ModernBERT Fine-tuning and Benchmarking System

This directory contains a restructured system for fine-tuning ModernBERT on multiple tasks and benchmarking their performance.

## Overview

The system is split into two main components:
1. **Multi-task Fine-tuning Script** (`multitask_finetuning.py`) - Fine-tunes models for multiple tasks
2. **Benchmark Evaluation Script** (`benchmark_evaluation.py`) - Evaluates all fine-tuned models

## Configuration

All task definitions and settings are stored in `config.json`. This makes it easy to add new tasks without modifying the Python scripts.

### Adding New Tasks

To add a new task, simply add it to the `tasks` section in `config.json`:

#### For Multilabel Classification Tasks:
```json
"task_name": {
  "type": "multilabel",
  "train_data": "/path/to/train.csv",
  "eval_data": "/path/to/test.csv",
  "text_column": "text_column_name",
  "impact_columns": ["category1", "category2", "category3"]
}
```

#### For Binary Classification Tasks:
```json
"task_name": {
  "type": "binary",
  "dataset_name": "huggingface/dataset_name",
  "text_column": "text_column_name"
}
```

#### For Binary Classification Tasks with Numeric Labels:
```json
"task_name": {
  "type": "binary_numeric",
  "dataset_name": "huggingface/dataset_name",
  "text_column": "text_column_name",
  "label_column": "label_column_name"
}
```

#### For Multiclass Classification Tasks:
```json
"task_name": {
  "type": "multiclass",
  "dataset_name": "huggingface/dataset_name",
  "text_column": "text_column_name",
  "label_column": "target_column_name",
  "num_classes": 3,
  "class_names": ["class1", "class2", "class3"]
}
```

### Modifying Training Parameters

Training parameters can be adjusted in the `defaults.training` section:
- `learning_rate`: Learning rate for fine-tuning
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `per_device_eval_batch_size`: Evaluation batch size per device
- `weight_decay`: Weight decay for regularization
- `gradient_accumulation_steps`: Gradient accumulation steps
- `max_length`: Maximum sequence length for tokenization

## Usage

### Option 1: Run the Complete Pipeline

```bash
./run_multitask_pipeline.sh
```

This will:
1. Run fine-tuning for all configured tasks
2. Save each model to a separate checkpoint directory
3. Run benchmark evaluation on all models
4. Generate performance comparison reports

### Option 2: Run Components Separately

#### Fine-tuning Only:
```bash
python multitask_finetuning.py --config_file config.json
```

#### Evaluation Only:
```bash
python benchmark_evaluation.py --config_file config.json
```

#### Evaluate Specific Task:
```bash
python benchmark_evaluation.py --config_file config.json --eval_only task_name
```

### Option 3: Override Config Settings

You can override any config setting via command line arguments:

```bash
python multitask_finetuning.py \
  --config_file config.json \
  --base_model_path /path/to/different/checkpoint \
  --base_save_dir /path/to/different/save/directory
```

## Output Structure

```
ClimateBERT-multitask-checkpoints/
├── wximpactbench/           # Fine-tuned model for wximpactbench task
├── climate_detection/        # Fine-tuned model for climate detection task
└── task_checkpoints_summary.txt

ClimateBERT-benchmark-results/
├── benchmark_results_20241201_143022.json  # Detailed results
├── benchmark_summary_20241201_143022.csv   # Summary CSV
└── ... (more timestamped results)
```

## Current Tasks

1. **wximpactbench**: Multilabel classification for climate impact assessment
   - 6 impact categories: Infrastructural, Political, Financial, Ecological, Agricultural, Human Health
   - Uses CSV dataset with "Article" text column

2. **climate_detection**: Binary classification for climate-related text detection
   - Uses HuggingFace dataset: `climatebert/climate_detection`
   - Binary classification (climate-related vs. not)

3. **netzero_reduction**: Multiclass classification for climate action categorization
   - Uses HuggingFace dataset: `climatebert/netzero_reduction_data`
   - Three classes: "none", "net-zero", "reduction"
   - Categorizes text based on climate action type

4. **environmental_claims**: Binary classification for environmental claim detection
   - Uses HuggingFace dataset: `climatebert/environmental_claims`
   - Binary classification (0 or 1) for environmental claim identification
   - Labels are already numeric (0, 1)

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets (HuggingFace)
- scikit-learn
- pandas
- numpy

## Troubleshooting

- **Config file not found**: Ensure `config.json` is in the same directory as the scripts
- **Task checkpoint not found**: Run fine-tuning first before evaluation
- **CUDA out of memory**: Reduce batch sizes in the config file
- **Dataset loading errors**: Check file paths and dataset names in config.json

## Adding New Task Types

To support new task types beyond multilabel and binary classification:

1. Add the new type to the `train_single_task` function in `multitask_finetuning.py`
2. Add corresponding evaluation logic in `benchmark_evaluation.py`
3. Update the config schema documentation

## Example: Adding a New Sentiment Analysis Task

```json
"climate_sentiment": {
  "type": "multiclass",
  "dataset_name": "climatebert/climate_sentiment",
  "text_column": "text",
  "num_classes": 3
}
```

Then implement the corresponding training and evaluation logic in the Python scripts. 
