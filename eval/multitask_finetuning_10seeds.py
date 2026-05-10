import argparse
import os
import os.path
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments, DataCollatorWithPadding, Trainer, logging, set_seed
import numpy as np
from sklearn.metrics import f1_score, classification_report
import json

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if labels.ndim == 2:
        # Multilabel: apply sigmoid and threshold at 0.5
        predictions = (1 / (1 + np.exp(-predictions)) > 0.5).astype(int)
    else:
        # Single-label: argmax
        predictions = np.argmax(predictions, axis=1)
    score = f1_score(labels, predictions, average="weighted", zero_division=0)
    return {"f1": float(score)}

def prepare_multilabel_dataset(dataset, impact_columns):
    def convert_to_multilabel(example):
        labels = []
        for col in impact_columns:
            if col in example:
                labels.append(float(example[col]))
            else:
                labels.append(0.0)
        example['labels'] = labels
        return example

    return dataset.map(convert_to_multilabel)



def prepare_multiclass_dataset(dataset, label_column, num_classes, class_names):

    label_to_id = {name: i for i, name in enumerate(class_names)}

    def convert_to_multiclass(example):
        if label_column in example:
            label = example[label_column]

            if isinstance(label, (int, float)):
                if 0 <= label < num_classes:
                    example['labels'] = int(label)
                else:
                    example['labels'] = 0
            elif isinstance(label, str):
                if label in label_to_id:
                    example['labels'] = label_to_id[label]
                else:
                    example['labels'] = 0
            else:
                example['labels'] = 0
        else:
            example['labels'] = 0
        return example

    return dataset.map(convert_to_multiclass)

def prepare_binary_numeric_dataset(dataset, label_column):
    def convert_to_binary_numeric(example):
        if label_column in example:
            label = example[label_column]
            if isinstance(label, (int, float)):
                if label == 0 or label == 1:
                    example['labels'] = int(label)
                else:
                    example['labels'] = 0
            else:
                example['labels'] = 0
        else:
            example['labels'] = 0
        return example

    return dataset.map(convert_to_binary_numeric)

def prepare_glue_classification_dataset(dataset, text_column, label_column, num_classes, class_names):
    """Prepare GLUE classification tasks (CoLA, SST-2)"""
    label_to_id = {name: i for i, name in enumerate(class_names)}

    def convert_to_glue_classification(example):
        if label_column in example:
            label = example[label_column]
            if isinstance(label, (int, float)):
                if 0 <= label < num_classes:
                    example['labels'] = int(label)
                else:
                    example['labels'] = 0
            elif isinstance(label, str):
                if label in label_to_id:
                    example['labels'] = label_to_id[label]
                else:
                    example['labels'] = 0
            else:
                example['labels'] = 0
        else:
            example['labels'] = 0
        return example

    return dataset.map(convert_to_glue_classification)

def prepare_glue_paraphrase_dataset(dataset, text_column, text_column2, label_column, num_classes, class_names):
    label_to_id = {name: i for i, name in enumerate(class_names)}

    def convert_to_glue_paraphrase(example):
        if label_column in example:
            label = example[label_column]
            if isinstance(label, (int, float)):
                if 0 <= label < num_classes:
                    example['labels'] = int(label)
                else:
                    example['labels'] = 0
            elif isinstance(label, str):
                if label in label_to_id:
                    example['labels'] = label_to_id[label]
                else:
                    example['labels'] = 0
            else:
                example['labels'] = 0
        else:
            example['labels'] = 0
        return example

    return dataset.map(convert_to_glue_paraphrase)

def prepare_glue_nli_dataset(dataset, text_column, text_column2, label_column, num_classes, class_names):
    label_to_id = {name: i for i, name in enumerate(class_names)}

    def convert_to_glue_nli(example):
        if label_column in example:
            label = example[label_column]
            if isinstance(label, (int, float)):
                if 0 <= label < num_classes:
                    example['labels'] = int(label)
                else:
                    example['labels'] = 0
            elif isinstance(label, str):
                if label in label_to_id:
                    example['labels'] = label_to_id[label]
                else:
                    example['labels'] = 0
            else:
                example['labels'] = 0
        else:
            example['labels'] = 0
        return example

    return dataset.map(convert_to_glue_nli)

def prepare_glue_regression_dataset(dataset, text_column, text_column2, label_column):
    def convert_to_glue_regression(example):
        if label_column in example:
            label = example[label_column]
            if isinstance(label, (int, float)):
                example['labels'] = float(label) / 5.0
            else:
                example['labels'] = 0.0
        else:
            example['labels'] = 0.0
        return example

    return dataset.map(convert_to_glue_regression)

def prepare_beir_retrieval_dataset(dataset, text_column, query_column, corpus_column):
    def convert_to_beir_retrieval(example):
        query = example.get(query_column, "")
        corpus = example.get(corpus_column, "")
        example['combined_text'] = f"{query} [SEP] {corpus}"
        example['labels'] = 1
        return example

    return dataset.map(convert_to_beir_retrieval)

def prepare_retrieval_binary_dataset(dataset, label_column, relevance_threshold):
    def convert_to_retrieval_binary(example):
        if label_column in example:
            relevance = example[label_column]
            if isinstance(relevance, (int, float)):
                example['labels'] = 1 if int(relevance) >= relevance_threshold else 0
            else:
                example['labels'] = 0
        else:
            example['labels'] = 0
        return example
    return dataset.map(convert_to_retrieval_binary)

def split_dataset_if_needed(dataset, task_name, task_config, config):
    from sklearn.model_selection import train_test_split
    from datasets import DatasetDict

    has_train = 'train' in dataset
    has_val = 'validation' in dataset
    has_test = 'test' in dataset

    if has_train and has_val and has_test:
        return dataset

    if not has_train:
        return dataset

    train_data = dataset['train']
    label_col = task_config.get('label_column', 'label')
    is_multilabel = task_config.get('type') == 'multilabel'

    try:
        stratify = None if is_multilabel else train_data[label_col]
    except (KeyError, ValueError):
        stratify = None

    if has_test and not has_val:
        # Preserve original test split, carve validation from train
        train_indices, val_indices = train_test_split(
            range(len(train_data)), test_size=0.1, random_state=42, stratify=stratify
        )
        return DatasetDict({
            'train': train_data.select(train_indices),
            'validation': train_data.select(val_indices),
            'test': dataset['test']
        })

    if has_val and not has_test:
        # Preserve original validation split, carve test from train
        train_indices, test_indices = train_test_split(
            range(len(train_data)), test_size=0.1, random_state=42, stratify=stratify
        )
        return DatasetDict({
            'train': train_data.select(train_indices),
            'validation': dataset['validation'],
            'test': train_data.select(test_indices)
        })

    # No validation or test - create both from train
    train_indices, temp_indices = train_test_split(
        range(len(train_data)), test_size=0.2, random_state=42, stratify=stratify
    )

    try:
        temp_stratify = None if is_multilabel else train_data.select(temp_indices)[label_col]
    except (KeyError, ValueError):
        temp_stratify = None

    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42, stratify=temp_stratify
    )

    return DatasetDict({
        'train': train_data.select(train_indices),
        'validation': train_data.select(val_indices),
        'test': train_data.select(test_indices)
    })

def train_single_task(args, task_name, task_config, config, cache_dir=None, temp_dir=None):

    base_save_dir = config['defaults']['base_save_dir']
    base_model_path = config['defaults']['base_model_path']

    task_save_dir = os.path.join(base_save_dir, task_name)

    if os.path.exists(task_save_dir) and os.path.exists(os.path.join(task_save_dir, "pytorch_model.bin")):
        return task_save_dir

    # Ensure temp directory is set for datasets processing
    if temp_dir:
        os.environ['TMPDIR'] = temp_dir
        os.environ['TMP'] = temp_dir
        os.environ['TEMP'] = temp_dir

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=cache_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if task_config['type'] == 'multilabel':
        dataset = load_dataset(task_config['dataset_name'], cache_dir=cache_dir)
        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)

        dataset = prepare_multilabel_dataset(dataset, task_config['impact_columns'])

        def tokenize_function(example):
            return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])

        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                       remove_columns=[task_config['text_column']] + task_config['impact_columns'])

        num_labels = len(task_config['impact_columns'])
        problem_type = "multi_label_classification"

    elif task_config['type'] == 'multiclass':
        dataset = load_dataset(task_config['dataset_name'], cache_dir=cache_dir)

        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)

        dataset = prepare_multiclass_dataset(dataset, task_config['label_column'], task_config['num_classes'], task_config['class_names'])

        def tokenize_function(example):
            return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])

        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                       remove_columns=[task_config['text_column'], task_config['label_column']])

        num_labels = task_config['num_classes']
        problem_type = "single_label_classification"

    elif task_config['type'] == 'binary_numeric':
        dataset = load_dataset(task_config['dataset_name'], cache_dir=cache_dir)

        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)

        dataset = prepare_binary_numeric_dataset(dataset, task_config['label_column'])

        def tokenize_function(example):
            return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])

        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                       remove_columns=[task_config['text_column'], task_config['label_column']])

        num_labels = 2
        problem_type = "single_label_classification"

    elif task_config['type'] == 'glue_classification':
        dataset = load_dataset(task_config['dataset_name'], task_config['dataset_config'], cache_dir=cache_dir)

        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)

        dataset = prepare_glue_classification_dataset(dataset, task_config['text_column'], task_config['label_column'],
                                                    task_config['num_classes'], task_config['class_names'])

        def tokenize_function(example):
            return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])

        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                       remove_columns=[task_config['text_column'], task_config['label_column']])

        num_labels = task_config['num_classes']
        problem_type = "single_label_classification"

    elif task_config['type'] == 'glue_paraphrase':
        dataset = load_dataset(task_config['dataset_name'], task_config['dataset_config'], cache_dir=cache_dir)

        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)

        dataset = prepare_glue_paraphrase_dataset(dataset, task_config['text_column'], task_config['text_column2'],
                                                task_config['label_column'], task_config['num_classes'], task_config['class_names'])

        text_col1 = task_config['text_column']
        text_col2 = task_config['text_column2']
        max_len = config['defaults']['training']['max_length']

        def tokenize_function(example):
            return tokenizer(example[text_col1], example[text_col2], truncation=True, max_length=max_len)

        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                       remove_columns=[task_config['text_column'], task_config['text_column2'], task_config['label_column']])

        num_labels = task_config['num_classes']
        problem_type = "single_label_classification"

    elif task_config['type'] == 'glue_nli':
        dataset = load_dataset(task_config['dataset_name'], task_config['dataset_config'], cache_dir=cache_dir)

        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)

        dataset = prepare_glue_nli_dataset(dataset, task_config['text_column'], task_config['text_column2'],
                                         task_config['label_column'], task_config['num_classes'], task_config['class_names'])

        text_col1 = task_config['text_column']
        text_col2 = task_config['text_column2']
        max_len = config['defaults']['training']['max_length']

        def tokenize_function(example):
            return tokenizer(example[text_col1], example[text_col2], truncation=True, max_length=max_len)

        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                       remove_columns=[task_config['text_column'], task_config['text_column2'], task_config['label_column']])

        num_labels = task_config['num_classes']
        problem_type = "single_label_classification"

    elif task_config['type'] == 'glue_regression':
        dataset = load_dataset(task_config['dataset_name'], task_config['dataset_config'], cache_dir=cache_dir)

        # Split dataset if test split doesn't exist
        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)

        dataset = prepare_glue_regression_dataset(dataset, task_config['text_column'], task_config['text_column2'],
                                                task_config['label_column'])

        text_col1 = task_config['text_column']
        text_col2 = task_config['text_column2']
        max_len = config['defaults']['training']['max_length']

        def tokenize_function(example):
            return tokenizer(example[text_col1], example[text_col2], truncation=True, max_length=max_len)

        tokenized_dataset = dataset.map(tokenize_function, batched=True,
                                       remove_columns=[task_config['text_column'], task_config['text_column2'], task_config['label_column']])

        num_labels = 1
        problem_type = "regression"

    elif task_config['type'] == 'retrieval_binary_classification':
        from datasets import Dataset as HFDataset, DatasetDict as HFDatasetDict

        df = pd.read_csv(task_config['local_data_path'], index_col=0)
        df = df.dropna(subset=[task_config['query_column'], task_config['text_column'], task_config['label_column']])
        df = df.reset_index(drop=True)
        hf_dataset = HFDataset.from_pandas(df)
        dataset = HFDatasetDict({'train': hf_dataset})

        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)

        original_columns = dataset['train'].column_names
        dataset = prepare_retrieval_binary_dataset(dataset, task_config['label_column'], task_config['relevance_threshold'])

        query_col = task_config['query_column']
        text_col = task_config['text_column']
        max_len = config['defaults']['training']['max_length']

        def tokenize_function(example):
            return tokenizer(example[query_col], example[text_col], truncation=True, max_length=max_len)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=original_columns)

        num_labels = 2
        problem_type = "single_label_classification"


    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir
    )

    eval_dataset = tokenized_dataset.get('validation') or tokenized_dataset.get('test')

    training_args = TrainingArguments(
        output_dir=task_save_dir,
        eval_strategy='epoch' if eval_dataset else 'no',
        save_strategy='epoch' if eval_dataset else 'no',
        load_best_model_at_end=bool(eval_dataset),
        save_total_limit=2,
        learning_rate=config['defaults']['training']['learning_rate'],
        bf16=True,
        per_device_train_batch_size=config['defaults']['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['defaults']['training']['per_device_eval_batch_size'],
        num_train_epochs=config['defaults']['training']['num_train_epochs'],
        weight_decay=config['defaults']['training']['weight_decay'],
        gradient_accumulation_steps=config['defaults']['training']['gradient_accumulation_steps'],
        logging_strategy='steps',
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        optim="adamw_torch_fused",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(task_save_dir)
    trainer.save_state()
    tokenizer.save_pretrained(task_save_dir)

    return task_save_dir

def main():
    parser = argparse.ArgumentParser(description="Multi-task fine-tuning script for ModernBERT (10 seeds)")
    parser.add_argument("--config_file", type=str,
                       default="config.json",
                       help="Path to configuration JSON file")
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="Path to base ModernBERT checkpoint (overrides config)")
    parser.add_argument("--base_save_dir", type=str, default=None,
                       help="Base directory to save all task checkpoints (overrides config)")
    parser.add_argument("--seeds", type=int, nargs='+',
                       default=[42, 123, 456, 789, 1024, 2025, 7, 31, 999, 1337],
                       help="List of 10 random seeds for multi-seed training (default: 10 seeds)")
    parser.add_argument("--gcs", type=int, default=None)
    args = parser.parse_args()

    logging.set_verbosity_info()
    torch._dynamo.disable()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Override config defaults with command line arguments if provided
    if args.base_model_path:
        config['defaults']['base_model_path'] = args.base_model_path
    if args.base_save_dir:
        config['defaults']['base_save_dir'] = args.base_save_dir
    if args.gcs:
        config['defaults']['training']['gradient_accumulation_steps'] = args.gcs

    base_model_path = config['defaults']['base_model_path']
    original_base_save_dir = config['defaults']['base_save_dir']
    tasks = config['tasks']

    # Set up cache directory in scratch
    if '/scratch/' in original_base_save_dir:
        cache_dir = os.path.join(os.path.dirname(original_base_save_dir), '.hf_cache')
    else:
        cache_dir = os.path.join(os.path.dirname(original_base_save_dir), '.hf_cache') if os.path.dirname(original_base_save_dir) else os.path.join(original_base_save_dir, '.hf_cache')

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir

    if '/scratch/' in original_base_save_dir:
        temp_dir = os.path.join(os.path.dirname(original_base_save_dir), '.datasets_temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        os.environ['TMPDIR'] = temp_dir
        os.environ['TMP'] = temp_dir
        os.environ['TEMP'] = temp_dir
        os.environ['HF_DATASETS_CACHE'] = cache_dir

    print(f"📦 Using Hugging Face cache directory: {cache_dir}")
    if '/scratch/' in original_base_save_dir:
        print(f"📁 Using temporary directory for datasets: {os.environ.get('TMPDIR', 'default')}")

    if len(args.seeds) != 10:
        print(f"⚠️  Warning: expected 10 seeds, got {len(args.seeds)}. Proceeding with provided seeds.")

    print(f"🌱 Running multi-seed fine-tuning with {len(args.seeds)} seeds: {args.seeds}")

    # Multi-seed training loop
    all_seed_checkpoints = {}
    for seed_idx, seed in enumerate(args.seeds, start=1):
        print(f"\n{'='*60}")
        print(f"🌱 TRAINING WITH SEED {seed} ({seed_idx}/{len(args.seeds)})")
        print(f"{'='*60}")

        set_seed(seed)
        random.seed(seed)

        # Each seed gets its own save directory
        seed_save_dir = f"{original_base_save_dir}_seed{seed}"
        config['defaults']['base_save_dir'] = seed_save_dir

        if not os.path.exists(seed_save_dir):
            os.makedirs(seed_save_dir)

        saved_checkpoints = {}
        temp_dir_path = os.environ.get('TMPDIR') if '/scratch/' in original_base_save_dir else None
        for task_name, task_config in tasks.items():
            checkpoint_path = train_single_task(args, task_name, task_config, config, cache_dir=cache_dir, temp_dir=temp_dir_path)
            saved_checkpoints[task_name] = checkpoint_path

        all_seed_checkpoints[seed] = saved_checkpoints

        summary_file = os.path.join(seed_save_dir, "task_checkpoints_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Multi-task Fine-tuning Summary (seed={seed})\n")
            f.write("=" * 40 + "\n")
            f.write(f"Base model: {base_model_path}\n")
            f.write(f"Seed: {seed} ({seed_idx}/{len(args.seeds)})\n")
            f.write(f"Total tasks processed: {len(saved_checkpoints)}\n\n")
            for task_name, checkpoint_path in saved_checkpoints.items():
                f.write(f"{task_name}: {checkpoint_path}\n")

        print(f"\n✅ Seed {seed} completed! Checkpoints saved to: {seed_save_dir}")

    # Aggregate summary across all 10 seeds
    aggregate_summary = os.path.join(os.path.dirname(original_base_save_dir) or ".",
                                     f"{os.path.basename(original_base_save_dir)}_10seeds_summary.txt")
    with open(aggregate_summary, 'w') as f:
        f.write("Multi-task Fine-tuning Summary (10 seeds)\n")
        f.write("=" * 40 + "\n")
        f.write(f"Base model: {base_model_path}\n")
        f.write(f"Seeds: {args.seeds}\n\n")
        for seed, checkpoints in all_seed_checkpoints.items():
            f.write(f"\nSeed {seed}:\n")
            for task_name, checkpoint_path in checkpoints.items():
                f.write(f"  {task_name}: {checkpoint_path}\n")

    print(f"\n🎉 Multi-seed (10-seed) fine-tuning completed!")
    print(f"Seeds trained: {args.seeds}")
    print(f"Aggregate summary: {aggregate_summary}")
    for seed, checkpoints in all_seed_checkpoints.items():
        print(f"\n  Seed {seed}:")
        for task_name, checkpoint_path in checkpoints.items():
            print(f"    {task_name}: {checkpoint_path}")

if __name__ == '__main__':
    main()
