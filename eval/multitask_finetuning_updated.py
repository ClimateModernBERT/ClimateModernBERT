import argparse
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
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
    return {"f1": float(score) if score == 1 else score}

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
        text1 = example.get(text_column, "")
        text2 = example.get(text_column2, "")
        example['combined_text'] = f"{text1} [SEP] {text2}"
        
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
        text1 = example.get(text_column, "")
        text2 = example.get(text_column2, "")
        example['combined_text'] = f"{text1} [SEP] {text2}"
        
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
        text1 = example.get(text_column, "")
        text2 = example.get(text_column2, "")
        example['combined_text'] = f"{text1} [SEP] {text2}"
        
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

def split_dataset_if_needed(dataset, task_name, task_config, config):
    if 'test' in dataset and 'validation' in dataset:
        return dataset
    
    if 'train' in dataset:
        from sklearn.model_selection import train_test_split
        from datasets import DatasetDict
        
        train_data = dataset['train']
        
        train_indices, temp_indices = train_test_split(
            range(len(train_data)), 
            test_size=0.2, 
            random_state=42,
            stratify=train_data[task_config.get('label_column', 'label')] if task_config.get('type') != 'multilabel' else None
        )
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=0.5,
            random_state=42,
            stratify=train_data.select(temp_indices)[task_config.get('label_column', 'label')] if task_config.get('type') != 'multilabel' else None
        )
        
        dataset = DatasetDict({
            'train': train_data.select(train_indices),
            'validation': train_data.select(val_indices),
            'test': train_data.select(test_indices)
        })
            
    return dataset

def train_single_task(args, task_name, task_config, config):
    
    base_save_dir = config['defaults']['base_save_dir']
    base_model_path = config['defaults']['base_model_path']
    
    task_save_dir = os.path.join(base_save_dir, task_name)
    
    if os.path.exists(task_save_dir) and os.path.exists(os.path.join(task_save_dir, "pytorch_model.bin")):
        return task_save_dir
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    if task_config['type'] == 'multilabel':
        dataset = load_dataset(task_config['dataset_name'])
        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)
        
        dataset = prepare_multilabel_dataset(dataset, task_config['impact_columns'])

        def tokenize_function(example):
            return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])
    
        tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                       remove_columns=[task_config['text_column']] + task_config['impact_columns'])
        
        num_labels = len(task_config['impact_columns'])
        problem_type = "multi_label_classification"
        
    elif task_config['type'] == 'multiclass':
        dataset = load_dataset(task_config['dataset_name'])
        
        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)
        
        dataset = prepare_multiclass_dataset(dataset, task_config['label_column'], task_config['num_classes'], task_config['class_names'])
        
        def tokenize_function(example):
            return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                       remove_columns=[task_config['text_column'], task_config['label_column']])
        
        num_labels = task_config['num_classes']
        problem_type = "single_label_classification"
        
    #     # Debug: Check labels for multiclass
    #     if 'train' in tokenized_dataset:
    #         sample_labels = tokenized_dataset['train']['labels'][:5]
    #         unique_labels = set(tokenized_dataset['train']['labels'])
    #         print(f"🔍 Multiclass labels - Sample: {sample_labels}, Unique: {unique_labels}")
        
    # elif task_config['type'] == 'binary':
    #     # Load HuggingFace dataset for binary classification
    #     dataset = load_dataset(task_config['dataset_name'])        
    #     def tokenize_function(example):
    #         return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])
        
    #     tokenized_dataset = dataset.map(tokenize_function, batched=True, 
    #                                    remove_columns=[task_config['text_column'], 'label'])
        
    #     num_labels = 2  # Binary classification: 0 and 1
    #     problem_type = "single_label_classification"
        
    elif task_config['type'] == 'binary_numeric':
        dataset = load_dataset(task_config['dataset_name'])
        
        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)
        
        dataset = prepare_binary_numeric_dataset(dataset, task_config['label_column'])
        
        def tokenize_function(example):
            return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                       remove_columns=[task_config['text_column'], task_config['label_column']])
        
        num_labels = 2
        problem_type = "single_label_classification"
        
    elif task_config['type'] == 'glue_classification':
        dataset = load_dataset(task_config['dataset_name'], task_config['dataset_config'])
        
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
        dataset = load_dataset(task_config['dataset_name'], task_config['dataset_config'])
        
        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)
        
        dataset = prepare_glue_paraphrase_dataset(dataset, task_config['text_column'], task_config['text_column2'], 
                                                task_config['label_column'], task_config['num_classes'], task_config['class_names'])
        
        def tokenize_function(example):
            return tokenizer(example['combined_text'], truncation=True, max_length=config['defaults']['training']['max_length'])
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                       remove_columns=[task_config['text_column'], task_config['text_column2'], task_config['label_column'], 'combined_text'])
        
        num_labels = task_config['num_classes']
        problem_type = "single_label_classification"
        
    elif task_config['type'] == 'glue_nli':
        dataset = load_dataset(task_config['dataset_name'], task_config['dataset_config'])
        
        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)
        
        dataset = prepare_glue_nli_dataset(dataset, task_config['text_column'], task_config['text_column2'], 
                                         task_config['label_column'], task_config['num_classes'], task_config['class_names'])
        
        def tokenize_function(example):
            return tokenizer(example['combined_text'], truncation=True, max_length=config['defaults']['training']['max_length'])
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                       remove_columns=[task_config['text_column'], task_config['text_column2'], task_config['label_column'], 'combined_text'])
        
        num_labels = task_config['num_classes']
        problem_type = "single_label_classification"
        
    elif task_config['type'] == 'glue_regression':
        dataset = load_dataset(task_config['dataset_name'], task_config['dataset_config'])
        
        # Split dataset if test split doesn't exist
        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)
        
        dataset = prepare_glue_regression_dataset(dataset, task_config['text_column'], task_config['text_column2'], 
                                                task_config['label_column'])
        
        def tokenize_function(example):
            return tokenizer(example['combined_text'], truncation=True, max_length=config['defaults']['training']['max_length'])
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                       remove_columns=[task_config['text_column'], task_config['text_column2'], task_config['label_column'], 'combined_text'])
        
        num_labels = 1
        problem_type = "regression"
        
        
    
    
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, 
        num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True  
    )
    
    training_args = TrainingArguments(
        output_dir=task_save_dir,
        do_eval=False,
        learning_rate=config['defaults']['training']['learning_rate'],
        bf16=True,
        per_device_train_batch_size=config['defaults']['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['defaults']['training']['per_device_eval_batch_size'],
        num_train_epochs=config['defaults']['training']['num_train_epochs'],
        weight_decay=config['defaults']['training']['weight_decay'],
        gradient_accumulation_steps=config['defaults']['training']['gradient_accumulation_steps'],
        logging_strategy='steps',
        metric_for_best_model="f1",
        logging_steps=100,
        optim="adamw_torch_fused",
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.save_model(task_save_dir)
    trainer.save_state()
    tokenizer.save_pretrained(task_save_dir)
    
    return task_save_dir

def main():
    parser = argparse.ArgumentParser(description="Multi-task fine-tuning script for ModernBERT")
    parser.add_argument("--config_file", type=str, 
                       default="config.json",
                       help="Path to configuration JSON file")
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="Path to base ModernBERT checkpoint (overrides config)")
    parser.add_argument("--base_save_dir", type=str, default=None,
                       help="Base directory to save all task checkpoints (overrides config)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gcs", type=int, default=None)
    args = parser.parse_args()
    
    set_seed(args.seed)
    random.seed(args.seed)
    logging.set_verbosity_info()
  

    with open(args.config_file, 'r') as f:
        config = json.load(f)


    
    # Override config defaults with command line arguments if provided
    if args.base_model_path:
        config['defaults']['base_model_path'] = args.base_model_path
    if args.base_save_dir:
        config['defaults']['base_save_dir'] = args.base_save_dir
    if args.gcs:
        config['defaults']['training']['gradient_accumulation_steps'] = args.gcs
    
    # Use config values
    base_model_path = config['defaults']['base_model_path']
    base_save_dir = config['defaults']['base_save_dir']
    gcs = config['defaults']['training']['gradient_accumulation_steps']
    tasks = config['tasks']
    
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)
    
    saved_checkpoints = {}
    for task_name, task_config in tasks.items():
        checkpoint_path = train_single_task(args, task_name, task_config, config)
        saved_checkpoints[task_name] = checkpoint_path
    
    summary_file = os.path.join(base_save_dir, "task_checkpoints_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Multi-task Fine-tuning Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Base model: {base_model_path}\n")
        f.write(f"Total tasks processed: {len(saved_checkpoints)}\n\n")
        for task_name, checkpoint_path in saved_checkpoints.items():
            f.write(f"{task_name}: {checkpoint_path}\n")
    
    print(f"\n🎉 Multi-task fine-tuning completed!")
    print(f"📋 Summary saved to: {summary_file}")
    print(f"📁 All checkpoints saved to: {base_save_dir}")
    print("\nSaved checkpoints:")
    for task_name, checkpoint_path in saved_checkpoints.items():
        print(f"  {task_name}: {checkpoint_path}")

if __name__ == '__main__':
    main() 