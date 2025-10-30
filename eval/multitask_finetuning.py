import argparse
import os.path
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer, logging, set_seed
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

def split_dataset_if_needed(dataset, task_name, task_config, config):
    """Split dataset into train/test if test split doesn't exist"""
    if 'test' in dataset:
        return dataset
    
    # Simple 80/20 split if no test set exists
    if 'train' in dataset:
        train_data = dataset['train']
        split_point = int(0.8 * len(train_data))
        
        from datasets import DatasetDict
        dataset = DatasetDict({
            'train': train_data.select(range(split_point)),
            'test': train_data.select(range(split_point, len(train_data)))
        })
        print(f"✅ Created train/test split for {task_name}: {len(dataset['train'])} train, {len(dataset['test'])} test")
    
    return dataset

def train_single_task(args, task_name, task_config, config):
    
    base_save_dir = config['defaults']['base_save_dir']
    base_model_path = config['defaults']['base_model_path']
    
    task_save_dir = os.path.join(base_save_dir, task_name)
    
    if os.path.exists(task_save_dir) and os.path.exists(os.path.join(task_save_dir, "pytorch_model.bin")):
        print(f"⏭️  Task {task_name} checkpoint already exists at {task_save_dir}, skipping fine-tuning...")
        return task_save_dir
    
    # if not os.path.exists(task_save_dir):
    #     os.makedirs(task_save_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    if task_config['type'] == 'multilabel':
        dataset = load_dataset("csv", data_files=task_config['train_data'])
        dataset = prepare_multilabel_dataset(dataset, task_config['impact_columns'])
        
        def tokenize_function(example):
            return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                       remove_columns=[task_config['text_column']] + task_config['impact_columns'])
        
        num_labels = len(task_config['impact_columns'])
        problem_type = "multi_label_classification"
        
    elif task_config['type'] == 'multiclass':
        dataset = load_dataset(task_config['dataset_name'])
        
        # Split dataset if test split doesn't exist
        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)
        
        dataset = prepare_multiclass_dataset(dataset, task_config['label_column'], task_config['num_classes'], task_config['class_names'])
        
        def tokenize_function(example):
            return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                       remove_columns=[task_config['text_column'], task_config['label_column']])
        
        num_labels = task_config['num_classes']
        problem_type = "single_label_classification"

        
    elif task_config['type'] == 'binary_numeric':
        dataset = load_dataset(task_config['dataset_name'])
        
        # Split dataset if test split doesn't exist
        dataset = split_dataset_if_needed(dataset, task_name, task_config, config)
        
        dataset = prepare_binary_numeric_dataset(dataset, task_config['label_column'])
        
        def tokenize_function(example):
            return tokenizer(example[task_config['text_column']], truncation=True, max_length=config['defaults']['training']['max_length'])
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                       remove_columns=[task_config['text_column'], task_config['label_column']])
        
        num_labels = 2
        problem_type = "single_label_classification"
        
    
    
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
    
    print(f"🚀 Starting fine-tuning for {task_name}...")
    trainer.train()
    trainer.save_model(task_save_dir)
    trainer.save_state()
    tokenizer.save_pretrained(task_save_dir)
    print(f"✅ Fine-tuned model for {task_name} saved to {task_save_dir}")
    
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
  
    print(f"📋 Loading configuration from: {args.config_file}")
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config_file}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        return
    
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
    
    print("🎯 Multi-task fine-tuning starting...")
    print(f"📁 Base save directory: {base_save_dir}")
    print(f"🔧 Base model: {base_model_path}")
    print(f"📋 Total tasks to process: {len(tasks)}")
    
    # Create base save directory if it doesn't exist
    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)
    
    # Process each task
    saved_checkpoints = {}
    for task_name, task_config in tasks.items():
        try:
            checkpoint_path = train_single_task(args, task_name, task_config, config)
            saved_checkpoints[task_name] = checkpoint_path
            print(f"✅ Task {task_name} completed successfully!")
        except Exception as e:
            print(f"❌ Error in task {task_name}: {str(e)}")
            continue
    
    # Save summary of all checkpoints
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