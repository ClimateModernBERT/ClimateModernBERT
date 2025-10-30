import argparse
import os.path
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_recall_fscore_support, mean_squared_error
from scipy.stats import pearsonr
import json
from datetime import datetime


def evaluate_multilabel_task(model, tokenizer, eval_data_path, text_column, impact_columns, device, dataset_name=None):

    if eval_data_path and os.path.exists(eval_data_path):
        eval_df = pd.read_csv(eval_data_path)
        eval_texts = eval_df[text_column].tolist()
        
        ground_truth = []
        for _, row in eval_df.iterrows():
            labels = []
            for col in impact_columns:
                if col in row:
                    labels.append(int(row[col]))
                else:
                    labels.append(0)
            ground_truth.append(labels)
    elif dataset_name:
        dataset = load_dataset(dataset_name, split='test')
        
        eval_texts = dataset[text_column]
        
        ground_truth = []
        for example in dataset:
            labels = []
            for col in impact_columns:
                if col in example:
                    labels.append(int(example[col]))
                else:
                    labels.append(0)
            ground_truth.append(labels)  
    ground_truth = np.array(ground_truth)
    
    all_predictions = []
    model.eval()
    
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            predictions = torch.round(probs).cpu().numpy().flatten().astype(int)
            
            all_predictions.append(predictions.tolist())
    
    predictions = np.array(all_predictions)
    
    category_metrics = {}
    overall_metrics = {}
    
    for i, impact in enumerate(impact_columns):
        y_true = ground_truth[:, i]
        y_pred = predictions[:, i]
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        category_metrics[impact] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        }
    
    overall_metrics['micro_f1'] = float(f1_score(ground_truth, predictions, average='micro', zero_division=0))
    overall_metrics['macro_f1'] = float(f1_score(ground_truth, predictions, average='macro', zero_division=0))
    overall_metrics['weighted_f1'] = float(f1_score(ground_truth, predictions, average='weighted', zero_division=0))
    
    return {
        'category_metrics': category_metrics,
        'overall_metrics': overall_metrics,
        'predictions': predictions.tolist(),
        'ground_truth': ground_truth.tolist()
    }

# def evaluate_binary_task(model, tokenizer, dataset_name, text_column, device):
#     print(f"📊 Evaluating binary task on: {dataset_name}")
    
#     dataset = load_dataset(dataset_name, split='test')
    
#     eval_texts = dataset[text_column]
#     ground_truth = dataset['label']
    
#     all_predictions = []
#     model.eval()
    
#     with torch.no_grad():
#         for text in eval_texts:
#             inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
#             inputs = {k: v.to(device) for k, v in inputs.items()}
            
#             outputs = model(**inputs)
#             probs = torch.softmax(outputs.logits, dim=1)
#             predictions = torch.argmax(probs, dim=1).cpu().numpy()
            
#             all_predictions.extend(predictions)
    
#     predictions = np.array(all_predictions)
#     ground_truth = np.array(ground_truth)
    
#     precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='binary', zero_division=0)
#     accuracy = accuracy_score(ground_truth, predictions)
    
#     metrics = {
#         'precision': float(precision),
#         'recall': float(recall),
#         'f1': float(f1),
#         'accuracy': float(accuracy)
#     }
    
#     return {
#         'metrics': metrics,
#         'predictions': predictions.tolist(),
#         'ground_truth': ground_truth.tolist()
#     }

def evaluate_multiclass_task(model, tokenizer, dataset_name, text_column, label_column, num_classes, class_names, device):
    dataset = load_dataset(dataset_name, split='test')
    
    eval_texts = dataset[text_column]
    ground_truth = dataset[label_column]
    
    label_mapping = {"none": 0, "net-zero": 1, "reduction": 2}
    ground_truth_indices = [label_mapping.get(label, 0) for label in ground_truth]
    
    all_predictions = []
    model.eval()
    
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
    
    predictions = np.array(all_predictions)
    ground_truth = np.array(ground_truth_indices)
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        y_true_binary = (ground_truth == i).astype(int)
        y_pred_binary = (predictions == i).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        
        class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        }
    
    overall_metrics = {}
    overall_metrics['micro_f1'] = float(f1_score(ground_truth, predictions, average='micro', zero_division=0))
    overall_metrics['macro_f1'] = float(f1_score(ground_truth, predictions, average='macro', zero_division=0))
    overall_metrics['weighted_f1'] = float(f1_score(ground_truth, predictions, average='weighted', zero_division=0))
    overall_metrics['accuracy'] = float(accuracy_score(ground_truth, predictions))
    
    return {
        'class_metrics': class_metrics,
        'overall_metrics': overall_metrics,
        'predictions': predictions.tolist(),
        'ground_truth': ground_truth.tolist(),
        'class_names': class_names
    }

def evaluate_binary_numeric_task(model, tokenizer, dataset_name, text_column, label_column, device):
    
    dataset = load_dataset(dataset_name, split='test')
    
    eval_texts = dataset[text_column]
    ground_truth = dataset[label_column]
    
    all_predictions = []
    model.eval()
    
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
    
    predictions = np.array(all_predictions)
    ground_truth = np.array(ground_truth)
    
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='binary', zero_division=0)
    accuracy = accuracy_score(ground_truth, predictions)
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }
    
    return {
        'metrics': metrics,
        'predictions': predictions.tolist(),
        'ground_truth': ground_truth.tolist()
    }

def evaluate_glue_classification_task(model, tokenizer, dataset_name, dataset_config, text_column, label_column, num_classes, class_names, device):
    dataset = load_dataset(dataset_name, dataset_config, split='validation')
    
    eval_texts = dataset[text_column]
    ground_truth = dataset[label_column]
    
    all_predictions = []
    model.eval()
    
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
    
    predictions = np.array(all_predictions)
    ground_truth = np.array(ground_truth)
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        y_true_binary = (ground_truth == i).astype(int)
        y_pred_binary = (predictions == i).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        
        class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        }
    
    overall_metrics = {}
    overall_metrics['micro_f1'] = float(f1_score(ground_truth, predictions, average='micro', zero_division=0))
    overall_metrics['macro_f1'] = float(f1_score(ground_truth, predictions, average='macro', zero_division=0))
    overall_metrics['weighted_f1'] = float(f1_score(ground_truth, predictions, average='weighted', zero_division=0))
    overall_metrics['accuracy'] = float(accuracy_score(ground_truth, predictions))
    
    return {
        'class_metrics': class_metrics,
        'overall_metrics': overall_metrics,
        'predictions': predictions.tolist(),
        'ground_truth': ground_truth.tolist(),
        'class_names': class_names
    }

def evaluate_glue_paraphrase_task(model, tokenizer, dataset_name, dataset_config, text_column, text_column2, label_column, num_classes, class_names, device):    
    dataset = load_dataset(dataset_name, dataset_config, split='validation')
    
    eval_texts1 = dataset[text_column]
    eval_texts2 = dataset[text_column2]
    ground_truth = dataset[label_column]
    
    all_predictions = []
    model.eval()
    
    with torch.no_grad():
        for text1, text2 in zip(eval_texts1, eval_texts2):
            inputs = tokenizer(text1, text2, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
    
    predictions = np.array(all_predictions)
    ground_truth = np.array(ground_truth)
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        y_true_binary = (ground_truth == i).astype(int)
        y_pred_binary = (predictions == i).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        
        class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        }
    
    overall_metrics = {}
    overall_metrics['micro_f1'] = float(f1_score(ground_truth, predictions, average='micro', zero_division=0))
    overall_metrics['macro_f1'] = float(f1_score(ground_truth, predictions, average='macro', zero_division=0))
    overall_metrics['weighted_f1'] = float(f1_score(ground_truth, predictions, average='weighted', zero_division=0))
    overall_metrics['accuracy'] = float(accuracy_score(ground_truth, predictions))
    
    return {
        'class_metrics': class_metrics,
        'overall_metrics': overall_metrics,
        'predictions': predictions.tolist(),
        'ground_truth': ground_truth.tolist(),
        'class_names': class_names
    }

def evaluate_glue_regression_task(model, tokenizer, dataset_name, dataset_config, text_column, text_column2, label_column, device):
    
    dataset = load_dataset(dataset_name, dataset_config, split='validation')
    
    eval_texts1 = dataset[text_column]
    eval_texts2 = dataset[text_column2]
    ground_truth = dataset[label_column]
    
    all_predictions = []
    model.eval()
    
    with torch.no_grad():
        for text1, text2 in zip(eval_texts1, eval_texts2):
            inputs = tokenizer(text1, text2, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            predictions = outputs.logits.cpu().numpy().flatten()
            
            all_predictions.extend(predictions)
    
    predictions = np.array(all_predictions)
    ground_truth = np.array(ground_truth)
    
    pearson_corr, _ = pearsonr(predictions, ground_truth)
    mse = mean_squared_error(ground_truth, predictions)
    
    metrics = {
        'pearson_correlation': float(pearson_corr),
        'mse': float(mse)
    }
    
    return {
        'metrics': metrics,
        'predictions': predictions.tolist(),
        'ground_truth': ground_truth.tolist()
    }

def evaluate_glue_nli_task(model, tokenizer, dataset_name, dataset_config, text_column, text_column2, label_column, num_classes, class_names, device):
    dataset = load_dataset(dataset_name, dataset_config, split='validation')
    
    eval_texts1 = dataset[text_column]
    eval_texts2 = dataset[text_column2]
    ground_truth = dataset[label_column]
    
    all_predictions = []
    model.eval()
    
    with torch.no_grad():
        for text1, text2 in zip(eval_texts1, eval_texts2):
            inputs = tokenizer(text1, text2, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
    
    predictions = np.array(all_predictions)
    ground_truth = np.array(ground_truth)
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        y_true_binary = (ground_truth == i).astype(int)
        y_pred_binary = (predictions == i).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        
        class_metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        }
    
    overall_metrics = {}
    overall_metrics['micro_f1'] = float(f1_score(ground_truth, predictions, average='micro', zero_division=0))
    overall_metrics['macro_f1'] = float(f1_score(ground_truth, predictions, average='macro', zero_division=0))
    overall_metrics['weighted_f1'] = float(f1_score(ground_truth, predictions, average='weighted', zero_division=0))
    overall_metrics['accuracy'] = float(accuracy_score(ground_truth, predictions))
    
    return {
        'class_metrics': class_metrics,
        'overall_metrics': overall_metrics,
        'predictions': predictions.tolist(),
        'ground_truth': ground_truth.tolist(),
        'class_names': class_names
    }

def evaluate_single_checkpoint(checkpoint_path, task_config, device):

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    model = model.to(device)
    model.eval()
    
    if task_config['type'] == 'multilabel':
        results = evaluate_multilabel_task(
            model, tokenizer, 
            task_config.get('eval_data'), 
            task_config['text_column'], 
            task_config['impact_columns'], 
            device,
            task_config.get('dataset_name')
        )
    # elif task_config['type'] == 'binary':
    #     results = evaluate_binary_task(
    #         model, tokenizer, 
    #         task_config['dataset_name'], 
    #         task_config['text_column'], 
    #         device
    #     )
    elif task_config['type'] == 'multiclass':
        results = evaluate_multiclass_task(
            model, tokenizer, 
            task_config['dataset_name'], 
            task_config['text_column'], 
            task_config['label_column'],
            task_config['num_classes'],
            task_config['class_names'],
            device
        )
    elif task_config['type'] == 'binary_numeric':
        results = evaluate_binary_numeric_task(
            model, tokenizer, 
            task_config['dataset_name'], 
            task_config['text_column'], 
            task_config['label_column'],
            device
        )
    elif task_config['type'] == 'glue_classification':
        results = evaluate_glue_classification_task(
            model, tokenizer, 
            task_config['dataset_name'], 
            task_config['dataset_config'],
            task_config['text_column'], 
            task_config['label_column'],
            task_config['num_classes'],
            task_config['class_names'],
            device
        )
    elif task_config['type'] == 'glue_paraphrase':
        results = evaluate_glue_paraphrase_task(
            model, tokenizer, 
            task_config['dataset_name'], 
            task_config['dataset_config'],
            task_config['text_column'], 
            task_config['text_column2'],
            task_config['label_column'],
            task_config['num_classes'],
            task_config['class_names'],
            device
        )
    elif task_config['type'] == 'glue_regression':
        results = evaluate_glue_regression_task(
            model, tokenizer, 
            task_config['dataset_name'], 
            task_config['dataset_config'],
            task_config['text_column'], 
            task_config['text_column2'],
            task_config['label_column'],
            device
        )
    elif task_config['type'] == 'glue_nli':
        results = evaluate_glue_nli_task(
            model, tokenizer, 
            task_config['dataset_name'], 
            task_config['dataset_config'],
            task_config['text_column'], 
            task_config['text_column2'],
            task_config['label_column'],
            task_config['num_classes'],
            task_config['class_names'],
            device
        )
    else:
        print(f"Unknown task type: {task_config['type']}")
        return None
    


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation script for multi-task ModernBERT models")
    parser.add_argument("--config_file", type=str, 
                       default="config.json",
                       help="Path to configuration JSON file")
    parser.add_argument("--checkpoints_dir", type=str, default=None,
                       help="Directory containing all task checkpoints (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save evaluation results (overrides config)")
    parser.add_argument("--eval_only", type=str, default=None,
                       help="Evaluate only a specific task (optional)")
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    if args.checkpoints_dir:
        config['defaults']['base_save_dir'] = args.checkpoints_dir
    if args.output_dir:
        config['defaults']['benchmark_output_dir'] = args.output_dir
    
    checkpoints_dir = config['defaults']['base_save_dir']
    output_dir = config['defaults']['benchmark_output_dir']
    tasks = config['tasks']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.eval_only:
        if args.eval_only not in tasks:
            return
        checkpoints_to_evaluate = {args.eval_only: os.path.join(checkpoints_dir, args.eval_only)}
    else:
        checkpoints_to_evaluate = {}
        for task_name in tasks.keys():
            checkpoint_path = os.path.join(checkpoints_dir, task_name)
            if os.path.exists(checkpoint_path):
                checkpoints_to_evaluate[task_name] = checkpoint_path
            else:
                print(f"Checkpoint not found for task '{task_name}': {checkpoint_path}")

    # Evaluate each checkpoint
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for task_name, checkpoint_path in checkpoints_to_evaluate.items():
        print(f"\n{'='*60}")
        print(f"Evaluating task: {task_name}")
        print(f"{'='*60}")
        
        task_config = tasks[task_name]
        results = evaluate_single_checkpoint(checkpoint_path, task_config, device)
        
        if results:
            all_results[task_name] = {
                'checkpoint_path': checkpoint_path,
                'task_config': task_config,
                'results': results,
                'timestamp': timestamp
            }
    
    results_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary CSV
    summary_data = []
    for task_name, task_results in all_results.items():
        if task_results['task_config']['type'] == 'multilabel':
            # For multilabel tasks, add each category metric
            for category, metrics in task_results['results']['category_metrics'].items():
                summary_data.append({
                    'task': task_name,
                    'category': category,
                    'type': 'category_metric',
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'accuracy': metrics['accuracy']
                })
            
            # Add overall metrics
            for metric_name, value in task_results['results']['overall_metrics'].items():
                summary_data.append({
                    'task': task_name,
                    'category': 'overall',
                    'type': metric_name,
                    'value': value,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'accuracy': None
                })
        elif task_results['task_config']['type'] == 'multiclass':
            # For multiclass tasks, add each class metric
            for class_name, metrics in task_results['results']['class_metrics'].items():
                summary_data.append({
                    'task': task_name,
                    'category': class_name,
                    'type': 'class_metric',
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'accuracy': metrics['accuracy']
                })
            
            # Add overall metrics
            for metric_name, value in task_results['results']['overall_metrics'].items():
                summary_data.append({
                    'task': task_name,
                    'category': 'overall',
                    'type': metric_name,
                    'value': value,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'accuracy': None
                })
        elif task_results['task_config']['type'] == 'binary_numeric':
            # For binary numeric tasks
            metrics = task_results['results']['metrics']
            summary_data.append({
                'task': task_name,
                'category': 'binary_numeric_classification',
                'type': 'overall',
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy']
            })
        elif task_results['task_config']['type'] in ['glue_classification', 'glue_paraphrase', 'glue_nli']:
            # For GLUE classification, paraphrase, and NLI tasks
            for class_name, metrics in task_results['results']['class_metrics'].items():
                summary_data.append({
                    'task': task_name,
                    'category': class_name,
                    'type': 'class_metric',
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'accuracy': metrics['accuracy']
                })
            
            # Add overall metrics
            for metric_name, value in task_results['results']['overall_metrics'].items():
                summary_data.append({
                    'task': task_name,
                    'category': 'overall',
                    'type': metric_name,
                    'value': value,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'accuracy': None
                })
        elif task_results['task_config']['type'] == 'glue_regression':
            # For GLUE regression tasks
            metrics = task_results['results']['metrics']
            summary_data.append({
                'task': task_name,
                'category': 'regression',
                'type': 'overall',
                'pearson_correlation': metrics['pearson_correlation'],
                'mse': metrics['mse'],
                'precision': None,
                'recall': None,
                'f1': None,
                'accuracy': None
            })
        else:
            # For binary tasks
            metrics = task_results['results']['metrics']
            summary_data.append({
                'task': task_name,
                'category': 'binary_classification',
                'type': 'overall',
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🎉 BENCHMARK EVALUATION COMPLETED!")
    print(f"{'='*60}")
    print(f"📁 Results saved to: {results_file}")
    print(f"📊 Summary saved to: {summary_file}")
    print(f"📋 Tasks evaluated: {len(all_results)}")
    
    # Print key metrics
    print(f"\n📊 KEY METRICS SUMMARY:")
    for task_name, task_results in all_results.items():
        print(f"\n{task_name}:")
        if task_results['task_config']['type'] == 'multilabel':
            print(f"  Overall F1 (micro): {task_results['results']['overall_metrics']['micro_f1']:.4f}")
            print(f"  Overall F1 (macro): {task_results['results']['overall_metrics']['macro_f1']:.4f}")
        elif task_results['task_config']['type'] == 'multiclass':
            print(f"  Overall F1 (micro): {task_results['results']['overall_metrics']['micro_f1']:.4f}")
            print(f"  Overall F1 (macro): {task_results['results']['overall_metrics']['macro_f1']:.4f}")
            print(f"  Overall Accuracy: {task_results['results']['overall_metrics']['accuracy']:.4f}")
        elif task_results['task_config']['type'] == 'binary_numeric':
            print(f"  F1 Score: {task_results['results']['metrics']['f1']:.4f}")
            print(f"  Accuracy: {task_results['results']['metrics']['accuracy']:.4f}")
        elif task_results['task_config']['type'] in ['glue_classification', 'glue_paraphrase', 'glue_nli']:
            print(f"  Overall F1 (micro): {task_results['results']['overall_metrics']['micro_f1']:.4f}")
            print(f"  Overall F1 (macro): {task_results['results']['overall_metrics']['macro_f1']:.4f}")
            print(f"  Overall Accuracy: {task_results['results']['overall_metrics']['accuracy']:.4f}")
        elif task_results['task_config']['type'] == 'glue_regression':
            print(f"  Pearson Correlation: {task_results['results']['metrics']['pearson_correlation']:.4f}")
            print(f"  MSE: {task_results['results']['metrics']['mse']:.4f}")
        else:
            print(f"  F1 Score: {task_results['results']['metrics']['f1']:.4f}")
            print(f"  Accuracy: {task_results['results']['metrics']['accuracy']:.4f}")

if __name__ == '__main__':
    main() 