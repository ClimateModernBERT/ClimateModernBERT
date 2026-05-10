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
    # Handle both local CSV files and Hugging Face datasets
    if eval_data_path and os.path.exists(eval_data_path):
        # Load from local CSV file
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
        # Load from Hugging Face dataset
        print(f"📊 Loading multilabel dataset from Hugging Face: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name, split='test')
            print(f"✅ Using test split with {len(dataset)} examples")
        except ValueError as e:
            if "Unknown split" in str(e):
                print(f"⚠️  Test split not found, using same split logic as fine-tuning")
                dataset = load_dataset(dataset_name, split='train')
                # Use same 80/20 split as fine-tuning script
                split_point = int(0.8 * len(dataset))
                dataset = dataset.select(range(split_point, len(dataset)))
                print(f"✅ Using train split subset with {len(dataset)} examples for evaluation")
            else:
                raise e
        
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
    else:
        raise ValueError("Either eval_data_path must be provided and exist, or dataset_name must be provided")
    
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
    print(f"📊 Evaluating multiclass task on: {dataset_name}")
    
    # Try to load test split, fallback to same split logic as fine-tuning
    try:
        dataset = load_dataset(dataset_name, split='test')
        print(f"✅ Using test split with {len(dataset)} examples")
    except ValueError as e:
        if "Unknown split" in str(e):
            print(f"⚠️  Test split not found, using same split logic as fine-tuning")
            dataset = load_dataset(dataset_name, split='train')
            # Use same 80/20 split as fine-tuning script
            split_point = int(0.8 * len(dataset))
            dataset = dataset.select(range(split_point, len(dataset)))
            print(f"✅ Using train split subset with {len(dataset)} examples for evaluation")
        else:
            raise e
    
    eval_texts = dataset[text_column]
    ground_truth = dataset[label_column]
    
    label_mapping = {name: i for i, name in enumerate(class_names)}
    ground_truth_indices = []
    for label in ground_truth:
        if isinstance(label, str):
            ground_truth_indices.append(label_mapping.get(label, 0))
        elif isinstance(label, (int, float)):
            idx = int(label)
            ground_truth_indices.append(idx if 0 <= idx < num_classes else 0)
        else:
            ground_truth_indices.append(0)
    
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
    print(f"📊 Evaluating binary numeric task on: {dataset_name}")
    
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
    print(f"📊 Evaluating GLUE classification task on: {dataset_name}/{dataset_config}")
    
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
    print(f"📊 Evaluating GLUE paraphrase task on: {dataset_name}/{dataset_config}")
    
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
    print(f"📊 Evaluating GLUE regression task on: {dataset_name}/{dataset_config}")
    
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
    
    # Scale predictions back from [0, 1] to [0, 5] to match ground truth
    predictions = np.array(all_predictions) * 5.0
    ground_truth = np.array(ground_truth)

    # For regression, calculate Pearson correlation and MSE
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
    print(f"📊 Evaluating GLUE NLI task on: {dataset_name}/{dataset_config}")
    
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

def compute_map_at_k(query_groups, k=None):
    """Compute Mean Average Precision (MAP@k) across all queries.

    For each query, compute AP by sorting candidate paragraphs by predicted score
    and computing precision at each relevant document position.
    """
    average_precisions = []
    for query, items in query_groups.items():
        # Sort by predicted score descending
        sorted_items = sorted(items, key=lambda x: x['score'], reverse=True)
        if k is not None:
            sorted_items = sorted_items[:k]

        num_relevant = 0
        precision_sum = 0.0
        for rank, item in enumerate(sorted_items, start=1):
            if item['binary_label'] == 1:
                num_relevant += 1
                precision_sum += num_relevant / rank

        total_relevant = sum(1 for item in items if item['binary_label'] == 1)
        if total_relevant > 0:
            average_precisions.append(precision_sum / total_relevant)
        else:
            average_precisions.append(0.0)

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def compute_ndcg_at_k(query_groups, k=None, use_graded=True):
    """Compute NDCG@k across all queries.

    Uses graded relevance scores when available, otherwise binary labels.
    """
    ndcg_scores = []
    for query, items in query_groups.items():
        # Sort by predicted score descending
        sorted_items = sorted(items, key=lambda x: x['score'], reverse=True)
        if k is not None:
            sorted_items = sorted_items[:k]

        rel_key = 'graded_relevance' if use_graded else 'binary_label'

        # DCG
        dcg = 0.0
        for rank, item in enumerate(sorted_items, start=1):
            rel = item[rel_key]
            dcg += (2 ** rel - 1) / np.log2(rank + 1)

        # Ideal DCG: sort by true relevance
        ideal_items = sorted(items, key=lambda x: x[rel_key], reverse=True)
        if k is not None:
            ideal_items = ideal_items[:k]

        idcg = 0.0
        for rank, item in enumerate(ideal_items, start=1):
            rel = item[rel_key]
            idcg += (2 ** rel - 1) / np.log2(rank + 1)

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def evaluate_retrieval_binary_task(model, tokenizer, local_data_path, query_column, text_column, label_column, relevance_threshold, device):
    from sklearn.model_selection import train_test_split as sk_train_test_split
    from datasets import Dataset as HFDataset

    print(f"📊 Evaluating retrieval binary classification task from: {local_data_path}")

    df = pd.read_csv(local_data_path, index_col=0)
    df = df.dropna(subset=[query_column, text_column, label_column])
    df = df.reset_index(drop=True)

    # Re-create the same 80/10/10 split as fine-tuning (seed=42) so test set matches
    indices = list(range(len(df)))
    train_indices, temp_indices = sk_train_test_split(indices, test_size=0.2, random_state=42)
    _, test_indices = sk_train_test_split(temp_indices, test_size=0.5, random_state=42)

    test_df = df.iloc[test_indices].reset_index(drop=True)
    print(f"✅ Using test split with {len(test_df)} examples")

    eval_queries = test_df[query_column].tolist()
    eval_articles = test_df[text_column].tolist()
    raw_relevance = [int(r) for r in test_df[label_column].tolist()]
    ground_truth = [1 if r >= relevance_threshold else 0 for r in raw_relevance]

    all_predictions = []
    all_scores = []  # softmax probability for the "answerable" class
    model.eval()

    with torch.no_grad():
        for query, article in zip(eval_queries, eval_articles):
            inputs = tokenizer(query, article, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            # Score = probability of the positive/answerable class (index 1)
            positive_score = probs[0, 1].cpu().item()

            all_predictions.extend(predictions)
            all_scores.append(positive_score)

    predictions = np.array(all_predictions)
    ground_truth = np.array(ground_truth)

    # Binary classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average='binary', zero_division=0)
    accuracy = accuracy_score(ground_truth, predictions)

    # Group by query for ranking metrics (MAP, NDCG)
    query_groups = {}
    for i, query in enumerate(eval_queries):
        if query not in query_groups:
            query_groups[query] = []
        query_groups[query].append({
            'score': all_scores[i],
            'binary_label': ground_truth[i],
            'graded_relevance': raw_relevance[i],
        })

    num_queries = len(query_groups)
    avg_docs_per_query = np.mean([len(items) for items in query_groups.values()])
    print(f"📊 Ranking metrics: {num_queries} unique queries, avg {avg_docs_per_query:.1f} docs/query")

    # Compute MAP and NDCG at various cutoffs
    map_full = compute_map_at_k(query_groups, k=None)
    map_5 = compute_map_at_k(query_groups, k=5)
    map_10 = compute_map_at_k(query_groups, k=10)

    ndcg_full = compute_ndcg_at_k(query_groups, k=None, use_graded=True)
    ndcg_5 = compute_ndcg_at_k(query_groups, k=5, use_graded=True)
    ndcg_10 = compute_ndcg_at_k(query_groups, k=10, use_graded=True)

    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'map': float(map_full),
        'map@5': float(map_5),
        'map@10': float(map_10),
        'ndcg': float(ndcg_full),
        'ndcg@5': float(ndcg_5),
        'ndcg@10': float(ndcg_10),
    }

    print(f"  Binary: F1={f1:.4f}, Acc={accuracy:.4f}")
    print(f"  Ranking: MAP={map_full:.4f}, MAP@5={map_5:.4f}, MAP@10={map_10:.4f}")
    print(f"  Ranking: NDCG={ndcg_full:.4f}, NDCG@5={ndcg_5:.4f}, NDCG@10={ndcg_10:.4f}")

    return {
        'metrics': metrics,
        'predictions': predictions.tolist(),
        'ground_truth': ground_truth.tolist(),
        'ranking_info': {
            'num_queries': num_queries,
            'avg_docs_per_query': float(avg_docs_per_query)
        }
    }


def evaluate_single_checkpoint(checkpoint_path, task_config, device):
    print(f"\n🔍 Evaluating checkpoint: {checkpoint_path}")
    
    try:
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
        elif task_config['type'] == 'retrieval_binary_classification':
            results = evaluate_retrieval_binary_task(
                model, tokenizer,
                task_config['local_data_path'],
                task_config['query_column'],
                task_config['text_column'],
                task_config['label_column'],
                task_config['relevance_threshold'],
                device
            )
        else:
            print(f"❌ Unknown task type: {task_config['type']}")
            return None
        
        print(f"✅ Evaluation completed for {checkpoint_path}")
        return results
        
    except Exception as e:
        print(f"❌ Error evaluating {checkpoint_path}: {str(e)}")
        return None

def aggregate_seed_results(all_seed_results, tasks_config):
    """Aggregate results across seeds and compute mean ± std."""
    aggregated = {}

    # Collect all task names across seeds
    all_task_names = set()
    for seed_results in all_seed_results.values():
        all_task_names.update(seed_results.keys())

    for task_name in sorted(all_task_names):
        task_type = tasks_config[task_name]['type']
        seed_metrics = []

        for seed, seed_results in all_seed_results.items():
            if task_name not in seed_results:
                continue
            results = seed_results[task_name]['results']

            if task_type == 'multilabel':
                seed_metrics.append(results['overall_metrics'])
            elif task_type in ('multiclass', 'glue_classification', 'glue_paraphrase', 'glue_nli'):
                seed_metrics.append(results['overall_metrics'])
            elif task_type == 'glue_regression':
                seed_metrics.append(results['metrics'])
            else:
                # binary_numeric, retrieval_binary_classification, etc.
                seed_metrics.append(results['metrics'])

        if not seed_metrics:
            continue

        # Compute mean ± std for each metric key
        all_keys = seed_metrics[0].keys()
        agg = {}
        for key in all_keys:
            values = [m[key] for m in seed_metrics if key in m]
            agg[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
        aggregated[task_name] = agg

    return aggregated


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
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 456],
                       help="List of seeds to evaluate across (default: 42 123 456)")
    args = parser.parse_args()
    
    print(f"📋 Loading configuration from: {args.config_file}")
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    print("✅ Configuration loaded successfully!")

    if args.output_dir:
        config['defaults']['benchmark_output_dir'] = args.output_dir

    original_checkpoints_dir = args.checkpoints_dir or config['defaults']['base_save_dir']
    output_dir = config['defaults']['benchmark_output_dir']
    tasks = config['tasks']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Multi-seed evaluation loop
    all_seed_results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"🌱 EVALUATING SEED {seed}")
        print(f"{'='*60}")

        checkpoints_dir = f"{original_checkpoints_dir}_seed{seed}"

        if not os.path.exists(checkpoints_dir):
            print(f"⚠️  Checkpoint directory not found: {checkpoints_dir}, skipping seed {seed}")
            continue

        if args.eval_only:
            if args.eval_only not in tasks:
                print(f"❌ Task '{args.eval_only}' not found in task configurations")
                return
            checkpoints_to_evaluate = {args.eval_only: os.path.join(checkpoints_dir, args.eval_only)}
        else:
            checkpoints_to_evaluate = {}
            for task_name in tasks.keys():
                checkpoint_path = os.path.join(checkpoints_dir, task_name)
                if os.path.exists(checkpoint_path):
                    checkpoints_to_evaluate[task_name] = checkpoint_path
                else:
                    print(f"⚠️  Checkpoint not found for task '{task_name}': {checkpoint_path}")

        print(f"📁 Checkpoints directory: {checkpoints_dir}")
        print(f"📋 Tasks to evaluate: {list(checkpoints_to_evaluate.keys())}")

        seed_results = {}
        for task_name, checkpoint_path in checkpoints_to_evaluate.items():
            print(f"\n--- Evaluating task: {task_name} (seed {seed}) ---")

            task_config = tasks[task_name]
            results = evaluate_single_checkpoint(checkpoint_path, task_config, device)

            if results:
                seed_results[task_name] = {
                    'checkpoint_path': checkpoint_path,
                    'task_config': task_config,
                    'results': results,
                    'timestamp': timestamp,
                    'seed': seed
                }

        all_seed_results[seed] = seed_results

        # Save per-seed results
        seed_results_file = os.path.join(output_dir, f"benchmark_results_seed{seed}_{timestamp}.json")
        with open(seed_results_file, 'w') as f:
            json.dump(seed_results, f, indent=2)
        print(f"✅ Seed {seed} results saved to: {seed_results_file}")

    # Aggregate results across seeds
    aggregated = aggregate_seed_results(all_seed_results, tasks)

    # Save aggregated results
    agg_results_file = os.path.join(output_dir, f"benchmark_aggregated_{timestamp}.json")
    with open(agg_results_file, 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Generate aggregated summary CSV
    summary_data = []
    for task_name, task_agg in aggregated.items():
        for metric_name, stats in task_agg.items():
            summary_data.append({
                'task': task_name,
                'metric': metric_name,
                'mean': stats['mean'],
                'std': stats['std'],
                'seeds': str(stats['values'])
            })

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"benchmark_summary_aggregated_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)

    # Print aggregated summary
    print(f"\n{'='*60}")
    print(f"🎉 MULTI-SEED BENCHMARK EVALUATION COMPLETED!")
    print(f"{'='*60}")
    print(f"📁 Aggregated results: {agg_results_file}")
    print(f"📊 Aggregated summary: {summary_file}")
    print(f"🌱 Seeds evaluated: {list(all_seed_results.keys())}")

    print(f"\n📊 AGGREGATED METRICS (mean ± std):")
    for task_name, task_agg in aggregated.items():
        print(f"\n{task_name}:")
        for metric_name, stats in task_agg.items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}  {stats['values']}")

if __name__ == '__main__':
    main() 
