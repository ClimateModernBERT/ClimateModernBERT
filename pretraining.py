from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AdamW, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling, ModernBertForMaskedLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AdamW, get_linear_schedule_with_warmup

from torch.optim import AdamW
import torch
from torch.cuda.amp import autocast
import os
import json
from tqdm import tqdm
import random
import numpy as np

def preprocess_text(txt):
    txt = "\n".join([i.strip() for i in txt.split("\n")])
    txt = re.sub(" +", " ", txt)
    return txt

def main():
    
    # Model configuration
    model_name = 'answerdotai/ModernBERT-base'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Downloading model: {model_name}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model = model.to(device)
    
    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training hyperparameters (based on ModernBERT paper)
    learning_rate = 3e-4
    weight_decay = 8e-5
    betas = (0.9, 0.98)
    eps = 1e-6
    gradient_accumulation_steps = 512
    max_seq_length = 8192
    mlm_probability = 0.3
    warmup_steps = 500
    max_steps = 10000
    save_steps = 1000
    
    # Setup optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps, betas=betas)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=max_steps
    )
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm_probability=mlm_probability
    )
    
    # Output directory
    output_dir = "/cluster/project/sachan/yongan/climate-modernbert-pretrained"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load climate data
    data_path = "/cluster/project/sachan/yongan/processed_data/combined_temp.txt"
    print(f"Loading data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        all_text = f.read()
    
    # Split into chunks
    print("Preprocessing and chunking text...")
    all_text = preprocess_text(all_text)
    tokens = tokenizer.tokenize(all_text)
    
    # Create training chunks
    chunk_size = max_seq_length - 2  # Account for [CLS] and [SEP]
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        if len(chunk) > 100:  # Skip very short chunks
            chunks.append(chunk)
    
    print(f"Created {len(chunks)} training chunks")
    
    # Training loop
    model.train()
    model.zero_grad()
    
    loss_history = []
    num_tokens_seen = 0
    global_step = 0
    accumulation_counter = 0
    
    pbar = tqdm(total=max_steps, desc="Training")
    
    while global_step < max_steps:
        random.shuffle(chunks)
        
        for chunk in chunks:
            if global_step >= max_steps:
                break
                
            # Convert tokens to input_ids
            input_ids = tokenizer.convert_tokens_to_ids(chunk)
            input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
            
            # Create batch
            batch = {
                'input_ids': torch.tensor([input_ids], device=device),
                'attention_mask': torch.ones(1, len(input_ids), device=device)
            }
            
            # Apply data collator for MLM
            batch = data_collator([input_ids])
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(dtype=torch.bfloat16, enabled=True):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
            
            loss_history.append(loss.item() * gradient_accumulation_steps)
            num_tokens_seen += len(input_ids)
            
            # Backward pass
            loss.backward()
            accumulation_counter += 1
            
            # Update weights
            if accumulation_counter % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accumulation_counter = 0
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss_history[-1]:.4f}",
                    'tokens': f"{num_tokens_seen:,}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Save training info
                    training_info = {
                        'step': global_step,
                        'tokens_seen': num_tokens_seen,
                        'loss_history': loss_history[-save_steps:],
                        'learning_rate': scheduler.get_last_lr()[0]
                    }
                    with open(os.path.join(checkpoint_dir, 'training_info.json'), 'w') as f:
                        json.dump(training_info, f, indent=2)
                    
                    print(f"\nSaved checkpoint at step {global_step}")
    
    pbar.close()
    
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump({
            'loss_history': loss_history,
            'total_steps': global_step,
            'total_tokens': num_tokens_seen
        }, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Total steps: {global_step}")
    print(f"Total tokens seen: {num_tokens_seen:,}")
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
