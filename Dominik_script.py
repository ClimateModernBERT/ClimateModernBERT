from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AdamW, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling, ModernBertForMaskedLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AdamW, get_linear_schedule_with_warmup

import argparse
import datasets
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch
import os, json
import numpy as np
import re


def preprocess_text(txt):
    txt = "\n".join([i.strip() for i in txt.split("\n")])
    txt = re.sub(" +", " ", txt)
    return txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/scratch/gpfs/ds8100/transformer_cache/ModernBERT-base')

    args = parser.parse_args()

    #device = "cpu"
    device = "cuda"

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ModernBertForMaskedLM.from_pretrained(args.model_name)
    model = model.to(device)

    # some hyper-params, inspired by the paper and modernbert fine-tuning scripts in the official github rep

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    wd = 8e-5
    lr = 3e-4
    betas = (0.9, 0.98)
    eps = 1e-6

    gradient_accumulation_steps = 512
    t_total = 10000
    warmup_steps = 50

    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': wd},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps, betas=betas)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    model.zero_grad()
    optimizer.zero_grad()

    # load dataset
    dataset_name = "/scratch/gpfs/ds8100/datasets/cold-cases"
    train = datasets.load_dataset(dataset_name, cache_dir="/scratch/gpfs/ds8100/transformer_cache")["train"]

    # shuffle
    iterable_dataset = train.to_iterable_dataset(num_shards=139)
    shuffled_iterable_dataset = iterable_dataset.shuffle(seed=42, buffer_size=1000)

    # tokenzier 
    accum_counter = 0
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.3)

    # record some statistics during training
    use_amp = True
    num_tokens_seen = 0
    num_steps = 0
    save_checkpoint = 0
    outpath = "/scratch/gpfs/ds8100/legal-modernbert-finetuned"

    try:
        os.makedirs(outpath, exist_ok=True)
    except:
        pass
    loss_array = []

    for row in tqdm(shuffled_iterable_dataset, total=len(train)):
        # some rows in the dataset don't contain text
        opinion = row["opinions"]
        if opinion is None:
            continue
        for item in opinion:
            txt = item["opinion_text"] 
            # some or nan
            if not isinstance(txt, str):
                continue
            # some are too short
            if len(txt.split()) < 500:
                continue

            txt = preprocess_text(txt)
            input_ids = tokenizer(txt, add_special_tokens=False).input_ids

            for idx in range(0, len(input_ids), 8190):
                batch_ids = np.array([[tokenizer.cls_token_id] + input_ids[idx:idx+8190] + [tokenizer.sep_token_id]])
                if len(batch_ids[0]) < 500:
                    continue
                num_tokens_seen += len(batch_ids[0])
                batch = data_collator(batch_ids)
                model.train()
                batch["input_ids"] = batch["input_ids"].to(device)
                batch["attention_mask"] = torch.ones_like(batch["input_ids"]).to(device)
                batch["labels"] = batch["labels"].to(device)
                with autocast(dtype=torch.bfloat16, enabled=use_amp):    
                    output = model(**batch)
                    loss = output.loss / gradient_accumulation_steps
                    loss_array.append(loss.item())
                loss.backward()
                accum_counter += 1
                if accum_counter % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    accum_counter = 0
                    num_steps += 1
                    print ("loss", loss)
                    print ("num_tokens_seen", num_tokens_seen)
                    if num_steps % 250 == 0:
                        print ("saving steps", num_steps)
                        print ("saving num tokens", num_tokens_seen)
                        model.save_pretrained(os.path.join(outpath, "save_checkpoint_" + str(save_checkpoint)))
                        with open(os.path.join(outpath, "save_checkpoint_" + str(save_checkpoint), "loss.json"), "w") as f:
                            json.dump(loss_array, f)
                        save_checkpoint += 1

# sbatch --nodes=1 --ntasks=1 --time=71:59:00 --mem=12G --gres=gpu:1 --partition=pli-c --wrap "python climate_pretraining_modernbert.py"