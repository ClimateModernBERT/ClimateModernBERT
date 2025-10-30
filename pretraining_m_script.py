from transformers import DataCollatorForLanguageModeling, ModernBertForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch
import os, json
import numpy as np
import re
import wandb
import os

os.environ["WANDB_API_KEY"] = "da811774abaaa02b22f9b09516ef66786115c613"
os.environ["WANDB_MODE"] = "online"

def preprocess_text(txt):
    txt = "\n".join([i.strip() for i in txt.split("\n")])
    txt = re.sub(" +", " ", txt)
    return txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/cluster/project/sachan/yongan/modernbert_climate/modernbert-base-local')
    args = parser.parse_args()

    device = "cuda"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ModernBertForMaskedLM.from_pretrained(args.model_name)
    model = model.to(device)

    # Hyperparameters
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    wd = 8e-5
    lr = 3e-4
    betas = (0.9, 0.98)
    eps = 1e-6
    gradient_accumulation_steps = 512
    t_total = 10000
    warmup_steps = 50

    # WandB init
    wandb.init(
        project="modernbert-climate",
        name="finetune-modernbert",
        config={
            "learning_rate": lr,
            "weight_decay": wd,
            "betas": betas,
            "eps": eps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "total_steps": t_total,
            "warmup_steps": warmup_steps,
            "mlm_probability": 0.3,
            "model": args.model_name
        }
    )

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': wd},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps, betas=betas)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    model.zero_grad()
    optimizer.zero_grad()

    # Load dataset
    with open("/cluster/project/sachan/yongan/processed_data/combined_temp.txt", 'r', encoding='utf-8', errors='ignore') as f:
        text_data = f.read()

    accum_counter = 0
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.3)

    use_amp = True
    num_tokens_seen = 0
    num_steps = 0
    save_checkpoint = 0
    outpath = "/cluster/project/sachan/yongan/climate-modernbert-finetuned"

    os.makedirs(outpath, exist_ok=True)
    loss_array = []

    chunks = text_data.split('\n\n')

    for chunk in tqdm(chunks):
        txt = chunk.strip()
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

            # Step and optimizer update
            if accum_counter % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_counter = 0
                num_steps += 1

                print("loss", loss)
                print("num_tokens_seen", num_tokens_seen)

                # Log to wandb
                wandb.log({
                    "loss": loss.item(),
                    "step": num_steps,
                    "accum_counter": accum_counter,
                    "num_tokens_seen": num_tokens_seen
                })

            # Save every 3000 accum_counter
            if accum_counter % 3000 == 0:
                print("saving steps", num_steps)
                print("saving num tokens", num_tokens_seen)
                checkpoint_path = os.path.join(outpath, f"save_checkpoint_{save_checkpoint}")
                model.save_pretrained(checkpoint_path)
                with open(os.path.join(checkpoint_path, "loss.json"), "w") as f:
                    json.dump(loss_array, f)
                wandb.save(os.path.join(checkpoint_path, "*"))
                save_checkpoint += 1

    print("Saving final model...")
    final_path = os.path.join(outpath, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    wandb.finish()
