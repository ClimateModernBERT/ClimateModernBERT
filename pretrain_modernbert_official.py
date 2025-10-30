import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed
)
from accelerate import Accelerator
from accelerate.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: str = field(
        default='/cluster/project/sachan/yongan/modernbert_climate/modernbert-base-local',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(
        default="/cluster/project/sachan/yongan/processed_data/combined_temp.txt", 
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_seq_length: Optional[int] = field(
        default=8192,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        },
    )
    mlm_probability: float = field(
        default=0.3, 
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def main():
    # Initialize accelerator
    os.environ["WANDB_API_KEY"] = "da811774abaaa02b22f9b09516ef66786115c613"
    os.environ["WANDB_MODE"] = "online"
    accelerator = Accelerator(log_with="wandb")
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,
    )

    if accelerator.is_local_main_process:
        logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    if data_args.train_file is not None:
        train_dataset = load_dataset(
            'text',
            data_files=data_args.train_file,
            cache_dir=model_args.cache_dir,
            num_proc=data_args.preprocessing_num_workers,
            encoding='utf-8',
            errors='ignore'
        )['train']
    else:
        raise ValueError("Training requires a train_file")

    validation_dataset = None
    if data_args.validation_file is not None:
        validation_dataset = load_dataset(
            'text',
            data_files=data_args.validation_file,
            cache_dir=model_args.cache_dir,
            num_proc=data_args.preprocessing_num_workers,
            encoding='utf-8',
            errors='ignore'
        )['train']

    # Preprocessing the datasets
    def tokenize_function(examples):
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ] # remove empty lines
        
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length,
            return_special_tokens_mask=True,
        )

    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=["text"],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        if validation_dataset is not None:
            validation_dataset = validation_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=["text"],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_args.mlm_probability
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    # Prepare everything with our `accelerator`
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, trainer.optimizer, trainer.get_train_dataloader(), trainer.get_eval_dataloader()
    )

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        trainer.train()
        
        # Save model
        if accelerator.is_local_main_process:
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)
            logger.info(f"Model saved to {training_args.output_dir}")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        perplexity = torch.exp(torch.tensor(metrics["eval_loss"]))
        
        results = {"perplexity": perplexity}
        
        if accelerator.is_local_main_process:
            logger.info(f"Perplexity: {perplexity}")
            
            # Save metrics
            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key, value in results.items():
                    writer.write(f"{key} = {value}\n")

if __name__ == "__main__":
    main() 