import os
import logging
from dataclasses import dataclass, field
from typing import Optional
import re
import sys
import yaml
import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples to use (for debugging)"}
    )

def preprocess_text(txt):
    txt = "\n".join([i.strip() for i in txt.split("\n")])
    txt = re.sub(" +", " ", txt)
    return txt

def create_chunked_dataset(file_path, max_samples=None):
    def text_generator():
        count = 0
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_data = f.read()
            
            # Split into chunks like original script
            chunks = text_data.split('\n\n')
            
            for chunk in chunks:
                txt = chunk.strip()
                if len(txt.split()) < 500:  # Skip short chunks like original
                    continue
                
                txt = preprocess_text(txt)
                yield {"text": txt}
                count += 1
                
                if max_samples and count >= max_samples:
                    break
                    
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    # Create dataset from generator
    dataset = Dataset.from_generator(text_generator)
    return dataset

def main():
    # Initialize accelerator. Set WANDB_API_KEY in your shell environment before running.
    os.environ.setdefault("WANDB_MODE", "online")
    accelerator = Accelerator(log_with="wandb")

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 3 and sys.argv[1] == "--config":
        config_path = sys.argv[2]
        with open(config_path, "r") as f:
            raw_data = yaml.safe_load(f)

        # Split YAML dict into chunks for each dataclass
        model_keys = {field.name for field in ModelArguments.__dataclass_fields__.values()}
        data_keys = {field.name for field in DataTrainingArguments.__dataclass_fields__.values()}
        training_keys = {field.name for field in TrainingArguments.__dataclass_fields__.values()}

        model_dict = {k: raw_data[k] for k in raw_data if k in model_keys}
        data_dict = {k: raw_data[k] for k in raw_data if k in data_keys}
        training_dict = {k: raw_data[k] for k in raw_data if k in training_keys}

        # ✅ Correct parsing using separate parsers
        model_args = HfArgumentParser(ModelArguments).parse_dict(model_dict)[0]
        data_args = HfArgumentParser(DataTrainingArguments).parse_dict(data_dict)[0]
        training_args = HfArgumentParser(TrainingArguments).parse_dict(training_dict)[0]
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.gradient_checkpointing = False

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

    # Create datasets using chunked approach instead of load_dataset
    logger.info("Creating training dataset from chunks...")
    if data_args.train_file is not None:
        train_dataset = create_chunked_dataset(
            data_args.train_file, 
            max_samples=data_args.max_train_samples
        )
        logger.info(f"Created training dataset with {len(train_dataset)} samples")
    else:
        raise ValueError("Training requires a train_file")

    validation_dataset = None
    if data_args.validation_file is not None:
        logger.info("Creating validation dataset from chunks...")
        validation_dataset = create_chunked_dataset(data_args.validation_file)

    def tokenize_function(examples):
        examples["text"] = [
            line for line in examples["text"] if len(line) > 0 and not line.isspace()
        ]
        
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length,
            return_special_tokens_mask=True,
        )

    with accelerator.main_process_first():
        logger.info("Tokenizing training dataset...")
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,  # Process in smaller batches
            num_proc=1,  # Use single process to avoid memory issues
            remove_columns=["text"],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

        if validation_dataset is not None:
            logger.info("Tokenizing validation dataset...")
            validation_dataset = validation_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=100,
                num_proc=1,
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
            
            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key, value in results.items():
                    writer.write(f"{key} = {value}\n")

    # End tracking
    accelerator.end_training()

if __name__ == "__main__":
    main()

