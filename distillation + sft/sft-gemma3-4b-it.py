#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
google/gemma-3-4b-it Fine-tuning Script for Prompt-Response Dataset
Optimized for fine-tuning on prompt-response pairs
"""

import os
import json
import logging
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import os
os.environ["PYTORCH_SDP_BACKEND"] = "math"

import torch
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    AutoPeftModelForCausalLM
)
import bitsandbytes as bnb
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import EarlyStoppingCallback

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =======================================================
# HYPERPARAMETERS AND CONFIGURATION
# =======================================================

@dataclass
class FinetuningConfig:
    """Configuration for fine-tuning gemma-3 model"""

    # Fine tuning the google/gemma-3-4b-it model on a prompt-response dataset.
    
    # Model Parameters
    model_name_or_path: str = field(
        default="google/gemma-3-4b-it",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: str = field(
        default=None,
        metadata={"help": "Tokenizer name or path (defaults to model name if not specified)"}
    )
    
    # Data Parameters
    data_path: str = field(
        default="/home/sriramsrinivasan/SFT/final_gemini_data/gemini-2.0-flash.json",
        metadata={"help": "Path to the JSON file containing the fine-tuning data"}
    )
    output_dir: str = field(
        default="./finetuned-gemma-3-4b-it",
        metadata={"help": "Directory to save model checkpoints and outputs"}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length for training data"}
    )
    
    # Training Parameters
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning rate"}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before backward pass"}
    )
    num_train_epochs: float = field(
        default=10.0,
        metadata={"help": "Total number of training epochs"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay applied to parameters"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of total training steps used for warmup"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run evaluation every X steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X steps"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Limit the total number of checkpoints, delete old ones"}
    )
    
    # PEFT/LoRA Parameters
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout value"}
    )
    
    # Quantization Parameters
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_nested_quant: bool = field(
        default=True,
        metadata={"help": "Whether to use nested quantization for 4-bit (double quantization)"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type for 4-bit quantization (fp4 or nf4)"}
    )
    
    # Early Stopping
    early_stopping_patience: int = field(
        default=10,
        metadata={"help": "Stop training when the evaluation metric worsens for this many evaluation steps"}
    )
    
    # Wandb Parameters
    wandb_project: str = field(
        default="gemma-3-4b-it-finetuning",
        metadata={"help": "Wandb project name"}
    )
    wandb_run_name: str = field(
        default=None,
        metadata={"help": "Wandb run name (defaults to timestamp if not specified)"}
    )
    
    # Other Parameters
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )
    training_split: float = field(
        default=0.9,
        metadata={"help": "Fraction of data to use for training vs validation"}
    )

    def __post_init__(self):
        # Set default tokenizer to model path if not specified
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
            
        # Set wandb run name if not specified
        if self.wandb_run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = f"gemma-3-4b-it_finetune_{timestamp}"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)


# =======================================================
# DATA PROCESSING FUNCTIONS
# =======================================================

def load_and_prepare_data(config: FinetuningConfig):
    """Load and prepare the dataset for fine-tuning"""
    
    # Load the JSON data
    with open(config.data_path, 'r') as file:
        data = json.load(file)
    
    # Create formatted data for fine-tuning
    formatted_data = []
    
    for item in data:
        # Check for the new prompt-response format
        prompt = item.get('prompt', '')
        response = item.get('response', '')
        
        if prompt and response:
            formatted_data.append({
                'prompt': prompt,
                'response': response
            })
    
    logger.info(f"Loaded {len(formatted_data)} examples from {config.data_path}")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    
    # Split into train and validation sets
    dataset = dataset.shuffle(seed=config.seed)
    split_dataset = dataset.train_test_split(
        test_size=(1-config.training_split),
        seed=config.seed
    )
    
    logger.info(f"Split dataset: {len(split_dataset['train'])} training examples, "
                f"{len(split_dataset['test'])} validation examples")
    
    return split_dataset


def format_prompt(prompt, tokenizer):
    """Format the prompt for gemma-3-4b-it model"""
    
    # For gemma-3-4b-it format
    messages = [
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_data(dataset, tokenizer, config: FinetuningConfig):
    """Tokenize and prepare the dataset for training"""
    
    def tokenize_function(examples):
        # Create properly formatted prompts
        formatted_prompts = [format_prompt(p, tokenizer) for p in examples["prompt"]]
        
        # Combine prompts with responses (for a causal LM setting)
        completions = []
        for formatted_prompt, response in zip(formatted_prompts, examples["response"]):
            completions.append(f"{formatted_prompt}{response}{tokenizer.eos_token}")
        
        # Tokenize completions
        tokenized = tokenizer(
            completions,
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt"
        )
        
        # Create labels for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        # Mask out prompt tokens for loss calculation
        for i, formatted_prompt in enumerate(formatted_prompts):
            prompt_length = len(tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"])
            tokenized["labels"][i, :prompt_length] = -100
        
        return tokenized
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "response"],
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


# =======================================================
# MODEL INITIALIZATION FUNCTIONS
# =======================================================

def get_compute_dtype(config: FinetuningConfig):
    """Get the compute dtype for the model"""
    if config.bnb_4bit_compute_dtype == "float16":
        return torch.float16
    elif config.bnb_4bit_compute_dtype == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float32


def create_bnb_config(config: FinetuningConfig):
    """Create BitsAndBytes configuration for quantization"""
    if config.use_4bit:
        compute_dtype = get_compute_dtype(config)
        
        bnb_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": config.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_use_double_quant": config.use_nested_quant,
        }
        
        logger.info(f"Using 4-bit quantization with {bnb_config}")
        return bnb_config
    else:
        return {}


def create_peft_config(config: FinetuningConfig):
    """Create PEFT configuration for LoRA"""
    if config.use_lora:
        # Define target modules for gemma-3-4b-it 
        # These are typical attention modules, adjust as needed for gemma-3-4b-it  architecture
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "dense_h_to_4h", "dense_4h_to_h"]

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        logger.info(f"Using LoRA with config: {peft_config}")
        return peft_config
    else:
        return None


def load_model_and_tokenizer(config: FinetuningConfig):
    """Load the model and tokenizer with appropriate configurations"""
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name_or_path,
        trust_remote_code=True,
        use_fast=False  # Some tokenizers need this
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create BitsAndBytes config for quantization if needed
    quantization_config = create_bnb_config(config)
    
    # Load the base model
    logger.info(f"Loading base model: {config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=quantization_config if quantization_config else None,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for kbit training if using quantization
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters if using LoRA
    if config.use_lora:
        peft_config = create_peft_config(config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


# =======================================================
# TRAINING FUNCTIONS
# =======================================================

def train(config: FinetuningConfig):
    """Train the model with the given configuration"""
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=vars(config)
    )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load and prepare data
    dataset = load_and_prepare_data(config)
    tokenized_dataset = tokenize_data(dataset, tokenizer, config)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        num_train_epochs=config.num_train_epochs,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        remove_unused_columns=False,
        fp16=False,
        bf16=False,
        # fp16=True if get_compute_dtype(config) == torch.float16 else False,
        # bf16=True if get_compute_dtype(config) == torch.bfloat16 else False,
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Check for existing checkpoint
    last_checkpoint = get_last_checkpoint(config.output_dir)
    resume_from_checkpoint = last_checkpoint if last_checkpoint else None
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    logger.info("Saving final model...")
    if config.use_lora:
        model.save_pretrained(os.path.join(config.output_dir, "final_model"))
    else:
        trainer.save_model(os.path.join(config.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final_model"))
    
    # End wandb run
    wandb.finish()
    
    return model, tokenizer


# =======================================================
# MAIN FUNCTION
# =======================================================

def main():
    # Parse arguments


    parser = HfArgumentParser(FinetuningConfig)
    config = parser.parse_args_into_dataclasses()[0]
    
    # Log configuration
    logger.info(f"Training configuration: {config}")
    
    # Start training
    train(config)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()