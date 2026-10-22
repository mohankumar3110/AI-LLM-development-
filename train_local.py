#!/usr/bin/env python3
"""Test training with smaller alternative model that doesn't require auth"""

import os
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset # type: ignore
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType # type: ignore
from transformers import ( # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import (
    MAX_SEQ_LENGTH,
    CHECKPOINT_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    LOGS_DIR,
)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOGS_DIR, "train_local.log")),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("TRAINING WITH LOCAL/ALTERNATIVE MODEL")
    logger.info("=" * 70)

    set_seed(42)

    # Use alternative model that doesn't require auth
    # These are open models available without authentication
    models_to_try = [
        "gpt2",  # Small, fast, no auth needed
        "meta-llama/Llama-2-7b",  # Alternative
    ]

    model_id = models_to_try[0]  # Try GPT2 first
    logger.info(f"Using model: {model_id}")

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        logger.info("Using CPU for training")
        device = "cpu"

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Format data
    def format_text(example):
        text = example["text"]
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    val_dataset = load_dataset("json", data_files=VAL_DATA_PATH, split="train")

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Val set: {len(val_dataset)} samples")

    # Limit to 100 samples for quick test
    train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
    val_dataset = val_dataset.select(range(min(20, len(val_dataset))))
    logger.info(f"Using reduced dataset for quick test: {len(train_dataset)} train, {len(val_dataset)} val")

    # Format datasets
    original_columns_train = train_dataset.column_names
    train_dataset = train_dataset.map(
        format_text,
        batched=False,
        remove_columns=original_columns_train,
        desc="Formatting train data",
    )

    original_columns_val = val_dataset.column_names
    val_dataset = val_dataset.map(
        format_text,
        batched=False,
        remove_columns=original_columns_val,
        desc="Formatting val data",
    )

    # Load model
    logger.info(f"Loading model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    logger.info("Model loaded")

    # Training
    logger.info("Starting training...")
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir=LOGS_DIR,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    try:
        result = trainer.train()
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Checkpoint: {trainer.state.best_model_checkpoint}")

        # Save report
        report = {
            "status": "success",
            "model": model_id,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "epochs": 1,
            "checkpoint": trainer.state.best_model_checkpoint,
        }
        report_path = os.path.join(LOGS_DIR, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report: {report_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
