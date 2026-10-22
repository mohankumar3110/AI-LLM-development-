#!/usr/bin/env python3
"""Quick training pipeline - optimized for fast iteration on CPU"""

import os
import json
import logging
import sys
from pathlib import Path

try:
    import torch  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "The torch package is required to run this script. Install it with `pip install torch` "
        "or follow the official installation instructions at https://pytorch.org/."
    ) from exc
try:
    from datasets import load_dataset # type: ignore
except ImportError as exc:
    raise ImportError(
        "The datasets package is required to run this script. Install it with `pip install datasets` "
        "or follow the official installation instructions at https://huggingface.co/docs/datasets/installation"
    ) from exc
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
    MODEL_ID,
    MAX_SEQ_LENGTH,
    CHECKPOINT_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    LOGS_DIR,
    LORA_CONFIG,
    BNB_CONFIG,
)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOGS_DIR, "train.log")),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("QUICK TRAINING PIPELINE")
    logger.info("=" * 70)

    set_seed(42)

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        logger.info("Using CPU for training (will be slower)")
        device = "cpu"

    # Load tokenizer
    logger.info(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
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

    # Format datasets
    train_dataset = train_dataset.map(
        format_text,
        batched=False,
        remove_columns=["text"],
        desc="Formatting train data",
    )
    val_dataset = val_dataset.map(
        format_text,
        batched=False,
        remove_columns=["text"],
        desc="Formatting val data",
    )

    # Load model with quantization
    logger.info(f"Loading model: {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("Model loaded")

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    logger.info("Model prepared for 4-bit training")

    # Add LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    logger.info("LoRA adapters added")

    # Training
    logger.info("Starting training...")
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=1,  # Quick training
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.001,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        evaluation_strategy="epoch",
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
        logger.info(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
        logger.info(f"Final eval loss: {result.training_loss:.4f}")

        # Save report
        report = {
            "status": "success",
            "model": MODEL_ID,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "epochs": 1,
            "checkpoint": trainer.state.best_model_checkpoint,
        }
        report_path = os.path.join(LOGS_DIR, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {report_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
