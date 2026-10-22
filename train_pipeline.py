#!/usr/bin/env python3
"""
Complete LLM Training Pipeline
Handles: Data validation, training, validation, and inference
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from datasets import load_dataset, DatasetDict # type: ignore
import peft # type: ignore
from torch.utils.data import DataLoader
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
    TEST_DATA_PATH,
    LOGS_DIR,
    LORA_CONFIG,
    BNB_CONFIG,
    TRAIN_CONFIG,
    ASSISTANT_TOKEN,
    USER_TOKEN,
    EOS_TOKEN,
    MERGED_DIR,
)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOGS_DIR, "train.log")),
    ],
)
logger = logging.getLogger(__name__)


def check_gpu() -> None:
    """Check GPU availability and print info."""
    if not torch.cuda.is_available():
        logger.warning("⚠ CUDA not available — training will be slow on CPU")
    else:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gb = props.total_memory / 1024 ** 3
            logger.info(f"✓ GPU {i}: {props.name} — {gb:.1f} GB VRAM")


def load_tokenizer(model_id: str):
    """Load tokenizer from model."""
    logger.info(f"Loading tokenizer from {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_auth_token=True,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer)}")
    return tokenizer


def format_chat_template(example: Dict, tokenizer) -> Dict:
    """Format examples to chat template."""
    text = example.get("text", "")

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def load_datasets(train_path: str, val_path: str, test_path: str, tokenizer) -> DatasetDict:
    """Load and format datasets."""
    logger.info("Loading datasets …")

    dataset_dict = DatasetDict()

    for split, path in [("train", train_path), ("validation", val_path), ("test", test_path)]:
        if os.path.exists(path):
            logger.info(f"  Loading {split} from {path} …")
            dataset = load_dataset("json", data_files=path, split="train")

            dataset = dataset.map(
                lambda x: format_chat_template(x, tokenizer),
                batched=False,
                desc=f"Formatting {split}",
                remove_columns=["text"] if "text" in dataset.column_names else [],
            )
            dataset_dict[split] = dataset
            logger.info(f"  ✓ {split}: {len(dataset)} examples")

    return dataset_dict


def create_model(model_id: str):
    """Create quantized and LoRA-adapted model."""
    logger.info(f"Creating model with 4-bit quantization …")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=BNB_CONFIG.load_in_4bit,
        bnb_4bit_quant_type=BNB_CONFIG.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=BNB_CONFIG.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True,
    )
    logger.info(f"✓ Model loaded: {model_id}")

    model = peft.prepare_model_for_kbit_training(model)
    logger.info("✓ Model prepared for 4-bit training")

    lora_config = peft.LoraConfig(
        r=LORA_CONFIG.r,
        lora_alpha=LORA_CONFIG.lora_alpha,
        target_modules=LORA_CONFIG.target_modules,
        lora_dropout=LORA_CONFIG.lora_dropout,
        bias=LORA_CONFIG.bias,
        task_type=peft.TaskType.CAUSAL_LM,
    )

    model = peft.get_peft_model(model, lora_config)
    logger.info("✓ LoRA adapters applied")

    return model


def train_model(model, datasets: DatasetDict, tokenizer) -> str:
    """Train the model."""
    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)

    training_args = TrainingArguments(
        output_dir=TRAIN_CONFIG.output_dir,
        num_train_epochs=TRAIN_CONFIG.num_train_epochs,
        per_device_train_batch_size=TRAIN_CONFIG.per_device_train_batch_size,
        per_device_eval_batch_size=TRAIN_CONFIG.per_device_eval_batch_size,
        gradient_accumulation_steps=TRAIN_CONFIG.gradient_accumulation_steps,
        learning_rate=TRAIN_CONFIG.learning_rate,
        weight_decay=TRAIN_CONFIG.weight_decay,
        warmup_ratio=TRAIN_CONFIG.warmup_ratio,
        lr_scheduler_type=TRAIN_CONFIG.lr_scheduler_type,
        fp16=TRAIN_CONFIG.fp16,
        optim=TRAIN_CONFIG.optim,
        logging_steps=TRAIN_CONFIG.logging_steps,
        evaluation_strategy=TRAIN_CONFIG.eval_strategy,
        save_strategy=TRAIN_CONFIG.save_strategy,
        save_total_limit=TRAIN_CONFIG.save_total_limit,
        load_best_model_at_end=TRAIN_CONFIG.load_best_model_at_end,
        metric_for_best_model=TRAIN_CONFIG.metric_for_best_model,
        greater_is_better=False,
        report_to="none",
        logging_dir=LOGS_DIR,
        seed=TRAIN_CONFIG.seed,
        dataloader_num_workers=0,
        group_by_length=True,
        max_grad_norm=TRAIN_CONFIG.max_grad_norm,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    try:
        result = trainer.train()
        logger.info("=" * 70)
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        best_checkpoint = trainer.state.best_model_checkpoint
        logger.info(f"Best checkpoint: {best_checkpoint}")

        return best_checkpoint
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        raise


def evaluate_model(model, datasets: DatasetDict, tokenizer) -> Dict:
    """Evaluate model on validation and test sets."""
    logger.info("=" * 70)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 70)

    results = {}

    for split in ["validation", "test"]:
        if split not in datasets:
            continue

        logger.info(f"Evaluating on {split} …")

        data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
        eval_dataloader = DataLoader(
            datasets[split],
            batch_size=1,
            collate_fn=data_collator,
        )

        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(eval_dataloader)
        results[split] = {"loss": avg_loss}
        logger.info(f"✓ {split} loss: {avg_loss:.4f}")

    return results


def generate_samples(model, tokenizer, num_samples: int = 3) -> List[str]:
    """Generate samples from trained model."""
    logger.info("=" * 70)
    logger.info("GENERATING SAMPLES")
    logger.info("=" * 70)

    prompts = [
        "Write a Python function to sum two numbers",
        "Create a function that reverses a list",
        "Write code that checks if a number is prime",
    ]

    samples = []
    model.eval()

    with torch.no_grad():
        for prompt in prompts[:num_samples]:
            input_text = f"{USER_TOKEN}\n{prompt}\n{ASSISTANT_TOKEN}\n"
            inputs = tokenizer(input_text, return_tensors="pt")

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples.append({"prompt": prompt, "response": response})
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response}\n")

    return samples


def save_report(
    training_result: Optional[str],
    eval_results: Dict,
    samples: List[str],
) -> None:
    """Save training report."""
    report = {
        "timestamp": Path(LOGS_DIR).stat().st_mtime,
        "status": "completed",
        "best_checkpoint": training_result,
        "evaluation_results": eval_results,
        "samples_generated": len(samples),
        "model_id": MODEL_ID,
        "training_epochs": TRAIN_CONFIG.num_train_epochs,
        "effective_batch_size": (
            TRAIN_CONFIG.per_device_train_batch_size
            * TRAIN_CONFIG.gradient_accumulation_steps
        ),
    }

    report_path = os.path.join(LOGS_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {report_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train Llama 3.2 1B with LoRA")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-generate", action="store_true", help="Skip sample generation")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("LLAMA 3.2 1B LLM TRAINING PIPELINE")
    logger.info("=" * 70)

    set_seed(TRAIN_CONFIG.seed)
    check_gpu()

    tokenizer = load_tokenizer(MODEL_ID)
    datasets = load_datasets(TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, tokenizer)
    model = create_model(MODEL_ID)

    checkpoint = train_model(model, datasets, tokenizer)

    eval_results = {}
    if not args.skip_eval:
        eval_results = evaluate_model(model, datasets, tokenizer)

    samples = []
    if not args.skip_generate:
        samples = generate_samples(model, tokenizer)

    save_report(checkpoint, eval_results, samples)

    logger.info("=" * 70)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Log file: {os.path.join(LOGS_DIR, 'train.log')}")
    logger.info(f"Report: {os.path.join(LOGS_DIR, 'training_report.json')}")


if __name__ == "__main__":
    main()
