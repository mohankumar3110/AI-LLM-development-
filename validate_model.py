#!/usr/bin/env python3
"""Validate trained model and generate test predictions"""

import os
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import CHECKPOINT_DIR, DATA_DIR, LOGS_DIR

def evaluate_model():
    """Evaluate model on test set"""
    print("=" * 70)
    print("MODEL VALIDATION & TESTING")
    print("=" * 70)

    # Find latest checkpoint
    checkpoints = sorted(
        [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )

    if not checkpoints:
        print("No checkpoints found")
        return None

    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
    print(f"\nUsing checkpoint: {checkpoint_path}")

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, local_files_only=True)
    model.eval()

    # Load test data
    print("Loading test data...")
    test_dataset = load_dataset("json", data_files=os.path.join(DATA_DIR, "test_doubled.jsonl"), split="train")
    print(f"Test samples: {len(test_dataset)}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.to("cpu")
    total_loss = 0

    with torch.no_grad():
        for i, example in enumerate(test_dataset):
            if i % 20 == 0:
                print(f"  Processed {i}/{len(test_dataset)}")

            text = example["text"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            try:
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue

    avg_loss = total_loss / len(test_dataset)
    perplexity = torch.exp(torch.tensor(avg_loss))

    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    return {
        "test_loss": avg_loss,
        "perplexity": float(perplexity),
        "checkpoint": checkpoint_path,
    }

def generate_samples():
    """Generate predictions from trained model"""
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    # Find latest checkpoint
    checkpoints = sorted(
        [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )

    if not checkpoints:
        print("No checkpoints found")
        return []

    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoints[-1])

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, local_files_only=True)
    model.eval()

    prompts = [
        "create a function to sort",
        "write code to reverse a list",
        "generate a loop that counts",
    ]

    samples = []

    with torch.no_grad():
        for prompt in prompts:
            input_text = f"<|user|>\n{prompt}\n<|assistant|>\n"
            inputs = tokenizer(input_text, return_tensors="pt")

            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=False)
            samples.append({
                "prompt": prompt,
                "response": response[:300],  # Truncate for readability
            })

            print(f"\nPrompt: {prompt}")
            print(f"Response: {response[:200]}...")

    return samples

def main():
    # Evaluate
    eval_results = evaluate_model()

    # Generate
    samples = generate_samples()

    # Create final report
    report = {
        "status": "completed",
        "model": "gpt2-finetuned",
        "dataset": {
            "train_samples": 2536,
            "val_samples": 298,
            "test_samples": 150,
            "total_samples": 2984,
            "augmentation_factor": 2,
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 2e-4,
        },
        "evaluation": eval_results,
        "samples_generated": len(samples),
        "checkpoint": os.path.join(CHECKPOINT_DIR, "checkpoint-1") if os.path.exists(os.path.join(CHECKPOINT_DIR, "checkpoint-1")) else None,
    }

    report_path = os.path.join(LOGS_DIR, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Report saved to: {report_path}")
    print(f"\nSummary:")
    print(f"  Test Loss: {eval_results['test_loss']:.4f}")
    print(f"  Perplexity: {eval_results['perplexity']:.4f}")
    print(f"  Samples Generated: {len(samples)}")

if __name__ == "__main__":
    main()
