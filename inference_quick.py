#!/usr/bin/env python3
"""Inference script - generate predictions from trained model"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
from peft import PeftModel # type: ignore

from config import MODEL_ID, CHECKPOINT_DIR, LOGS_DIR


def load_model_and_tokenizer(checkpoint_path=None):
    """Load base model and apply LoRA adapters"""
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapters if checkpoint provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading LoRA adapters from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()

    return model, tokenizer


def generate(prompt: str, model, tokenizer, max_tokens=256):
    """Generate response for prompt"""
    input_text = f"<|user|>\n{prompt}\n<|assistant|>\n"

    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with trained model")
    parser.add_argument("prompt", nargs="?", default="Write a Python function to sum two numbers")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to generate")
    args = parser.parse_args()

    # Use best checkpoint if not specified
    if not args.checkpoint:
        checkpoints = sorted(
            [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if checkpoints:
            args.checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    print("\n" + "=" * 70)
    print("GENERATING SAMPLES")
    print("=" * 70 + "\n")

    for i in range(args.num_samples):
        print(f"Sample {i + 1}:")
        print(f"Prompt: {args.prompt}")
        response = generate(args.prompt, model, tokenizer)
        print(f"Response:\n{response}\n")


if __name__ == "__main__":
    main()
