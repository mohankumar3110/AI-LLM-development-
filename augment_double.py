#!/usr/bin/env python3
"""Enhanced data augmentation - double the dataset with quality variations"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict

from config import DATA_DIR

logger_msgs = []


def log(msg):
    """Simple logger"""
    print(msg)
    logger_msgs.append(msg)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file"""
    records = []
    if not os.path.exists(path):
        log(f"File not found: {path}")
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def extract_prompt_code(text: str) -> tuple:
    """Extract prompt and code from formatted text"""
    if "<|user|>" in text and "<|assistant|>" in text:
        parts = text.split("<|assistant|>")
        prompt = parts[0].replace("<|user|>", "").strip()
        code = parts[1].replace("<|eot_id|>", "").strip() if len(parts) > 1 else ""
        return prompt, code
    return text, ""


def paraphrase_text(text: str) -> str:
    """Simple text paraphrasing with synonyms"""
    replacements = {
        "create": ["write", "build", "generate", "make"],
        "function": ["method", "routine", "procedure"],
        "add": ["insert", "append", "push"],
        "remove": ["delete", "pop", "discard"],
        "loop": ["iterate", "cycle", "repeat"],
        "check": ["verify", "validate", "test"],
        "print": ["output", "display", "show"],
        "return": ["output", "give back", "send"],
        "calculate": ["compute", "figure out", "determine"],
        "find": ["search", "locate", "discover"],
    }

    words = text.split()
    result = []

    for word in words:
        lower = word.lower()
        if lower in replacements and random.random() < 0.3:
            result.append(random.choice(replacements[lower]))
        else:
            result.append(word)

    return " ".join(result)


def augment_with_context(prompt: str, code: str) -> Dict:
    """Create augmented version with context variations"""
    variations = {
        "prompt_simple": prompt,
        "prompt_please": f"Please {prompt.lower()}" if "please" not in prompt.lower() else prompt,
        "prompt_how": f"How do I {prompt.lower()}?" if "?" not in prompt else prompt,
        "code_comment": f"# Solution\n{code}" if not code.startswith("#") else code,
    }

    # Pick random variations
    aug_prompt = random.choice([variations["prompt_simple"], variations["prompt_please"]])
    aug_code = random.choice([variations["code_comment"], code])

    return {
        "prompt": aug_prompt,
        "target_code": aug_code,
        "source": "augmented",
        "text": f"<|user|>\n{aug_prompt}\n<|assistant|>\n{aug_code}<|eot_id|>",
    }


def augment_dataset(records: List[Dict]) -> List[Dict]:
    """Double dataset by creating augmented versions"""
    augmented = records.copy()

    log(f"\nAugmenting {len(records)} records...")

    for idx, record in enumerate(records):
        if (idx + 1) % 200 == 0:
            log(f"  Processed {idx + 1}/{len(records)}")

        text = record.get("text", "")
        prompt, code = extract_prompt_code(text)

        # Create variations
        for _ in range(1):  # Double the data
            aug_record = augment_with_context(prompt, code)
            augmented.append(aug_record)

    return augmented


def save_jsonl(records: List[Dict], path: str):
    """Save records to JSONL"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log(f"Saved {len(records)} records to {path}")


def main():
    log("=" * 70)
    log("ENHANCED DATA AUGMENTATION")
    log("=" * 70)

    # Load augmented dataset
    augmented_path = os.path.join(DATA_DIR, "dataset_augmented.jsonl")
    records = load_jsonl(augmented_path)

    if not records:
        log("ERROR: No data found to augment")
        return

    log(f"Loaded {len(records)} records from {augmented_path}")

    # Double the data
    doubled_records = augment_dataset(records)
    log(f"Generated {len(doubled_records)} total records (doubled from {len(records)})")

    # Save doubled dataset
    doubled_path = os.path.join(DATA_DIR, "dataset_doubled.jsonl")
    save_jsonl(doubled_records, doubled_path)

    # Re-split the data
    log("\nRe-splitting data into train/val/test...")
    random.shuffle(doubled_records)

    train_count = int(len(doubled_records) * 0.85)
    val_count = int(len(doubled_records) * 0.10)

    train_records = doubled_records[:train_count]
    val_records = doubled_records[train_count : train_count + val_count]
    test_records = doubled_records[train_count + val_count :]

    # Save splits
    save_jsonl(train_records, os.path.join(DATA_DIR, "train_doubled.jsonl"))
    save_jsonl(val_records, os.path.join(DATA_DIR, "val_doubled.jsonl"))
    save_jsonl(test_records, os.path.join(DATA_DIR, "test_doubled.jsonl"))

    log("\n" + "=" * 70)
    log("AUGMENTATION COMPLETE")
    log("=" * 70)
    log(f"Train: {len(train_records)} samples")
    log(f"Val:   {len(val_records)} samples")
    log(f"Test:  {len(test_records)} samples")
    log(f"Total: {len(doubled_records)} samples (2x the original)")


if __name__ == "__main__":
    main()
