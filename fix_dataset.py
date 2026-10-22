#!/usr/bin/env python3
"""Fix dataset by adding text field to all records"""

import json
import os
from config import DATA_DIR


def load_jsonl(path: str):
    """Load JSONL"""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    continue
    return records


def fix_record(record):
    """Add text field if missing"""
    if not record.get("text"):
        prompt = record.get("prompt", "")
        code = record.get("target_code", "")
        record["text"] = f"<|user|>\n{prompt}\n<|assistant|>\n{code}<|eot_id|>"
    return record


def save_jsonl(records, path):
    """Save JSONL"""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# Fix the doubled datasets
for fname in ["train_doubled.jsonl", "val_doubled.jsonl", "test_doubled.jsonl"]:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        print(f"Fixing {fname}...")
        records = load_jsonl(fpath)
        records = [fix_record(r) for r in records]
        save_jsonl(records, fpath)
        print(f"  Saved {len(records)} records")

print("Done!")
