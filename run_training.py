#!/usr/bin/env python3
"""
Complete automated training script - Does everything for you!
"""

import os
import sys
import subprocess

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAW_DATA_PATH, DATA_DIR, TRAIN_DATA_PATH

print("=" * 80)
print("  LLAMA 3.2 AUTOMATED TRAINING - STARTING")
print("=" * 80)

# Step 1: Check if data exists
print("\n[1/4] CHECKING DATA...")
if not os.path.exists(RAW_DATA_PATH):
    print(f"❌ ERROR: Dataset not found at {RAW_DATA_PATH}")
    sys.exit(1)

import json
with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
    count = sum(1 for _ in f if _.strip())
print(f"✅ Found {count} samples in {RAW_DATA_PATH}")

# Step 2: Check if data is already prepared
if os.path.exists(TRAIN_DATA_PATH):
    print("\n[2/4] USING EXISTING PREPARED DATA...")
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_count = sum(1 for _ in f if _.strip())
    print(f"✅ Found {train_count} training samples")
else:
    print("\n[2/4] PREPARING DATA...")
    print("⚠️  Data not prepared yet. Preparing now...")
    
    # Prepare the dataset
    result = subprocess.run(
        [sys.executable, "prepare_dataset.py", "--input", RAW_DATA_PATH, "--preview", "0"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False
    )
    
    if result.returncode != 0:
        print("❌ Dataset preparation failed!")
        sys.exit(1)
    
    if not os.path.exists(TRAIN_DATA_PATH):
        print("❌ Prepared data not found!")
        sys.exit(1)
    
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_count = sum(1 for _ in f if _.strip())
    print(f"✅ Prepared {train_count} training samples")

# Step 3: Train the model
print("\n[3/4] STARTING TRAINING...")
print("=" * 80)
print("This will take 30 minutes to several hours depending on your GPU.")
print("If you see 'CUDA out of memory', edit config.py and reduce batch size.")
print("=" * 80 + "\n")

result = subprocess.run(
    [sys.executable, "train.py"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
    capture_output=False
)

if result.returncode != 0:
    print("\n❌ Training failed!")
    print("Check logs/train.log for details")
    sys.exit(1)

# Step 4: Check results
print("\n[4/4] CHECKING RESULTS...")
import glob
checkpoints = glob.glob(os.path.join(os.path.dirname(__file__), "checkpoints", "checkpoint-*"))
if checkpoints:
    print(f"✅ Training completed! Found {len(checkpoints)} checkpoints")
    latest = sorted(checkpoints)[-1]
    print(f"✅ Latest checkpoint: {os.path.basename(latest)}")
else:
    print("⚠️  No checkpoints found - check training output above")

print("\n" + "=" * 80)
print("  ✅ TRAINING COMPLETE!")
print("=" * 80)
print(f"\nYour trained model is here:")
print(f"  → checkpoints/checkpoint-{len(checkpoints)}/")
print(f"\nTo test it, run:")
print(f"  python inference.py \"Your prompt here\"")
print("=" * 80)
