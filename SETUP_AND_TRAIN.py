#!/usr/bin/env python3
"""
🚀 COMPLETE SETUP & TRAINING PIPELINE
Run this once to install dependencies, verify setup, and start training.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"✓ {description}")
    print(f"{'='*70}")
    print(f"$ {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ Completed: {description}\n")

def main():
    print("\n" + "="*70)
    print("🚀 LLAMA 3.2 1B LORA FINE-TUNING PIPELINE")
    print("="*70)
    
    # Get project directory
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(PROJECT_DIR)
    
    print(f"\n📁 Project directory: {PROJECT_DIR}")
    
    # Step 1: Install dependencies
    run_command(
        f'"{sys.executable}" -m pip install -r requirements.txt --quiet',
        "STEP 1: Installing dependencies (this may take 5-10 minutes)"
    )
    
    # Step 2: Verify setup
    print("\n" + "="*70)
    print("✓ STEP 2: Verifying setup")
    print("="*70)
    
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚠️  No GPU detected - training will be slow on CPU")
    except ImportError as e:
        print(f"  ❌ PyTorch import failed: {e}")
        sys.exit(1)
    
    # Verify datasets
    data_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    data_dir = Path(PROJECT_DIR) / "data"
    print(f"\n  Dataset files:")
    for file in data_files:
        path = data_dir / file
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"    ✅ {file} ({size:.1f} KB)")
        else:
            print(f"    ❌ {file} (MISSING!)")
    
    # Step 3: Start training
    print("\n" + "="*70)
    print("✓ STEP 3: Starting training...")
    print("="*70)
    print("\n⏱️  This will take 30 minutes to 12+ hours depending on your GPU")
    print("📊 Training progress will appear below:\n")
    
    run_command(
        f'"{sys.executable}" train.py',
        "STEP 3: Training the model"
    )
    
    # Step 4: Success!
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print("\n✅ Your trained model is in: checkpoints/")
    print("\n📝 Next steps:")
    print("  1. Test your model:")
    print(f"     python inference.py \"Write a Python function\"")
    print("\n  2. Merge LoRA adapter into base model:")
    print(f"     python merge_lora.py")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
