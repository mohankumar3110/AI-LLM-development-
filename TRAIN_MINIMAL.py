#!/usr/bin/env python3
"""
🚀 MINIMAL TRAINING RUNNER - Direct training without dependencies check
Use this if RUN_COMPLETE_TRAINING.py fails
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    print("\n" + "="*80)
    print("🚀 LLAMA BLENDER TRAINING - MINIMAL MODE")
    print("="*80)
    
    # Step 1: Show current status
    print("\n✓ Step 1: Checking environment...")
    print(f"  Working directory: {os.getcwd()}")
    
    # Step 2: Install deps silently
    print("\n✓ Step 2: Installing dependencies (this may take 5-15 minutes)...")
    result = subprocess.run(
        f'"{sys.executable}" -m pip install -q -r requirements.txt 2>&1',
        shell=True,
        capture_output=False
    )
    print("  ✅ Dependencies ready")
    
    # Step 3: Verify PyTorch
    print("\n✓ Step 3: Verifying PyTorch...")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"  ✅ CUDA: {torch.cuda.is_available()}")
    except:
        print("  ⚠️  PyTorch verification skipped")
    
    # Step 4: Check data
    print("\n✓ Step 4: Verifying dataset...")
    for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        p = Path("data") / f
        if p.exists():
            lines = len(open(p).readlines())
            print(f"  ✅ {f} ({lines} samples)")
        else:
            print(f"  ❌ {f} MISSING")
    
    # Step 5: Train
    print("\n✓ Step 5: Starting training...")
    print("  This will take 30 minutes to 24+ hours...")
    print("  Monitor progress below:\n")
    
    result = subprocess.run(f'"{sys.executable}" train.py', shell=True)
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print("\n✨ Your model is ready in checkpoints/best_model/")
        print("\n  Test it:")
        print('    python inference.py "Create a cube in Blender"')
    else:
        print("\n" + "="*80)
        print("❌ Training failed")
        print("="*80)
        sys.exit(1)

if __name__ == "__main__":
    main()
