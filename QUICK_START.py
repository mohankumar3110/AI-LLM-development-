#!/usr/bin/env python3
"""
🚀 LLAMA BLENDER TRAINING SYSTEM - QUICK START
Simplest possible entry point - just run this file!
"""

import os
import sys

def main():
    print("\n" + "="*80)
    print("🚀 LLAMA BLENDER TRAINING SYSTEM")
    print("="*80)
    
    print("\nChoose an option:")
    print("  1. Run COMPLETE training (recommended)")
    print("  2. Cleanup duplicates only")
    print("  3. Install dependencies only")
    print("  4. Check GPU status")
    print("  5. Quick data check")
    print("  6. Run training only")
    print("  7. Test inference")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting complete training pipeline...")
        os.system("python RUN_COMPLETE_TRAINING.py")
    
    elif choice == "2":
        print("\n🧹 Cleaning up duplicates...")
        os.system("python auto_cleanup.py")
    
    elif choice == "3":
        print("\n📦 Installing dependencies...")
        os.system(f'"{sys.executable}" -m pip install -r requirements.txt')
    
    elif choice == "4":
        print("\n🖥️  Checking GPU...")
        try:
            import torch
            print(f"  PyTorch: {torch.__version__}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"  VRAM: {vram:.1f} GB")
        except ImportError:
            print("  ❌ PyTorch not installed")
    
    elif choice == "5":
        print("\n📊 Checking data...")
        from pathlib import Path
        data_dir = Path("data")
        for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            p = data_dir / f
            if p.exists():
                with open(p) as file:
                    lines = len(file.readlines())
                size = p.stat().st_size / 1024
                print(f"  {f}: {lines} samples ({size:.1f} KB)")
            else:
                print(f"   {f}: MISSING")
    
    elif choice == "6":
        print("\n🎯 Running training...")
        os.system(f'"{sys.executable}" train.py')
    
    elif choice == "7":
        print("\n Testing inference...")
        prompt = input("Enter test prompt: ").strip()
        if prompt:
            os.system(f'"{sys.executable}" inference.py "{prompt}"')
    
    else:
        print("  Invalid choice")

if __name__ == "__main__":
    main()
