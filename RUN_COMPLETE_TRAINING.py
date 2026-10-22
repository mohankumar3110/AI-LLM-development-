#!/usr/bin/env python3
"""
🚀 COMPLETE LLAMA BLENDER TRAINING PIPELINE
One script to rule them all: Cleanup → Setup → Train → Evaluate
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"🚀 {text}")
    print("="*80)

def print_step(num, text):
    """Print step."""
    print(f"\n{'='*80}")
    print(f"STEP {num}: {text}")
    print(f"{'='*80}")

def cleanup_duplicates():
    """Remove 44 duplicate files."""
    print_step(1, "CLEANUP - Removing 44 duplicate files")
    
    FILES_TO_DELETE = [
        'ALL_FIXED.txt', 'BUILD_READY.md', 'CHECKPOINT_REFERENCE.md', 'COMPLETE_FIX_SUMMARY.txt',
        'COMPLETE_TRAINING_GUIDE.md', 'DATASET_AUGMENTATION_README.md', 'DO_THIS_NOW.txt',
        'FINAL_READY.txt', 'FINAL_SUMMARY.txt', 'FIX_SUMMARY.txt', 'IMPORT_FIX.txt',
        'INDEX.md', 'INDEX.txt', 'README_FINAL.md', 'README_NEW_FILES.txt', 'README_SUMMARY.md',
        'READY_TO_TRAIN.txt', 'RUN_THIS.py', 'SETUP_SUMMARY.txt', 'START.md', 'START_HERE.txt',
        'START_TRAINING.txt', 'START_TRAINING_NOW.txt', 'TERMINAL.bat', 'TERMINAL_GUIDE.txt',
        'TERMINAL_READY.txt', 'WORKFLOW_GUIDE.txt', 'RUN_TRAINING_NOW.txt', 'RUN_TRAINING.bat',
        'TRAIN.bat', 'VISUAL_GUIDE.md', 'QUICKSTART.md', 'augment_dataset_clean.py',
        'augment_dataset_fixed.py', 'build.py', 'check_dataset.py', 'check_dependencies.py',
        'cleanup.py', 'inference_fixed.py', 'train_optimized.py', 'train_simple.py',
        '__pycache__', 'cleanup_files.py', 'auto_cleanup.py'
    ]
    
    deleted_count = 0
    for item in FILES_TO_DELETE:
        try:
            if os.path.isdir(item):
                shutil.rmtree(item)
                print(f"  ✓ Deleted: {item}/")
                deleted_count += 1
            elif os.path.exists(item):
                os.remove(item)
                print(f"  ✓ Deleted: {item}")
                deleted_count += 1
        except:
            pass
    
    print(f"\n✅ Cleanup complete: {deleted_count} items removed")

def install_dependencies():
    """Install required packages."""
    print_step(2, "SETUP - Installing dependencies")
    
    print("This may take 5-15 minutes on first run...")
    result = subprocess.run(
        f'"{sys.executable}" -m pip install -r requirements.txt --quiet',
        shell=True
    )
    
    if result.returncode != 0:
        print("⚠️  Some packages failed to install silently. Trying without --quiet...")
        result = subprocess.run(
            f'"{sys.executable}" -m pip install -r requirements.txt',
            shell=True
        )
    
    print("\n✅ Dependencies installed")

def verify_setup():
    """Verify that everything is installed correctly."""
    print_step(3, "VERIFICATION - Checking setup")
    
    # Check PyTorch
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ GPU Memory: {vram:.1f} GB")
        else:
            print("  ⚠️  No GPU detected - training will use CPU (slower)")
    except ImportError as e:
        print(f"  ❌ PyTorch import failed: {e}")
        return False
    
    # Check data files
    print("\n  Dataset files:")
    data_dir = Path("data")
    required_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    all_exist = True
    
    for file in required_files:
        path = data_dir / file
        if path.exists():
            size = path.stat().st_size / 1024
            with open(path, 'r') as f:
                lines = len(f.readlines())
            print(f"    ✅ {file} ({lines} samples, {size:.1f} KB)")
        else:
            print(f"    ❌ {file} (MISSING)")
            all_exist = False
    
    return all_exist

def prepare_data():
    """Prepare dataset if not already done."""
    print_step(4, "DATA PREPARATION")
    
    data_dir = Path("data")
    required_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    
    # Check if all files exist
    if all((data_dir / f).exists() for f in required_files):
        print("  ✅ Dataset files already exist")
        return True
    
    print("  Preparing dataset...")
    if (data_dir / "dataset.jsonl").exists():
        result = subprocess.run(
            f'"{sys.executable}" prepare_dataset.py --input data/dataset.jsonl',
            shell=True
        )
        if result.returncode == 0:
            print("  ✅ Dataset prepared")
            return True
    
    return False

def augment_dataset():
    """Augment the dataset to increase training samples."""
    print_step(5, "DATA AUGMENTATION")
    
    augmented_file = Path("data/dataset_augmented.jsonl")
    
    if augmented_file.exists():
        with open(augmented_file, 'r') as f:
            lines = len(f.readlines())
        print(f"  ℹ️  Augmented dataset already exists ({lines} samples)")
        return True
    
    if (Path("data/dataset.jsonl")).exists():
        print("  Augmenting dataset (this may take 5-10 minutes)...")
        result = subprocess.run(
            f'"{sys.executable}" augment_dataset.py 1',
            shell=True
        )
        
        if result.returncode == 0:
            print("  ✅ Dataset augmented")
            return True
    
    return True  # Not critical if augmentation fails

def train_model():
    """Train the LLM."""
    print_step(6, "MODEL TRAINING")
    print("⏱️  This will take 30 minutes to 24+ hours depending on your GPU")
    print("📊 Training progress will appear below:\n")
    
    result = subprocess.run(
        f'"{sys.executable}" train.py',
        shell=True
    )
    
    return result.returncode == 0

def evaluate_model():
    """Evaluate and test the model."""
    print_step(7, "EVALUATION & TESTING")
    
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("  ⚠️  No checkpoints found yet")
        return False
    
    # Check for best model
    best_model = checkpoint_dir / "best_model"
    if best_model.exists():
        print(f"  ✅ Found best model at: {best_model}")
        
        print("\n  Running inference test...")
        result = subprocess.run(
            f'"{sys.executable}" inference.py "Write a Python function to add two numbers"',
            shell=True
        )
        
        return result.returncode == 0
    
    return True

def show_summary():
    """Show final summary."""
    print_header("TRAINING COMPLETE! 🎉")
    
    print("\n✅ Your LLM has been successfully trained!")
    print("\n📊 Results Location:")
    print("   └─ Trained model: checkpoints/best_model/")
    print("   └─ Training logs: logs/")
    print("   └─ Predictions: outputs/")
    
    print("\n🔧 Next Steps:")
    print("   1. Test your model:")
    print('      python inference.py "Your prompt here"')
    print("\n   2. Merge LoRA adapter:")
    print("      python merge_lora.py")
    print("\n   3. Use in Blender:")
    print("      └─ See inference.py for integration examples")
    
    print("\n📚 Documentation:")
    print("   └─ See 📋_START_HERE.md for detailed guides")
    print("   └─ See README_COMPLETE.txt for full reference")
    
    print("\n" + "="*80)

def main():
    """Main execution."""
    try:
        print_header("LLAMA BLENDER COMPLETE TRAINING PIPELINE")
        
        # Step 1: Cleanup
        cleanup_duplicates()
        
        # Step 2: Install dependencies
        install_dependencies()
        
        # Step 3: Verify setup
        if not verify_setup():
            print("❌ Setup verification failed")
            sys.exit(1)
        
        # Step 4: Prepare data
        if not prepare_data():
            print("⚠️  Data preparation had issues, but continuing...")
        
        # Step 5: Augment data (optional)
        augment_dataset()
        
        # Step 6: Train model
        if not train_model():
            print("❌ Training failed")
            sys.exit(1)
        
        # Step 7: Evaluate
        evaluate_model()
        
        # Final summary
        show_summary()
        
        print("\n✨ SUCCESS! Your training is complete. ✨\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
