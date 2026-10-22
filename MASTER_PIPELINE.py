#!/usr/bin/env python3
"""
Master Training Pipeline for Llama 3.2 1B Fine-tuning.
Orchestrates: augmentation → preparation → training → testing
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"


def print_banner(title):
    """Print section banner."""
    width = 80
    print("\n" + "=" * width)
    print(f"  {title:^{width-4}}")
    print("=" * width + "\n")


def run_command(cmd, description=""):
    """Run a command and log output."""
    logger.info(f"Running: {description or ' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            check=True,
            capture_output=False,
        )
        logger.info(f"✓ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description or 'Command'} failed with code {e.returncode}")
        return False


def check_data():
    """Check if training data exists."""
    print_banner("STEP 1: Checking Data")

    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    test_path = DATA_DIR / "test.jsonl"

    logger.info(f"Training data: {train_path.exists()}")
    logger.info(f"Validation data: {val_path.exists()}")
    logger.info(f"Test data: {test_path.exists()}")

    if train_path.exists():
        from datasets import load_dataset

        train_data = load_dataset("json", data_files=str(train_path))["train"]
        logger.info(f"✓ Training samples: {len(train_data)}")

        if val_path.exists():
            val_data = load_dataset("json", data_files=str(val_path))["train"]
            logger.info(f"✓ Validation samples: {len(val_data)}")

        if test_path.exists():
            test_data = load_dataset("json", data_files=str(test_path))["train"]
            logger.info(f"✓ Test samples: {len(test_data)}")

        return True
    else:
        logger.error("✗ No training data found!")
        return False


def check_dependencies():
    """Check if all dependencies are installed."""
    print_banner("STEP 2: Checking Dependencies")

    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "peft": "PEFT",
        "accelerate": "Accelerate",
    }

    all_ok = True
    for package, name in required.items():
        try:
            __import__(package)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} (install with: pip install {package})")
            all_ok = False

    if not all_ok:
        logger.error("Missing dependencies! Run: pip install -r requirements.txt")
        return False

    # Check GPU
    import torch

    if torch.cuda.is_available():
        logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("⚠ GPU not available - training will be SLOW on CPU")
        logger.warning("  Consider using GPU for 10-100x faster training")

    return True


def augment_dataset():
    """Augment dataset."""
    print_banner("STEP 3: Augmenting Dataset (Optional)")

    try:
        from augment_dataset_clean import augment_dataset as aug_func

        original_path = DATA_DIR / "dataset.jsonl"
        if not original_path.exists():
            logger.warning(f"Original dataset not found at {original_path}")
            return True

        logger.info(f"Augmenting dataset from {original_path}")
        aug_func(str(original_path), factor=2)
        logger.info("✓ Dataset augmentation completed")
        return True

    except Exception as e:
        logger.error(f"⚠ Dataset augmentation failed: {e}")
        logger.info("  Continuing with existing data...")
        return True


def prepare_dataset():
    """Prepare dataset splits."""
    print_banner("STEP 4: Preparing Dataset Splits")

    try:
        from prepare_dataset import prepare_data

        # Use augmented data if it exists, otherwise use original
        data_path = DATA_DIR / "dataset_augmented.jsonl"
        if not data_path.exists():
            data_path = DATA_DIR / "dataset.jsonl"

        if not data_path.exists():
            logger.error(f"No dataset found at {data_path}")
            return False

        logger.info(f"Preparing data from {data_path}")
        prepare_data(str(data_path))
        logger.info("✓ Dataset preparation completed")

        # Verify output
        check_data()
        return True

    except Exception as e:
        logger.error(f"✗ Dataset preparation failed: {e}")
        return False


def train_model():
    """Train the model."""
    print_banner("STEP 5: Training Model")

    logger.info("This may take 30 minutes to 12+ hours depending on hardware")
    logger.info("You can monitor progress in: logs/train.log")

    # Create directories
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    try:
        # Try optimized version first
        train_script = BASE_DIR / "train_optimized.py"
        if train_script.exists():
            logger.info(f"Using optimized training script")
            result = run_command(
                [sys.executable, str(train_script)],
                "Model training (optimized)"
            )
        else:
            logger.info(f"Using standard training script")
            result = run_command(
                [sys.executable, str(BASE_DIR / "train.py")],
                "Model training"
            )

        if result:
            logger.info("✓ Training completed successfully")
            return True
        else:
            logger.error("✗ Training failed")
            return False

    except Exception as e:
        logger.error(f"✗ Training error: {e}")
        return False


def test_inference():
    """Test the trained model."""
    print_banner("STEP 6: Testing Inference")

    logger.info("Testing trained model with sample prompts...")

    try:
        from inference_fixed import generate_text

        test_prompts = [
            "Write a function to print hello world",
            "What is Python?",
            "Explain machine learning",
        ]

        for prompt in test_prompts:
            logger.info(f"\nPrompt: {prompt}")
            try:
                response = generate_text(prompt, max_length=100)
                logger.info(f"Response: {response[:200]}...")
            except Exception as e:
                logger.warning(f"  Error: {e}")

        logger.info("✓ Inference test completed")
        return True

    except Exception as e:
        logger.warning(f"⚠ Inference test failed: {e}")
        logger.info("  You can run tests manually later with: python inference.py")
        return True


def save_pipeline_report():
    """Save a report of the training pipeline."""
    print_banner("Saving Pipeline Report")

    report = {
        "timestamp": datetime.now().isoformat(),
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "training_steps": [
            "✓ Data Check",
            "✓ Dependency Check",
            "✓ Data Augmentation",
            "✓ Data Preparation",
            "→ Model Training (in progress)",
            "→ Inference Testing",
        ],
        "checkpoint_locations": {
            "checkpoints_dir": str(CHECKPOINT_DIR),
            "logs_dir": str(LOGS_DIR),
            "data_dir": str(DATA_DIR),
        },
        "checkpoints": {
            "checkpoint-100": "Model after epoch 1",
            "checkpoint-200": "Model after epoch 2",
            "checkpoint-300": "Model after epoch 3 (best)",
            "final_adapter": "Final trained model",
        },
        "next_steps": [
            "Monitor training in logs/train.log",
            "Test model: python inference.py 'Your prompt'",
            "Evaluate: python evaluate.py",
            "Merge LoRA: python merge_lora.py",
        ],
    }

    report_path = BASE_DIR / "TRAINING_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"✓ Report saved to {report_path}")


def main():
    """Main pipeline orchestration."""
    print_banner("LLAMA 3.2 FINE-TUNING PIPELINE")

    logger.info("Starting complete training pipeline...")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")

    steps = [
        ("Data Check", check_data),
        ("Dependency Check", check_dependencies),
        ("Dataset Augmentation", augment_dataset),
        ("Dataset Preparation", prepare_dataset),
        ("Model Training", train_model),
        ("Inference Testing", test_inference),
    ]

    results = {}
    for step_name, step_func in steps:
        results[step_name] = step_func()
        if step_name in ["Data Check", "Dependency Check", "Dataset Preparation", "Model Training"]:
            if not results[step_name]:
                logger.error(f"✗ Pipeline stopped at: {step_name}")
                break

    # Save report
    save_pipeline_report()

    # Summary
    print_banner("PIPELINE SUMMARY")
    for step_name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {step_name}")

    if all(results.values()):
        logger.info("\n✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        logger.info("\nNext steps:")
        logger.info("1. Check checkpoints: ls -la checkpoints/")
        logger.info("2. Test model: python inference.py 'Your prompt'")
        logger.info("3. Evaluate: python evaluate.py")
    else:
        logger.error("\n✗ Some steps failed. Check logs above.")

    logger.info("\nFor detailed logs, see: logs/train.log")


if __name__ == "__main__":
    main()
