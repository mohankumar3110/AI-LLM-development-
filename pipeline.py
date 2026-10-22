#!/usr/bin/env python3
"""
Complete pipeline script for dataset augmentation, preparation, and training.
Usage:
    python pipeline.py --action augment      # Augment dataset (1x doubling)
    python pipeline.py --action augment --factor 2  # 3x total (original + 2x augmented)
    python pipeline.py --action prepare     # Prepare augmented dataset for training
    python pipeline.py --action train       # Start training
    python pipeline.py --action all         # Do everything
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

try:
    from rich.console import Console  # type: ignore
    from rich.panel import Panel  # type: ignore
except ImportError:
    Console = None
    Panel = None

class SimpleConsole:
    def print(self, *args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(arg) for arg in args)
        sys.stdout.write(text + end)

    def rule(self, text):
        line = "=" * len(text)
        self.print(line)
        self.print(text)
        self.print(line)

console = Console() if Console else SimpleConsole()

from config import RAW_DATA_PATH, DATA_DIR


def run_augmentation(factor: int = 1) -> str:
    """Run data augmentation."""
    console.print(Panel("🔄 [bold cyan]Running Data Augmentation[/bold cyan]", title="Step 1"))

    result = subprocess.run(
        [sys.executable, "augment_dataset.py", str(factor)],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False
    )
    
    if result.returncode != 0:
        console.print("[red]✘ Augmentation failed[/red]")
        return None
    
    augmented_path = os.path.join(DATA_DIR, "dataset_augmented.jsonl")
    console.print(f"[green]✔ Augmentation complete: {augmented_path}[/green]\n")
    return augmented_path


def run_preparation(input_path: str = None) -> None:
    """Run dataset preparation."""
    console.print(Panel("📊 [bold cyan]Running Dataset Preparation[/bold cyan]", title="Step 2"))
    
    if input_path is None:
        input_path = os.path.join(DATA_DIR, "dataset_augmented.jsonl")
    
    if not os.path.exists(input_path):
        console.print(f"[red]✘ Input file not found: {input_path}[/red]")
        return False
    
    result = subprocess.run(
        [sys.executable, "prepare_dataset.py", "--input", input_path, "--preview", "2"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False
    )
    
    if result.returncode != 0:
        console.print("[red]✘ Dataset preparation failed[/red]")
        return False
    
    console.print("[green]✔ Dataset preparation complete\n[/green]")
    return True


def run_training() -> None:
    """Run model training."""
    console.print(Panel("🚀 [bold cyan]Starting Model Training[/bold cyan]", title="Step 3"))
    console.print("[yellow]This may take a while depending on your hardware...[/yellow]\n")
    
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False
    )
    
    if result.returncode != 0:
        console.print("[red]✘ Training failed[/red]")
        return False
    
    console.print("[green]✔ Training complete\n[/green]")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline for dataset augmentation, preparation, and training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --action augment              # Double dataset (1x augmentation)
  python pipeline.py --action augment --factor 2   # Triple dataset (original + 2x augmented)
  python pipeline.py --action prepare              # Prepare augmented dataset
  python pipeline.py --action train                # Train the model
  python pipeline.py --action all                  # Do everything
        """
    )
    
    parser.add_argument(
        "--action",
        choices=["augment", "prepare", "train", "all"],
        default="all",
        help="Action to perform (default: all)"
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=1,
        help="Augmentation factor (1 = 2x dataset, 2 = 3x, etc.)"
    )
    
    args = parser.parse_args()
    
    console.rule("[bold blue]Llama 3.2 Fine-tuning Pipeline[/bold blue]")
    
    augmented_path = None
    
    if args.action in ["augment", "all"]:
        augmented_path = run_augmentation(args.factor)
        if augmented_path is None:
            sys.exit(1)
    
    if args.action in ["prepare", "all"]:
        success = run_preparation(augmented_path)
        if not success:
            sys.exit(1)
    
    if args.action in ["train", "all"]:
        success = run_training()
        if not success:
            sys.exit(1)
    
    console.rule("[bold green]✔ Pipeline Complete![/bold green]")
    console.print("""
[cyan]Your fine-tuned model is ready! Next steps:[/cyan]
  1. Evaluate: [bold]python evaluate.py[/bold]
  2. Test: [bold]python inference.py "Your prompt here"[/bold]
  3. Merge LoRA weights: [bold]python merge_lora.py[/bold]
    """)


if __name__ == "__main__":
    main()
