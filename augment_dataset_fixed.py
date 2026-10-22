#!/usr/bin/env python3
"""
Data augmentation script to double the dataset using paraphrasing and variations.
Doubles the dataset by creating variations of existing examples.
"""

import json
import random
import os
import sys
import importlib
from pathlib import Path
from typing import List, Dict

try:
    Console = getattr(importlib.import_module("rich.console"), "Console")
    Table = getattr(importlib.import_module("rich.table"), "Table")
    track = getattr(importlib.import_module("rich.progress"), "track")
except ImportError:

    class Console:
        def rule(self, text):
            print(f"\n{'='*70}\n{text}\n{'='*70}\n")
        
        def print(self, *args, **kwargs):
            print(*args)
    
    class Table:
        def __init__(self, title=None, show_header=True, header_style=None):
            self.rows = []
            self.columns = []
        
        def add_column(self, name, *args, **kwargs):
            self.columns.append(name)
        
        def add_row(self, *args, **kwargs):
            self.rows.append(args)
        
        def __str__(self):
            result = []
            if self.columns:
                result.append(" | ".join(self.columns))
            for row in self.rows:
                result.append(" | ".join(str(item) for item in row))
            return "\n".join(result)
    
    def track(iterable, description=None):
        return iterable

from config import RAW_DATA_PATH, DATA_DIR

console = Console()


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def paraphrase_simple(text: str) -> str:
    """Simple paraphrasing using word replacements and minor edits."""
    replacements = {
        "write": "create",
        "function": "method",
        "create": "write",
        "build": "construct",
        "make": "generate",
        "generate": "produce",
        "return": "output",
        "output": "return",
        "calculate": "compute",
        "compute": "calculate",
        "check": "verify",
        "verify": "check",
        "loop": "iterate",
        "iterate": "loop",
        "find": "search",
        "search": "find",
        "sort": "order",
        "order": "sort",
    }
    
    words = text.lower().split()
    new_words = []
    
    for word in words:
        clean_word = word.rstrip('.,;:!?')
        suffix = word[len(clean_word):]
        
        if clean_word in replacements and random.random() < 0.3:
            new_words.append(replacements[clean_word] + suffix)
        else:
            new_words.append(word)
    
    return " ".join(new_words)


def augment_prompt(prompt: str) -> List[str]:
    """Generate variations of a prompt."""
    variations = [prompt]
    
    variations.append(paraphrase_simple(prompt))
    
    if "please" not in prompt.lower():
        variations.append(f"Please {prompt.lower()}")
    
    if prompt.strip().endswith("?"):
        variations.append(prompt.strip().rstrip("?"))
    
    return variations[:3]


def augment_code(code: str) -> List[str]:
    """Generate variations of code through formatting and comment changes."""
    variations = [code]
    
    lines = code.split('\n')
    if len(lines) > 1 and not lines[0].strip().startswith('#'):
        variations.append(f"# Implementation\n{code}")
    
    if len(code) < 500:
        variations.append(code)
    
    return variations[:2]


def augment_dataset(records: List[Dict], factor: int = 1) -> List[Dict]:
    """Augment dataset by creating variations."""
    augmented = records.copy()
    
    for _ in range(factor):
        for record in track(records, description="Augmenting records..."):
            prompt_variations = augment_prompt(record.get("prompt", ""))
            code_variations = augment_code(record.get("target_code", ""))
            
            new_record = {
                "prompt": random.choice(prompt_variations),
                "target_code": random.choice(code_variations),
                "source": "augmented",
            }
            augmented.append(new_record)
    
    return augmented


def save_jsonl(records: List[Dict], path: str) -> None:
    """Save records to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_stats(original_count: int, augmented_count: int) -> None:
    """Print augmentation statistics."""
    table = Table(title="Dataset Augmentation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Count")
    
    table.add_row("Original Records", f"{original_count:,}")
    table.add_row("Augmented Records", f"{augmented_count:,}")
    table.add_row("Total Records", f"{original_count + augmented_count:,}")
    table.add_row("Growth Factor", f"{(original_count + augmented_count) / original_count:.1f}x")
    
    console.print(table)


def main() -> None:
    parser_args = sys.argv[1:] if len(sys.argv) > 1 else []
    factor = int(parser_args[0]) if parser_args else 1
    
    console.rule("[bold blue]Dataset Augmentation[/bold blue]")
    
    if not os.path.exists(RAW_DATA_PATH):
        console.print(f"[red]✘ Dataset not found at {RAW_DATA_PATH}[/red]")
        sys.exit(1)
    
    console.print(f"[yellow]Loading dataset from {RAW_DATA_PATH}...[/yellow]")
    original_records = load_jsonl(RAW_DATA_PATH)
    console.print(f"[green]✔ Loaded {len(original_records):,} records[/green]")
    
    console.print(f"\n[yellow]Augmenting dataset (factor={factor})...[/yellow]")
    augmented_records = augment_dataset(original_records, factor=factor)
    
    output_path = os.path.join(DATA_DIR, "dataset_augmented.jsonl")
    console.print(f"\n[yellow]Saving augmented dataset to {output_path}...[/yellow]")
    save_jsonl(augmented_records, output_path)
    
    print_stats(len(original_records), len(augmented_records) - len(original_records))
    console.rule("[bold green]✔ Augmentation complete![/bold green]")
    console.print(f"\n[cyan]Next step:[/cyan] Run: [bold]python prepare_dataset.py --input {output_path}[/bold]")


if __name__ == "__main__":
    main()
