
import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path
 
try:
    rich_console = importlib.import_module("rich.console")
    rich_table = importlib.import_module("rich.table")
    Console = rich_console.Console
    Table = rich_table.Table
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

    class Console:
        def print(self, *args, **kwargs):
            print(*args, **kwargs)

        def rule(self, title=""):
            if title:
                print("=" * len(title))
                print(title)
                print("=" * len(title))
            else:
                print("=" * 40)

    class Table:
        def __init__(self, title=None, show_header=True, header_style=None):
            self.title = title
            self.rows = []

        def add_column(self, *args, **kwargs):
            pass

        def add_row(self, *args, **kwargs):
            self.rows.append(tuple(str(arg) for arg in args))

        def __str__(self):
            output = []
            if self.title:
                output.append(self.title)
            for row in self.rows:
                output.append("  ".join(row))
            return "\n".join(output)

# ── project imports ────────────────────────────────────────────────────────────
from config import (
    RAW_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    USER_TOKEN, ASSISTANT_TOKEN,
    DATA_DIR,
)
 
console = Console()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Chat-template formatter
# ──────────────────────────────────────────────────────────────────────────────
 
def format_example(prompt: str, target_code: str) -> str:
    """
    Convert a prompt/target_code pair into the Llama 3.2 Instruct
    chat-template format:
 
        <|user|>
        {prompt}
        <|assistant|>
        {target_code}
 
    The model is trained to predict only the assistant turn (tokens
    after <|assistant|>).  The loss-masking step in train.py takes care
    of that; here we just build the full string.
    """
    return (
        f"{USER_TOKEN}\n"
        f"{prompt.strip()}\n"
        f"{ASSISTANT_TOKEN}\n"
        f"{target_code.strip()}"
    )
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Load & validate
# ──────────────────────────────────────────────────────────────────────────────
 
def load_jsonl(path: str) -> list[dict]:
    records = []
    errors  = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                console.print(f"[yellow]⚠  Line {line_no}: JSON error — {exc}[/yellow]")
                errors += 1
                continue
 
            if "prompt" not in obj or "target_code" not in obj:
                console.print(
                    f"[yellow]⚠  Line {line_no}: missing 'prompt' or 'target_code' key — skipped[/yellow]"
                )
                errors += 1
                continue
 
            records.append(obj)
 
    console.print(
        f"[green]✔  Loaded {len(records):,} valid records "
        f"({errors} skipped)[/green]"
    )
    return records
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────────────────────────────────────
 
def split_dataset(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
    seed: int          = 42,
) -> tuple[list, list, list]:
    random.seed(seed)
    data = records.copy()
    random.shuffle(data)
 
    n       = len(data)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
 
    train = data[:n_train]
    val   = data[n_train : n_train + n_val]
    test  = data[n_train + n_val :]
    return train, val, test
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Write processed JSONL
# ──────────────────────────────────────────────────────────────────────────────
 
def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            text = format_example(rec["prompt"], rec["target_code"])
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    console.print(f"  → wrote {len(records):,} examples to [cyan]{path}[/cyan]")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Preview helper
# ──────────────────────────────────────────────────────────────────────────────
 
def preview(records: list[dict], n: int = 2) -> None:
    console.rule("[bold]Dataset Preview[/bold]")
    for i, rec in enumerate(records[:n]):
        console.print(f"\n[bold]Example {i + 1}[/bold]")
        console.print(format_example(rec["prompt"], rec["target_code"]))
    console.rule()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Stats table
# ──────────────────────────────────────────────────────────────────────────────
 
def print_stats(train, val, test) -> None:
    table = Table(title="Dataset Split Summary", show_header=True, header_style="bold magenta")
    table.add_column("Split",   style="cyan",  justify="right")
    table.add_column("Count",   style="green", justify="right")
    table.add_column("% Total", style="yellow", justify="right")
    total = len(train) + len(val) + len(test)
    for name, subset in [("Train", train), ("Val", val), ("Test", test)]:
        pct = 100 * len(subset) / total if total else 0
        table.add_row(name, f"{len(subset):,}", f"{pct:.1f}%")
    table.add_row("[bold]Total[/bold]", f"[bold]{total:,}[/bold]", "100.0%")
    console.print(table)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and split the JSONL dataset.")
    parser.add_argument("--input",   default=RAW_DATA_PATH, help="Path to raw .jsonl file")
    parser.add_argument("--preview", type=int, default=2,   help="Number of examples to preview (0 = none)")
    parser.add_argument("--seed",    type=int, default=42,  help="Random seed for shuffle")
    args = parser.parse_args()
 
    console.rule("[bold blue]Dataset Preparation[/bold blue]")
 
    if not os.path.exists(args.input):
        console.print(
            f"[red]✘  Dataset not found at '{args.input}'.\n"
            f"   Place your JSONL file there and re-run.[/red]"
        )
        sys.exit(1)
 
    records = load_jsonl(args.input)
 
    if args.preview > 0:
        preview(records, args.preview)
 
    train, val, test = split_dataset(records, seed=args.seed)
    print_stats(train, val, test)
 
    console.print("\n[bold]Writing splits …[/bold]")
    write_jsonl(train, TRAIN_DATA_PATH)
    write_jsonl(val,   VAL_DATA_PATH)
    write_jsonl(test,  TEST_DATA_PATH)
 
    console.rule("[bold green]Done — dataset ready for training[/bold green]")
 
 
if __name__ == "__main__":
    main()
 