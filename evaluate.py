import argparse
import ast
import json
import logging
import math
import os
import sys
from typing import Any

import torch # pyright: ignore[reportMissingImports]
from datasets import load_dataset # pyright: ignore[reportMissingImports]
from rouge_score import rouge_scorer # pyright: ignore[reportMissingImports]

try:
    from tqdm.auto import tqdm  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
except (ImportError, ModuleNotFoundError):
    def tqdm(iterable, **kwargs):
        return iterable
 
from config import (
    MODEL_ID,
    MERGED_DIR,
    CHECKPOINT_DIR,
    TEST_DATA_PATH,
    OUTPUT_DIR,
    INFERENCE_CONFIG,
    MAX_SEQ_LENGTH,
)
from inference import load_model_and_tokenizer, generate, build_prompt
 
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
 
# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
 
def is_valid_python(code: str) -> bool:
    """Return True if 'code' is syntactically valid Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
 
 
def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    """Average perplexity over a list of fully-formatted texts."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
 
    for text in tqdm(texts, desc="Perplexity"):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(model.device)
 
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        nll  = out.loss.item()
        toks = enc["input_ids"].shape[1]
        total_nll    += nll * toks
        total_tokens += toks
 
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
 
def evaluate(
    model,
    tokenizer,
    examples: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
 
    scorer       = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores: list[float] = []
    exact_matches             = 0
    syntax_ok                 = 0
    per_example_results       = []
 
    for ex in tqdm(examples, desc="Generating"):
        # The test JSONL has {"text": "<|user|>\n...\n<|assistant|>\nCODE"}
        text  = ex["text"]
        # Extract the reference code (everything after <|assistant|>\n)
        parts = text.split("<|assistant|>\n", 1)
        if len(parts) < 2:
            continue
        prompt_part = parts[0].replace("<|user|>\n", "").strip()
        reference   = parts[1].strip()
 
        # Generate
        prediction = generate(
            model, tokenizer, prompt_part,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
        )
 
        # Metrics
        rl    = scorer.score(reference, prediction)["rougeL"].fmeasure
        rouge_scores.append(rl)
        exact = prediction.strip() == reference.strip()
        exact_matches += int(exact)
        valid = is_valid_python(prediction)
        syntax_ok += int(valid)
 
        per_example_results.append({
            "prompt":     prompt_part,
            "reference":  reference,
            "prediction": prediction,
            "rouge_l":    round(rl, 4),
            "exact":      exact,
            "syntax_ok":  valid,
        })
 
    n = len(per_example_results)
    return {
        "num_samples":   n,
        "rouge_l_mean":  round(sum(rouge_scores) / n, 4) if n else 0,
        "exact_match":   round(exact_matches / n * 100, 2) if n else 0,
        "syntax_ok_pct": round(syntax_ok / n * 100, 2) if n else 0,
        "per_example":   per_example_results,
    }
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Print summary
# ──────────────────────────────────────────────────────────────────────────────
 
def print_summary(results: dict, ppl: float | None) -> None:
    sep = "─" * 45
    print(f"\n{sep}")
    print("  Evaluation Results")
    print(sep)
    print(f"  Samples evaluated : {results['num_samples']}")
    if ppl is not None:
        print(f"  Perplexity        : {ppl:.3f}")
    print(f"  ROUGE-L (mean)    : {results['rouge_l_mean']:.4f}")
    print(f"  Exact match       : {results['exact_match']:.2f}%")
    print(f"  Syntax-valid      : {results['syntax_ok_pct']:.2f}%")
    print(sep)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned model.")
    parser.add_argument("--model_path",   default=MERGED_DIR,  help="Merged model path")
    parser.add_argument("--adapter",      default=None,        help="LoRA adapter path (optional)")
    parser.add_argument("--base_model",   default=MODEL_ID,    help="Base model when using --adapter")
    parser.add_argument("--test_file",    default=TEST_DATA_PATH, help="Test JSONL file")
    parser.add_argument("--num_samples",  type=int, default=None, help="Max samples to evaluate (default: all)")
    parser.add_argument("--quantize",     action="store_true",  help="4-bit inference")
    parser.add_argument("--perplexity",   action="store_true",  help="Also compute perplexity (slow)")
    parser.add_argument("--temperature",  type=float, default=0.1)
    parser.add_argument("--top_p",        type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_json",  default=None,         help="Save full results to JSON")
    args = parser.parse_args()
 
    # ── Validate inputs ──────────────────────────────────────────────────
    if not os.path.exists(args.test_file):
        logger.error(f"Test file not found: {args.test_file}\nRun: python prepare_dataset.py")
        sys.exit(1)
 
    model_src = args.base_model if args.adapter else args.model_path
    if not args.adapter and not os.path.isdir(model_src):
        logger.error(f"Model not found: {model_src}\nRun: python merge_lora.py")
        sys.exit(1)
 
    # ── Load ─────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_src, args.adapter, args.quantize)
 
    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files={"test": args.test_file}, split="test")
    examples = list(dataset)
    if args.num_samples:
        examples = examples[: args.num_samples]
    logger.info(f"Evaluating on {len(examples)} examples …")
 
    # ── Perplexity ────────────────────────────────────────────────────────
    ppl = None
    if args.perplexity:
        logger.info("Computing perplexity …")
        texts = [ex["text"] for ex in examples]
        ppl   = compute_perplexity(model, tokenizer, texts)
 
    # ── Generation metrics ────────────────────────────────────────────────
    results = evaluate(
        model, tokenizer, examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
 
    print_summary(results, ppl)
 
    # ── Save JSON ─────────────────────────────────────────────────────────
    out_path = args.output_json or os.path.join(OUTPUT_DIR, "eval_results.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = {"perplexity": ppl, **{k: v for k, v in results.items() if k != "per_example"}}
    with open(out_path, "w") as fh:
        json.dump({"summary": payload, "examples": results["per_example"]}, fh, indent=2)
    logger.info(f"Full results saved to {out_path}")
 
 
if __name__ == "__main__":
    main()
 
