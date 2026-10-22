
import argparse
import logging
import os
import sys
 
# torch is optional and not required in this script; remove unused import to avoid linter warnings
 
from config import (
    MERGED_DIR,
    CHECKPOINT_DIR,
    MODEL_ID,
    INFERENCE_CONFIG,
)
from inference import load_model_and_tokenizer, generate
 
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.WARNING)
 
 
BANNER = r"""

"""
 
 
def print_help() -> None:
    print("""
Commands:
  /set temperature <float>   — generation temperature (0.1–1.0)
  /set top_p <float>         — nucleus sampling threshold
  /set max_tokens <int>      — maximum new tokens to generate
  /set repetition <float>    — repetition penalty (1.0 = none)
  /model <path>              — reload model from a different path
  /help                      — show this message
  quit / exit                — exit the program
 
Tips:
  • Lower temperature (0.1–0.3) → deterministic, cleaner code
  • Higher temperature (0.7–1.0) → more creative, may hallucinate
""")
 
 
def repl(model, tokenizer, args) -> None:
    # Mutable settings
    settings = {
        "temperature":        args.temperature,
        "max_new_tokens":     args.max_tokens,
        "top_p":              args.top_p,
        "repetition_penalty": INFERENCE_CONFIG.repetition_penalty,
    }
 
    print(BANNER)
    print_help()
 
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
 
        if not user_input:
            continue
 
        low = user_input.lower()
 
        # ── Built-in commands ──────────────────────────────────────────────
        if low in ("quit", "exit", "q"):
            print("Goodbye!")
            break
 
        if low == "/help":
            print_help()
            continue
 
        if low.startswith("/set "):
            parts = user_input.split()
            if len(parts) != 3:
                print("Usage: /set <key> <value>")
                continue
            _, key, val = parts
            mapping = {
                "temperature":  ("temperature",        float),
                "top_p":        ("top_p",              float),
                "max_tokens":   ("max_new_tokens",     int),
                "repetition":   ("repetition_penalty", float),
            }
            if key not in mapping:
                print(f"Unknown key '{key}'. Try: {list(mapping.keys())}")
                continue
            setting_key, cast = mapping[key]
            try:
                settings[setting_key] = cast(val)
                print(f"  {setting_key} = {settings[setting_key]}")
            except ValueError:
                print(f"Invalid value '{val}' for {key}")
            continue
 
        if low.startswith("/model "):
            new_path = user_input[7:].strip()
            if not os.path.isdir(new_path):
                print(f"Directory not found: {new_path}")
                continue
            print(f"Loading model from {new_path} …")
            model, tokenizer = load_model_and_tokenizer(new_path, quantize=args.quantize)
            print("Model reloaded ✔")
            continue
 
        # ── Generate ──────────────────────────────────────────────────────
        print("\n" + "─" * 60)
        print("Generating …\n")
        try:
            code = generate(
                model,
                tokenizer,
                user_input,
                **settings,
            )
            print(code)
            print("\n" + "─" * 60)
 
          
            if args.save_outputs:
                _save_output(user_input, code)
 
        except Exception as exc:
            print(f"[ERROR] Generation failed: {exc}")
            logger.exception(exc)
 
def _save_output(prompt: str, code: str) -> None:
    from config import OUTPUT_DIR
    import hashlib, time
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = hashlib.md5(prompt.encode()).hexdigest()[:8]
    fname = os.path.join(OUTPUT_DIR, f"{int(time.time())}_{slug}.py")
    with open(fname, "w") as fh:
        fh.write(f"# Prompt: {prompt}\n\n")
        fh.write(code)
    print(f"  [saved → {fname}]")
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive code generation REPL.")
    parser.add_argument("--model_path",  default=MERGED_DIR,    help="Path to merged model")
    parser.add_argument("--adapter",     default=None,          help="LoRA adapter path (used with base model)")
    parser.add_argument("--base_model",  default=MODEL_ID,      help="Base model when using --adapter")
    parser.add_argument("--quantize",    action="store_true",   help="4-bit quantized inference")
    parser.add_argument("--temperature", type=float, default=INFERENCE_CONFIG.temperature)
    parser.add_argument("--top_p",       type=float, default=INFERENCE_CONFIG.top_p)
    parser.add_argument("--max_tokens",  type=int,   default=INFERENCE_CONFIG.max_new_tokens)
    parser.add_argument("--save_outputs",action="store_true",   help="Save each generation to outputs/")
    args = parser.parse_args()
 

    if args.adapter:
        model_src  = args.base_model
        adapter    = args.adapter
    else:
        model_src  = args.model_path
        adapter    = None
        if not os.path.isdir(model_src):
            print(
                f"ERROR: Merged model not found at '{model_src}'.\n"
                f"Run:   python merge_lora.py\n"
                f"  OR:  python test_generation.py --adapter checkpoints/final_adapter"
            )
            sys.exit(1)
 
    print(f"Loading model from: {model_src}")
    if adapter:
        print(f"LoRA adapter:       {adapter}")
    print(f"Quantized (4-bit):  {args.quantize}")
 
    model, tokenizer = load_model_and_tokenizer(model_src, adapter, args.quantize)
    repl(model, tokenizer, args)
  
if __name__ == "__main__":
    main()
 