"""
inference.py — Load the fine-tuned model (adapter OR merged) and run
generation. Supports:
  • LoRA adapter (loads base + PEFT, fast to start)
  • Merged model (standalone, slightly faster inference)
  • 4-bit QLoRA inference (lower VRAM)
 
Usage:
    # With merged model (recommended after merge_lora.py)
    python inference.py --prompt "Generate a procedural stone arch"
 
    # With LoRA adapter on top of base model
    python inference.py \
        --adapter checkpoints/final_adapter \
        --prompt  "Create a medieval tower"
 
    # With 4-bit quantization for low-VRAM machines
    python inference.py --quantize --prompt "Make a cobblestone path"
"""
 
import argparse
import logging
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
 
from config import (
    MODEL_ID,
    MERGED_DIR,
    CHECKPOINT_DIR,
    MAX_SEQ_LENGTH,
    INFERENCE_CONFIG,
    USER_TOKEN,
    ASSISTANT_TOKEN,
)
 
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Build prompt
# ──────────────────────────────────────────────────────────────────────────────
 
def build_prompt(prompt: str) -> str:
    return f"{USER_TOKEN}\n{prompt.strip()}\n{ASSISTANT_TOKEN}\n"
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────
 
def load_model_and_tokenizer(
    model_path: str,
    adapter_path: str | None = None,
    quantize: bool = False,
):
    logger.info(f"Loading tokenizer from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    bnb_cfg = None
    if quantize:
        logger.info("Enabling 4-bit quantization …")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
 
    logger.info(f"Loading model from {model_path} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
 
    # Optionally attach a LoRA adapter
    if adapter_path:
        try:
            # lazy import to avoid hard dependency when not using adapters
            import importlib
            peft = importlib.import_module("peft")
            PeftModel = getattr(peft, "PeftModel", None)
            if PeftModel is None:
                raise ImportError("peft.PeftModel not found")
        except Exception as exc:  # ImportError or attribute missing
            logger.error(
                "The 'peft' package is required to load a LoRA adapter. "
                "Install it with 'pip install peft'."
            )
            raise RuntimeError("Missing required package 'peft'.") from exc

 
    model.eval()
    logger.info("Model ready ✔")
    return model, tokenizer
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Generate
# ──────────────────────────────────────────────────────────────────────────────
 
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = INFERENCE_CONFIG.temperature,
    top_p: float = INFERENCE_CONFIG.top_p,
    top_k: int = INFERENCE_CONFIG.top_k,
    do_sample: bool = INFERENCE_CONFIG.do_sample,
    repetition_penalty: float = INFERENCE_CONFIG.repetition_penalty,
) -> str:

    formatted = build_prompt(prompt)

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
     max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    t0 = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - t0

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)

    logger.info(
        f"Generated {len(new_ids)} tokens in {elapsed:.2f}s"
    )

    return decoded.strip()
 
# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned model.")
    parser.add_argument("--prompt",          required=True,        help="User prompt / instruction")
    parser.add_argument("--model_path",      default=MERGED_DIR,   help="Path to merged model (default: merged_model/)")
    parser.add_argument("--adapter",         default=None,         help="Path to LoRA adapter (optional, used with base model)")
    parser.add_argument("--base_model",      default=MODEL_ID,     help="HF repo or local path to base model (used with --adapter)")
    parser.add_argument("--quantize",        action="store_true",  help="Load model in 4-bit for low VRAM")
    parser.add_argument("--max_new_tokens",  type=int, default=INFERENCE_CONFIG.max_new_tokens)
    parser.add_argument("--temperature",     type=float, default=INFERENCE_CONFIG.temperature)
    parser.add_argument("--top_p",           type=float, default=INFERENCE_CONFIG.top_p)
    args = parser.parse_args()
 
    # Decide which model path to use
    if args.adapter:
        # Load base model and attach adapter
        if not os.path.isdir(os.path.expanduser(args.base_model)) and not args.base_model.startswith("meta-llama"):
            logger.warning(f"Base model '{args.base_model}' not found locally; will download from HF Hub.")
        model_path = args.base_model
        adapter    = args.adapter
    else:
        # Merged model      
        model_path = args.model_path
        adapter    = None
        if not os.path.isdir(model_path):
            logger.error(
                f"Merged model not found at '{model_path}'.\n"
                f"Run:  python merge_lora.py   OR use --adapter checkpoints/final_adapter"
            )
            sys.exit(1)
 
    model, tokenizer = load_model_and_tokenizer(model_path, adapter, args.quantize)
 
    print("\n" + "=" * 60)
    print("PROMPT:")
    print(args.prompt)
    print("=" * 60)
    print("GENERATED CODE:")
    print("=" * 60) 
 
    result = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(result)
    print("=" * 60)
    import sys

print("=" * 60)
print("Inference Python:", sys.executable)
print("Inference cwd:", os.getcwd())
print("=" * 60)
 
 
if __name__ == "__main__":
    main()
 