"""
merge_lora.py — Merge a trained LoRA adapter into the base model and
save a standalone HuggingFace model ready for deployment / quantized
inference.
 
Usage:
    # Default: merges checkpoints/final_adapter → merged_model/
    python merge_lora.py
 
    # Custom paths
    python merge_lora.py \
        --adapter  checkpoints/final_adapter \
        --output   merged_model
"""
 
import argparse
import logging
import os
import sys

import torch  # type: ignore[import]
from peft import PeftModel  # type: ignore[import]
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
 
from config import MODEL_ID, CHECKPOINT_DIR, MERGED_DIR
 
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
 
 
def merge(adapter_path: str, output_path: str) -> None:
    if not os.path.isdir(adapter_path):
        logger.error(
            f"Adapter directory not found: {adapter_path}\n"
            f"Train first:  python train.py"
        )
        sys.exit(1)
 
    os.makedirs(output_path, exist_ok=True)
 
    # ── Load base model in fp16 (no quantization for merge) ───────────────
    logger.info(f"Loading base model {MODEL_ID} in fp16 (no quant) …")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
 
    # ── Attach LoRA adapter ───────────────────────────────────────────────
    logger.info(f"Loading LoRA adapter from {adapter_path} …")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
 
    # ── Merge + unload ────────────────────────────────────────────────────
    logger.info("Merging adapter weights into base model …")
    merged = peft_model.merge_and_unload()
 
    # ── Save merged model ─────────────────────────────────────────────────
    logger.info(f"Saving merged model to {output_path} …")
    merged.save_pretrained(output_path, safe_serialization=True)
 
    # ── Save tokenizer ────────────────────────────────────────────────────
    logger.info("Saving tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
 
    logger.info(f"Merged model saved to: {output_path}")
    logger.info("Next steps:")
    logger.info(f"  Inference:           python inference.py")
    logger.info(f"  Interactive test:    python test_generation.py")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument(
        "--adapter",
        default=os.path.join(CHECKPOINT_DIR, "final_adapter"),
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--output",
        default=MERGED_DIR,
        help="Output directory for the merged model",
    )
    args = parser.parse_args()
 
    logger.info("=" * 50)
    logger.info("  LoRA → Full Model Merge")
    logger.info("=" * 50)
    merge(args.adapter, args.output)
 
 
if __name__ == "__main__":
    main()
 