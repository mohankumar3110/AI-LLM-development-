"""
config.py — Central configuration for Llama 3.2 1B LoRA fine-tuning.
Edit values here; every other script imports from this file.
"""
 
import os
from dataclasses import dataclass, field
from typing import List, Optional
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Pat
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
MERGED_DIR      = os.path.join(BASE_DIR, "merged_model")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
 
# Raw dataset (JSONL)
RAW_DATA_PATH   = os.path.join(DATA_DIR, "dataset.jsonl")
# Processed splits (created by prepare_dataset.py)
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_doubled.jsonl")
VAL_DATA_PATH   = os.path.join(DATA_DIR, "val_doubled.jsonl")
TEST_DATA_PATH  = os.path.join(DATA_DIR, "test_doubled.jsonl")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Alternative: For 0.3B model (smaller, faster)
# MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # Use 1B-Instruct (0.3B not available as standalone instruct)
 
# Chat-template tokens used by Llama 3.2 Instruct
USER_TOKEN      = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN       = "<|eot_id|>"       # end-of-turn in Llama 3.x
 
# Sequence length
MAX_SEQ_LENGTH = 512
 
 
# ──────────────────────────────────────────────────────────────────────────────
# LoRA
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LoRAConfig:
    r: int                    = 16          # rank
    lora_alpha: int           = 32          # scaling factor (alpha/r = 2)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float       = 0.1
    bias: str                 = "none"
    task_type: str            = "CAUSAL_LM"
 
 
LORA_CONFIG = LoRAConfig()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# BitsAndBytes (4-bit QLoRA)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BnBConfig:
    load_in_4bit: bool                  = True
    bnb_4bit_quant_type: str            = "nf4"          # NormalFloat4
    bnb_4bit_compute_dtype: str         = "float16"      # compute in fp16
    bnb_4bit_use_double_quant: bool     = True           # nested quantization
 
 
BNB_CONFIG = BnBConfig()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    output_dir: str              = CHECKPOINT_DIR
    num_train_epochs: int        = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int  = 1
    gradient_accumulation_steps: int = 16   # effective batch = 2*8 = 16
    learning_rate: float         = 2e-4
    weight_decay: float          = 0.001
    warmup_ratio: float          = 0.03
    lr_scheduler_type: str       = "cosine"
    fp16: bool                   = False
    bf16: bool                   = False    # set True if A100/H100
    optim: str                   = "paged_adamw_32bit"
    logging_steps: int           = 10
    eval_strategy: str           = "epoch"
    save_strategy: str           = "epoch"
    save_total_limit: int        = 3        # keep last 3 checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model: str   = "eval_loss"
    greater_is_better: bool      = False
    report_to: str               = "tensorboard"
    dataloader_num_workers: int  = 0
    group_by_length: bool        = True     # speeds up training
    max_grad_norm: float         = 0.3
    seed: int                    = 42
    # Resume — set to a checkpoint path to resume, e.g. "checkpoints/checkpoint-100"
    resume_from_checkpoint: Optional[str] = None
 
 
TRAIN_CONFIG = TrainConfig()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Dataset split ratios
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.85
VAL_RATIO   = 0.10
TEST_RATIO  = 0.05
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Inference defaults
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class InferenceConfig:
    max_new_tokens: int = 128
    temperature: float      = 0.2
    top_p: float            = 0.95
    top_k: int              = 50
    do_sample: bool         = True
    repetition_penalty: float = 1.1
 
 
INFERENCE_CONFIG = InferenceConfig()
