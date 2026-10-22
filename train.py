import argparse
import os
import sys
import logging

from pathlib import Path

try:
    import torch  # type: ignore[import]
except ImportError as e:  # pragma: no cover - handle missing optional dependency
    raise ImportError(
        "The 'torch' library is required but could not be imported. "
        "Install it with: pip install torch"
    ) from e
try:
    from datasets import load_dataset, DatasetDict  # type: ignore[import]
except ImportError as e:  # pragma: no cover - handle missing optional dependency
    raise ImportError(
        "The 'datasets' library is required but could not be imported. "
        "Install it with: pip install datasets"
    ) from e
try:
    from peft import (  # type: ignore[import]
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )
except ImportError as e:  # pragma: no cover - handle missing optional dependency
    raise ImportError(
        "The 'peft' library is required but could not be imported. "
        "Install it with: pip install peft"
    ) from e
try:
    from transformers import (  # type: ignore[import]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ImportError as e:  # pragma: no cover - handle missing optional dependency
    raise ImportError(
        "The 'transformers' library is required but could not be imported. "
        "Install it with: pip install transformers"
    ) from e
from config import (
    MODEL_ID,
    MAX_SEQ_LENGTH,
    CHECKPOINT_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    LOGS_DIR,
    LORA_CONFIG,
    BNB_CONFIG,
    TRAIN_CONFIG,
    ASSISTANT_TOKEN,
    EOS_TOKEN,
)
 
# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOGS_DIR, "train.log")),
    ],
)
logger = logging.getLogger(__name__)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# GPU check
# ──────────────────────────────────────────────────────────────────────────────
def check_gpu() -> None:
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA not available — training will be very slow on CPU. "
            "Consider using a GPU instance."
        )
    else:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gb    = props.total_memory / 1024 ** 3
            logger.info(f"GPU {i}: {props.name} — {gb:.1f} GB VRAM")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────
def load_tokenizer(model_id: str):
    logger.info(f"Loading tokenizer from {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",   # right-padding for Causal LM
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")
    return tokenizer
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_id: str, use_4bit: bool = True):
    logger.info(f"Loading base model {model_id} (4-bit={use_4bit}) …")
 
    bnb_cfg = None
    if use_4bit:
        # use already-imported torch from top-level import
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit             = BNB_CONFIG.load_in_4bit,
            bnb_4bit_quant_type      = BNB_CONFIG.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype   = dtype_map[BNB_CONFIG.bnb_4bit_compute_dtype],
            bnb_4bit_use_double_quant= BNB_CONFIG.bnb_4bit_use_double_quant,
        )
 
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",   # safe default; change to "flash_attention_2" if installed
    )
    model.config.use_cache = False          # required for gradient-checkpointing
    model.config.pretraining_tp = 1        # disable tensor-parallel for fine-tuning
    return model
 
 
# ──────────────────────────────────────────────────────────────────────────────
# LoRA
# ──────────────────────────────────────────────────────────────────────────────
def apply_lora(model):
    logger.info("Preparing model for k-bit training …")
    model = prepare_model_for_kbit_training(model)
 
    lora_cfg = LoraConfig(
        r              = LORA_CONFIG.r,
        lora_alpha     = LORA_CONFIG.lora_alpha,
        target_modules = LORA_CONFIG.target_modules,
        lora_dropout   = LORA_CONFIG.lora_dropout,
        bias           = LORA_CONFIG.bias,
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
 
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%) — "
        f"LoRA rank={LORA_CONFIG.r}, alpha={LORA_CONFIG.lora_alpha}"
    )
    return model
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Tokenisation + loss masking
# ──────────────────────────────────────────────────────────────────────────────
def make_tokenize_fn(tokenizer, max_len: int):
    """
    Returns a function that tokenises a single example and masks the
    prompt (user turn) so the loss is only computed on the assistant turn.
    """
    # We need the token-id(s) that start the assistant block so we can
    # find the boundary in the token sequence.
    assistant_ids = tokenizer.encode(ASSISTANT_TOKEN, add_special_tokens=False)
 
    def tokenize(example: dict) -> dict:
        text = example["text"]
 
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_tensors=None,
        )
 
        input_ids = tokenized["input_ids"]
        labels    = list(input_ids)   # copy
 
        # ── Find assistant-turn boundary ──────────────────────────────────
        # Walk through input_ids looking for the assistant token sequence.
        boundary = None
        for i in range(len(input_ids) - len(assistant_ids) + 1):
            if input_ids[i : i + len(assistant_ids)] == assistant_ids:
                boundary = i + len(assistant_ids)   # first token AFTER <|assistant|>
                break
 
        # Mask everything up to (and including) the assistant token with -100
        if boundary is not None:
            labels[:boundary] = [-100] * boundary
 
        tokenized["labels"] = labels
        return tokenized
 
    return tokenize
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
def load_and_tokenize(tokenizer) -> DatasetDict:
    for path, split in [(TRAIN_DATA_PATH, "train"), (VAL_DATA_PATH, "validation")]:
        if not os.path.exists(path):
            logger.error(
                f"Split file not found: {path}\n"
                f"Run:  python prepare_dataset.py  first."
            )
            sys.exit(1)
 
    logger.info("Loading dataset splits …")
    raw = load_dataset(
        "json",
        data_files={"train": TRAIN_DATA_PATH, "validation": VAL_DATA_PATH},
        split=None,
    )
 
    tokenize_fn = make_tokenize_fn(tokenizer, MAX_SEQ_LENGTH)
    logger.info("Tokenising dataset …")
    tokenized = raw.map(
        tokenize_fn,
        remove_columns=raw["train"].column_names,
        batched=False,
        num_proc=max(1, os.cpu_count() // 2),
        desc="Tokenising",
    )
 
    logger.info(
        f"Train: {len(tokenized['train']):,} examples | "
        f"Val: {len(tokenized['validation']):,} examples"
    )
    return tokenized
 
 
# ──────────────────────────────────────────────────────────────────────────────
# TrainingArguments
# ──────────────────────────────────────────────────────────────────────────────
def build_training_args(resume: str | None = None) -> TrainingArguments:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
 
    cfg = TRAIN_CONFIG
 
    return TrainingArguments(
        output_dir                  = cfg.output_dir,
        num_train_epochs            = cfg.num_train_epochs,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        per_device_eval_batch_size  = cfg.per_device_eval_batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        learning_rate               = cfg.learning_rate,
        weight_decay                = cfg.weight_decay,
        warmup_ratio                = cfg.warmup_ratio,
        lr_scheduler_type           = cfg.lr_scheduler_type,
        fp16                        = cfg.fp16,
        bf16                        = cfg.bf16,
        optim                       = cfg.optim,
        logging_steps               = cfg.logging_steps,
        eval_strategy               = cfg.eval_strategy,
        save_strategy               = cfg.save_strategy,
        save_total_limit            = cfg.save_total_limit,
        load_best_model_at_end      = cfg.load_best_model_at_end,
        metric_for_best_model       = cfg.metric_for_best_model,
        greater_is_better           = cfg.greater_is_better,
        report_to                   = cfg.report_to,
        dataloader_num_workers      = cfg.dataloader_num_workers,
        group_by_length             = TRAIN_CONFIG.group_by_length,
        max_grad_norm               = cfg.max_grad_norm,
        seed                        = cfg.seed,
        logging_dir                 = LOGS_DIR,
        gradient_checkpointing      = True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────── 
def main() -> None:
    print(">>> main() started")
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 3B with LoRA.")
    parser.add_argument("--resume",     default=None,  help="Path to a checkpoint to resume from")
    parser.add_argument("--epochs",     type=int,      help="Override num_train_epochs")
    parser.add_argument("--batch_size", type=int,      help="Override per_device_train_batch_size")
    parser.add_argument("--no_4bit",    action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()
 
    # ── Apply CLI overrides ──────────────────────────────────────────────
    if args.epochs:
        TRAIN_CONFIG.num_train_epochs = args.epochs
    if args.batch_size:
        TRAIN_CONFIG.per_device_train_batch_size = args.batch_size
 
    set_seed(TRAIN_CONFIG.seed)
    os.makedirs(LOGS_DIR, exist_ok=True)
 
    logger.info("=" * 60)
    logger.info("  Llama 3.2 3B LoRA Fine-tuning")
    logger.info("=" * 60)
 
    check_gpu()
 
    # ── Load components ──────────────────────────────────────────────────
    tokenizer = load_tokenizer(MODEL_ID)
    model     = load_model(MODEL_ID, use_4bit=not args.no_4bit)
    model     = apply_lora(model)
 
    # ── Dataset ──────────────────────────────────────────────────────────
    dataset   = load_and_tokenize(tokenizer)
 
    # ── Collator ─────────────────────────────────────────────────────────
    from transformers import DataCollatorForLanguageModeling # type: ignore

    collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
 
    # ── Trainer ──────────────────────────────────────────────────────────
    training_args = build_training_args(resume=args.resume) 
 
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["validation"],
        tokenizer       = tokenizer,
        data_collator   = collator,
    )
  
    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=args.resume or TRAIN_CONFIG.resume_from_checkpoint)
 
    # ── Save final LoRA adapter ───────────────────────────────────────────
    final_adapter_path = os.path.join(CHECKPOINT_DIR, "final_adapter")
    logger.info(f"Saving final LoRA adapter to {final_adapter_path} …")
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
 
    logger.info("Training complete ✔")
    logger.info(f"Adapter saved to: {final_adapter_path}")
    logger.info("Next step → run:  python merge_lora.py")
 
 
if __name__ == "__main__":
    print(">>> main() started")
    # logger.info("main function called from command line; executing main() …")  
    main()
else:
    print(">>>train.py imported as a module; main() not executed.")
    # logger.info("train.py imported as a module; main() not executed.")  
   
   