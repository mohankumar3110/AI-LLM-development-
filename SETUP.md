# 🚀 Llama 3.2 Training Project - Setup Guide

## ✅ Quick Start (3 Commands)

```bash
# 1. Clean up duplicate files
python cleanup.py

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Run complete pipeline: augment → prepare → train
python pipeline.py --action all --factor 1
```

**OR** run everything at once:
```bash
python setup_and_train.py
```

---

## 📊 What You Have

- **Base Model:** Llama 3.2 1B Instruct
- **Dataset:** ~700 samples in `data/dataset.jsonl`
- **Fine-tuning:** LoRA with 4-bit quantization
- **Training Time:** 30 mins - 2 hours (GPU dependent)

---

## 🎯 Core Files (Cleaned)

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `pipeline.py` | Automates all steps |
| `augment_dataset.py` | Dataset augmentation (700 → 1,400) |
| `prepare_dataset.py` | Format & split data |
| `inference.py` | Test your trained model |
| `evaluate.py` | Evaluate on test set |
| `merge_lora.py` | Export final model |
| `config.py` | Configuration settings |
| `requirements.txt` | Dependencies |
| `cleanup.py` | Remove duplicate files |
| `setup_and_train.py` | Full automated pipeline |

---

## 📁 Project Structure

```
project/
├── data/
│   ├── dataset.jsonl         # Original data
│   ├── dataset_augmented.jsonl
│   ├── train.jsonl           # Training split
│   ├── val.jsonl             # Validation split
│   └── test.jsonl            # Test split
├── checkpoints/              # Saved models (after training)
├── models/                   # Base model cache
├── logs/                     # Training logs
├── outputs/                  # Generated outputs
├── train.py                  # Training script
├── pipeline.py               # Automation
├── config.py                 # Settings
├── requirements.txt          # Dependencies
└── *.py                      # Utility scripts
```

---

## 🔧 Installation & Training

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Augment Dataset (Optional but Recommended)
```bash
python augment_dataset.py 1  # Doubles dataset to 1,400 samples
```

### Step 3: Prepare Data
```bash
python prepare_dataset.py --input data/dataset_augmented.jsonl
```

### Step 4: Train Model
```bash
python train.py
```

Training will save checkpoints to `checkpoints/` folder after each epoch.

### Step 5: Test Inference
```bash
python inference.py "Write a function to sort a list"
```

---

## ⚡ One-Command Solution

Run everything automatically:
```bash
python pipeline.py --action all --factor 1
```

This handles:
- ✅ Dataset augmentation
- ✅ Data preparation & splits
- ✅ Model training
- ✅ Validation & metrics

---

## 📈 Expected Training Output

```
GPU 0: NVIDIA RTX 3090 — 24.0 GB VRAM
Loading base model...
Loading tokenizer...
Initializing LoRA adapters...

Training progress:
[████████████████████████████] 100% | 1190/1190 [00:45<00:00, 26 samples/s]

Training complete!
✅ Best model saved to: checkpoints/
```

---

## 🧪 After Training

### Test Your Model
```bash
python inference.py "Your prompt here"
```

### Evaluate on Test Set
```bash
python evaluate.py
```

### Export for Deployment
```bash
python merge_lora.py
# Creates: merged_model/ (standalone model without PEFT)
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  # Base model
MAX_SEQ_LENGTH = 512                            # Max token length
per_device_train_batch_size = 4                 # Batch size
num_train_epochs = 3                            # Training epochs
learning_rate = 5e-4                            # Learning rate
```

---

## 🐛 Troubleshooting

### Out of Memory (CUDA)
```python
# In config.py:
per_device_train_batch_size = 1      # Reduce batch size
gradient_accumulation_steps = 4      # Reduce gradients
MAX_SEQ_LENGTH = 256                 # Reduce sequence length
```

### HuggingFace Token Required
```bash
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens
```

### Model Download Issues
```bash
# Manually download model:
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B-Instruct', trust_remote_code=True)"
```

---

## 📊 GPU Training Times

| GPU | Time | VRAM |
|-----|------|------|
| CPU | 6-12 hours | N/A |
| RTX 3060 (12GB) | 45-90 mins | 11.5 GB |
| RTX 3090 (24GB) | 20-40 mins | 22 GB |
| RTX 4090 (24GB) | 15-30 mins | 24 GB |

---

## 📁 Clean Setup (Removed Files)

The `cleanup.py` script removes:
- ❌ Duplicate training scripts (`train_optimized.py`, `train_simple.py`)
- ❌ Duplicate augmentation scripts
- ❌ Guide files (.txt, .md guides)
- ❌ Batch files (.bat)
- ❌ Obsolete scripts

**Keeps:** Core training code only ✅

---

## 🎓 Next Steps

1. **Run cleanup:**
   ```bash
   python cleanup.py
   ```

2. **Start training:**
   ```bash
   python setup_and_train.py
   ```

3. **Monitor training:**
   ```bash
   # In another terminal:
   tensorboard --logdir logs/
   # Open: http://localhost:6006
   ```

4. **Evaluate results:**
   ```bash
   python evaluate.py
   ```

---

## 📞 Quick Reference

```bash
# Cleanup & install
python cleanup.py
pip install -r requirements.txt

# Quick train (all steps)
python pipeline.py --action all --factor 1

# OR step-by-step
python augment_dataset.py 1
python prepare_dataset.py --input data/dataset_augmented.jsonl
python train.py

# Test model
python inference.py "Your prompt"

# Export model
python merge_lora.py

# Evaluate
python evaluate.py
```

---

**Ready to train?** Run: `python cleanup.py && python setup_and_train.py` 🚀
