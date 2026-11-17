"""
QLoRA Fine-Tuning Script for Qwen3-Coder-30B-A3B-Instruct
DGX Spark (NVIDIA GB10 / aarch64 / CUDA 13.0)

Usage:
    source venv/bin/activate
    python train.py

Dataset format (train.jsonl):
    {"messages":[
        {"role":"system","content":"You are a coding assistant"},
        {"role":"user","content":"Write bubble sort in Python"},
        {"role":"assistant","content":"def bubble_sort(arr): ..."}
    ]}
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"     # ~500M params — fits any GPU
DATASET_PATH = "train_9800_300.jsonl"             # 304 Q&A pairs from PDF datasheet
OUTPUT_DIR   = "./qwen25-0.5b-9800v2-qlora"      # where the LoRA adapter is saved

# LoRA hyperparameters (smaller rank suits a 0.5B model)
LORA_R       = 16          # LoRA rank — 16 is sufficient for 0.5B
LORA_ALPHA   = 32          # 2× rank
LORA_DROPOUT = 0.05

# Training hyperparameters
MAX_SEQ_LEN  = 2048        # datasheet answers are short; 2048 is plenty
BATCH_SIZE   = 4           # 0.5B is tiny — use larger batch
GRAD_ACCUM   = 4           # effective batch = 4 × 4 = 16
EPOCHS       = 25           # small dataset benefits from more epochss
LR           = 3e-4        # slightly higher LR for small model
WARMUP_RATIO = 0.05
SAVE_STEPS   = 50
LOG_STEPS    = 5

# ─────────────────────────────────────────────
# 4-BIT QUANTIZATION CONFIG (QLoRA)
# ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # GB10 fully supports bf16
    bnb_4bit_use_double_quant=True,           # nested quantization = less VRAM
    bnb_4bit_quant_type="nf4",               # NormalFloat4 — best for LLMs
)

# ─────────────────────────────────────────────
# LOAD TOKENIZER
# ─────────────────────────────────────────────
print(f"📦 Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token   # required for batching
tokenizer.padding_side = "right"            # SFTTrainer expects right-padding

# ─────────────────────────────────────────────
# LOAD MODEL (4-bit QLoRA)
# ─────────────────────────────────────────────
print("🧠 Loading Qwen2.5-0.5B-Instruct in 4-bit (QLoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="eager",  # use "flash_attention_2" if installed
)

# Prepare model for k-bit (4-bit) training — crucial step for QLoRA
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # disable KV-cache during training

# ─────────────────────────────────────────────
# LORA CONFIG
# ─────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # expect ~1-2% trainable params

# ─────────────────────────────────────────────
# LOAD & FORMAT DATASET
# ─────────────────────────────────────────────
print(f"📂 Loading dataset from {DATASET_PATH}...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_chat(example):
    """Apply Qwen2.5 chat template to messages."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_chat, num_proc=4)
print(f"✅ Dataset ready: {len(dataset)} examples")

# ─────────────────────────────────────────────
# TRAINING ARGUMENTS
# ─────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    learning_rate=LR,
    bf16=True,
    fp16=False,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=False,
    dataloader_num_workers=4,
    report_to="none",
    ddp_find_unused_parameters=False,
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
)

# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=training_args,
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("🚀 Starting training on 9800 Series datasheet Q&A...")
trainer.train()

# ─────────────────────────────────────────────
# SAVE ADAPTER
# ─────────────────────────────────────────────
print(f"💾 Saving LoRA adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Training complete! Adapter saved to:", OUTPUT_DIR)
print("\nNext step: run   python inference_9800.py")
