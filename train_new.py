"""
QLoRA Fine-Tuning Script (CodeLlama-7B - Stable Version)
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME   = "codellama/CodeLlama-7b-hf"
DATASET_PATH = "qwen_coder_100_samples.jsonl"
OUTPUT_DIR   = "./codellama-7b-qlora"

# 🔥 VERY IMPORTANT (7B SAFE SETTINGS)
MAX_SEQ_LEN  = 256     # ⚠️ DO NOT increase initially
BATCH_SIZE   = 1
GRAD_ACCUM   = 4       # effective batch = 4

EPOCHS       = 10
LR           = 2e-4

# LoRA
LORA_R       = 8       # ↓ smaller = more stable
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05

# ─────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True

# ─────────────────────────────────────────────
# 4-bit CONFIG
# ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # ⚠️ safer than bf16 for CodeLlama
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ─────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────
print("📦 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
print("🧠 Loading model (4-bit)...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# ─────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],  # ⚠️ reduced modules for stability
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_chat(example):
    messages = example["messages"]
    
    text = ""
    for msg in messages:
        if msg["role"] == "user":
            text += f"### Instruction:\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            text += f"### Response:\n{msg['content']}\n"

    return {"text": text}

dataset = dataset.map(format_chat, num_proc=1)

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    gradient_checkpointing=True,

    learning_rate=LR,
    fp16=True,   # ⚠️ safer for CodeLlama
    bf16=False,

    optim="paged_adamw_32bit",

    logging_steps=5,
    save_steps=50,
    save_total_limit=2,

    dataloader_num_workers=0,

    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
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
print("🚀 Training started...")
trainer.train()

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Done!")