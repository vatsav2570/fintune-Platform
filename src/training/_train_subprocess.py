"""
src/training/_train_subprocess.py
──────────────────────────────────
Self-contained training script executed as a subprocess.
Reads config from a JSON file path given as argv[1].
Mirrors the logic of the existing train.py / train_new.py pipeline.

ALL output goes to stdout (redirected to the log file by the worker).
"""

import json
import sys
import pathlib
import os

def main():
    if len(sys.argv) < 2:
        print("ERROR: config path required as argv[1]")
        sys.exit(1)

    cfg_path = pathlib.Path(sys.argv[1])
    if not cfg_path.exists():
        print(f"ERROR: config file not found: {cfg_path}")
        sys.exit(1)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    method = cfg.get("method", "qlora_sft")

    print(f"{'='*60}")
    print(f"  Training Method : {method.upper()}")
    print(f"  Base Model      : {cfg['base_model_path']}")
    print(f"  Dataset         : {cfg['dataset_path']}")
    print(f"  Output Dir      : {cfg['output_dir']}")
    print(f"{'='*60}\n")

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # ── Tokenizer ─────────────────────────────────────────────
    print(f"📦 Loading tokenizer from {cfg['base_model_path']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model_path"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded.")

    # ── Model ─────────────────────────────────────────────────
    if method == "qlora_sft":
        print("🧠 Loading model in 4-bit (QLoRA)...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=cfg.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float16,
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_double_quant", True),
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model_path"],
            quantization_config=bnb_cfg,
            device_map=cfg.get("device_map", "cuda:0"),
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = prepare_model_for_kbit_training(model)

    elif method == "lora_sft":
        print("🧠 Loading model in bf16 (LoRA)...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model_path"],
            torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float16,
            device_map=cfg.get("device_map", "cuda:0"),
            trust_remote_code=True,
            attn_implementation="eager",
        )

    else:  # full_sft
        print("🧠 Loading model in bf16 (Full SFT)...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model_path"],
            torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float16,
            device_map=cfg.get("device_map", "cuda:0"),
            trust_remote_code=True,
            attn_implementation="eager",
        )

    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────
    if method in ("qlora_sft", "lora_sft"):
        lora_targets = cfg.get("lora_targets") or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_config = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            target_modules=lora_targets,
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────
    print(f"📂 Loading dataset from {cfg['dataset_path']}...")
    dataset = load_dataset("json", data_files=cfg["dataset_path"], split="train")

    def format_chat(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat, num_proc=1)  # num_proc>1 hangs on Windows in subprocess
    print(f"✅ Dataset ready: {len(dataset)} examples")

    # ── Training args ─────────────────────────────────────────
    output_dir = cfg["output_dir"]
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.get("epochs", 3),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        learning_rate=cfg.get("lr", 2e-4),
        bf16=cfg.get("bf16", True),
        fp16=cfg.get("fp16", False),
        optim=cfg.get("optim", "paged_adamw_32bit"),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        weight_decay=cfg.get("weight_decay", 0.01),
        logging_steps=cfg.get("log_steps", 5),
        save_steps=cfg.get("save_steps", 50),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=False,
        dataloader_num_workers=0,  # >0 causes hangs on Windows in a subprocess
        report_to="none",
        ddp_find_unused_parameters=False,
        dataset_text_field="text",
        max_length=cfg.get("max_seq_len", 2048),
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────
    print("🚀 Starting training...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────
    print(f"💾 Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Training complete! Saved to: {output_dir}")

    # Write completion marker
    marker = pathlib.Path(output_dir) / "_training_complete.json"
    marker.write_text(json.dumps({
        "status": "complete",
        "method": method,
        "output_dir": output_dir,
    }), encoding="utf-8")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as exc:
        print(f"\n❌ TRAINING FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
