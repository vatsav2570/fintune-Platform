"""
Inference script — load your LoRA adapter and run inference.

Usage:
    source venv/bin/activate
    python inference.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL  = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
LORA_PATH   = "./qwen3coder-qlora"   # path to your saved adapter

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

print("📦 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("🧠 Loading base model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print(f"🔌 Loading LoRA adapter from {LORA_PATH}...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

def chat(user_message: str, system: str = "You are an expert coding assistant.") -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_message},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    print("\n💬 Chat with your fine-tuned Qwen3-Coder:")
    print("   (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        response = chat(user_input)
        print(f"\nModel: {response}\n")
