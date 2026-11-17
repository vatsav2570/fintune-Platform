"""
inference_9800.py
─────────────────────────────────────────────────────────────────
Interactive CLI for testing the fine-tuned Qwen2.5-0.5B-Instruct
model on 9800/B Series Programmable AC Power Source questions.

Usage:
    source venv/bin/activate
    python inference_9800.py

Options (edit constants below):
    ADAPTER_PATH — path to your saved LoRA adapter (OUTPUT_DIR from train.py)
    USE_ADAPTER  — set False to load the original base model for comparison
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./qwen25-0.5b-9800v2-qlora"   # set to OUTPUT_DIR from train.py
USE_ADAPTER  = True                           # set False to compare with base model

SYSTEM_PROMPT = (
    "You are a strict technical assistant for B&K Precision 9800/B Series "
    "Programmable AC Power Sources. Answer ONLY using datasheet facts. "
    "If the answer is not explicitly known, say: 'I don't know'. "
    "Do NOT guess or approximate. Be precise and factual."
)

MAX_NEW_TOKENS = 300
TEMPERATURE    = 0.1    # low = more factual/deterministic
TOP_P          = 0.9

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print(f"📦 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print(f"🧠 Loading base model ({BASE_MODEL})...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="eager",
)

if USE_ADAPTER:
    print(f"🔌 Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()
print("✅ Model ready!\n")

# ─────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────
def ask(question: str) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Strip the prompt tokens from the output
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# EXAMPLE QUESTIONS (run as smoke-test)
# ─────────────────────────────────────────────
DEMO_QUESTIONS = [
    "What is the maximum power output of model 9805/B?",
    "What interfaces does the 9800/B Series support for remote control?",
    "What is List mode and how many steps does it support?",
    "What is the voltage measurement accuracy of the 9800/B Series?",
]


def run_demo():
    print("=" * 60)
    print("  9800/B Series Q&A — Fine-tuned Qwen2.5-0.5B Demo")
    print("=" * 60)
    for q in DEMO_QUESTIONS:
        print(f"\n❓ {q}")
        answer = ask(q)
        print(f"💡 {answer}")
        print("-" * 60)


# ─────────────────────────────────────────────
# INTERACTIVE CHAT LOOP
# ─────────────────────────────────────────────
def interactive():
    print("\n" + "=" * 60)
    print("  Interactive Mode — type 'quit' or 'exit' to stop")
    print("  Type 'demo' to run the demo questions")
    print("=" * 60 + "\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "demo":
            run_demo()
            continue

        answer = ask(user_input)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    run_demo()
    interactive()
