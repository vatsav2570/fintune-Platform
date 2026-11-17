"""
src/export/merged_model_export.py
──────────────────────────────────
Merge a LoRA adapter into its base model and export the full weights.
Only offered when quantization=False (can't reliably merge 4-bit).
"""

import pathlib
import json
import datetime
import zipfile
from typing import Optional


README_MERGED = """\
# Merged Fine-Tuned Model

**Exported:** {exported_at}
**Original Base Model:** `{base_model}`

This is a fully merged model — no adapter files or base model required.

## Loading Instructions

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./merged_model"  # path to this directory

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
```

## Requirements

- Python 3.10+
- transformers >= 4.45
- torch >= 2.0

## Notes

This model was produced by merging LoRA adapter weights into the base model.
It is a standalone model that does NOT require the original base model.
"""


def merge_and_export(
    base_model_path: str,
    adapter_path: str,
    output_dir: str | pathlib.Path,
    progress_callback=None,
) -> tuple[bool, str]:
    """
    Load base model (fp16/bf16 — NOT 4-bit), apply adapter, merge, save.
    Returns (success, merged_path_or_error).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        output_dir = pathlib.Path(output_dir)
        timestamp = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
        merged_dir = output_dir / f"merged_{timestamp}"
        merged_dir.mkdir(parents=True, exist_ok=True)

        def _cb(msg):
            if progress_callback:
                progress_callback(msg)

        _cb(f"Loading base model (bf16) from {base_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",   # merge on CPU to avoid VRAM constraint
            trust_remote_code=True,
        )

        _cb(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

        _cb("Merging adapter into base model weights...")
        model = model.merge_and_unload()

        _cb(f"Saving merged model to {merged_dir}...")
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

        # Write README
        readme = README_MERGED.format(
            exported_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
            base_model=base_model_path,
        )
        (merged_dir / "README.md").write_text(readme, encoding="utf-8")

        _cb(f"✅ Merge complete: {merged_dir}")
        return True, str(merged_dir)

    except Exception as exc:
        return False, f"Merge failed: {exc}"
