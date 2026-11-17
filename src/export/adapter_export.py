"""
src/export/adapter_export.py
─────────────────────────────
Export a LoRA adapter as a ZIP archive with README and loading instructions.
"""

import json
import pathlib
import shutil
import zipfile
import datetime
from typing import Optional


README_TEMPLATE = """\
# Fine-Tuned Model Adapter

**Exported:** {exported_at}
**Base Model:** `{base_model}`
**Training Method:** {method}
**Adapter Type:** LoRA / QLoRA

## Loading Instructions

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE_MODEL = "{base_model}"
ADAPTER_PATH = "./adapter"  # path to this directory

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
```

## Requirements

- Python 3.10+
- transformers >= 4.45
- peft >= 0.12
- bitsandbytes >= 0.43
- torch >= 2.0 (CUDA recommended)

## Notes

This is a LoRA adapter, NOT a standalone model.
The base model `{base_model}` must be downloaded separately.
"""


def export_adapter(
    adapter_dir: str | pathlib.Path,
    base_model_id: str,
    method: str,
    output_dir: str | pathlib.Path,
    extra_metadata: dict = None,
    progress_callback=None,
) -> tuple[bool, str]:
    """
    Copy adapter files + README into a ZIP archive.
    Returns (success, zip_path_or_error).
    """
    adapter_dir = pathlib.Path(adapter_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not adapter_dir.exists():
        return False, f"Adapter directory not found: {adapter_dir}"

    timestamp = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    safe_name = adapter_dir.name.replace(" ", "_")
    zip_path = output_dir / f"{safe_name}_adapter_{timestamp}.zip"

    try:
        # Estimate size
        total_bytes = sum(
            f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file()
        )
        if progress_callback:
            progress_callback(f"Exporting adapter ({total_bytes // 1024 // 1024} MB)...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write adapter files
            for f in adapter_dir.rglob("*"):
                if f.is_file() and not f.name.startswith("_train_"):
                    zf.write(f, f.relative_to(adapter_dir.parent))

            # Write README
            readme = README_TEMPLATE.format(
                exported_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
                base_model=base_model_id,
                method=method,
            )
            zf.writestr("README.md", readme)

            # Write metadata
            meta = {
                "exported_at": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
                "base_model": base_model_id,
                "adapter_dir": str(adapter_dir),
                "method": method,
                **(extra_metadata or {}),
            }
            zf.writestr("export_metadata.json", json.dumps(meta, indent=2))

        if progress_callback:
            progress_callback(f"✅ Export complete: {zip_path}")

        return True, str(zip_path)

    except Exception as exc:
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)
        return False, f"Export failed: {exc}"
