"""
src/export/full_model_export.py
────────────────────────────────
Export a full-parameter fine-tuned model as a ZIP archive
with README and loading instructions.
"""

import datetime
import json
import pathlib
import zipfile


README_FULL = """\
# Full Fine-Tuned Model

**Exported:** {exported_at}
**Original Base Model:** `{base_model}`
**Training Method:** Full SFT (all parameters updated)

## Loading Instructions

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./model"  # path to this directory

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
- torch >= 2.0 with CUDA (recommended)
- accelerate >= 0.33

## Notes

This is a **complete** fine-tuned model.
No base model or adapter files are required.
The model was trained using supervised fine-tuning on the messages schema:

```json
{{"messages":[
  {{"role":"system","content":"..."}},
  {{"role":"user","content":"..."}},
  {{"role":"assistant","content":"..."}}
]}}
```
"""


def export_full_model(
    model_dir: str | pathlib.Path,
    base_model_id: str,
    output_dir: str | pathlib.Path,
    extra_metadata: dict = None,
    progress_callback=None,
) -> tuple[bool, str]:
    """
    Package a full SFT model directory into a ZIP archive.
    Returns (success, zip_path_or_error).
    """
    model_dir = pathlib.Path(model_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        return False, f"Model directory not found: {model_dir}"

    timestamp = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    safe_name = model_dir.name.replace(" ", "_")
    zip_path = output_dir / f"{safe_name}_full_{timestamp}.zip"

    try:
        total_bytes = sum(
            f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
        )
        if progress_callback:
            progress_callback(
                f"Exporting full model ({total_bytes // 1024 // 1024} MB)..."
            )

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in model_dir.rglob("*"):
                if f.is_file() and not f.name.startswith("_train_"):
                    zf.write(f, f.relative_to(model_dir.parent))

            readme = README_FULL.format(
                exported_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
                base_model=base_model_id,
            )
            zf.writestr("README.md", readme)

            meta = {
                "exported_at": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
                "base_model": base_model_id,
                "model_dir": str(model_dir),
                "method": "full_sft",
                "model_type": "full_model",
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
