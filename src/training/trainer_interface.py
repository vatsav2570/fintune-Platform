"""
src/training/trainer_interface.py
──────────────────────────────────
Common interface / dataclass for all training configurations.
Converts user-selected settings into the arguments passed to the
training worker subprocess.
"""

from __future__ import annotations
import dataclasses
from typing import Optional, Literal


TrainingMethod = Literal["qlora_sft", "lora_sft", "full_sft"]


@dataclasses.dataclass
class TrainingConfig:
    # Identity
    method: TrainingMethod = "qlora_sft"
    fine_tuned_model_id: str = ""
    display_name: str = ""
    training_run_id: str = ""

    # Paths
    base_model_path: str = ""   # local path or HF model id
    dataset_path: str = ""      # .jsonl path
    output_dir: str = ""
    log_path: str = ""
    checkpoint_dir: str = ""

    # Sequence
    max_seq_len: int = 2048

    # Batch / optimisation
    batch_size: int = 4
    grad_accum: int = 4
    epochs: int = 3
    lr: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optim: str = "paged_adamw_32bit"
    lr_scheduler: str = "cosine"

    # Checkpointing / logging
    save_steps: int = 50
    log_steps: int = 5
    save_total_limit: int = 3

    # Precision
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # LoRA / QLoRA specific
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Optional[list[str]] = None

    # QLoRA specific
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_double_quant: bool = True

    # Hardware
    device_map: str = "cuda:0"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors = []
        import pathlib
        if not self.base_model_path:
            errors.append("base_model_path is required")
        if not self.dataset_path:
            errors.append("dataset_path is required")
        if not pathlib.Path(self.dataset_path).exists():
            errors.append(f"Dataset file not found: {self.dataset_path}")
        if not self.output_dir:
            errors.append("output_dir is required")
        if self.epochs < 1:
            errors.append("epochs must be >= 1")
        if self.lr <= 0:
            errors.append("learning_rate must be > 0")
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.method in ("qlora_sft", "lora_sft") and self.lora_r < 1:
            errors.append("lora_r must be >= 1 for LoRA/QLoRA")
        return errors


METHOD_DESCRIPTIONS = {
    "qlora_sft": {
        "name": "QLoRA — Quantized LoRA SFT",
        "short": "4-bit quantized base model + LoRA adapters. Lowest VRAM requirement.",
        "description": (
            "QLoRA loads the base model in 4-bit NormalFloat precision and trains "
            "small LoRA adapter matrices. This is the recommended method for consumer "
            "and mid-range GPUs. Only the adapter (~1-2% of parameters) is updated. "
            "The saved output is a small adapter file, not the full model weights."
        ),
        "output_type": "adapter",
        "saves_full_weights": False,
        "supported_architectures": "Most causal LMs",
        "memory_note": "~0.6 bytes/param for 4-bit base + activations",
    },
    "lora_sft": {
        "name": "LoRA — Standard LoRA SFT",
        "short": "bf16 base model + LoRA adapters. Higher quality than QLoRA.",
        "description": (
            "LoRA keeps the base model in bf16 and trains adapter matrices. "
            "Produces better results than QLoRA but requires more VRAM. "
            "The saved output is a small adapter file."
        ),
        "output_type": "adapter",
        "saves_full_weights": False,
        "supported_architectures": "Most causal LMs",
        "memory_note": "~2 bytes/param for bf16 base + optimizer states",
    },
    "full_sft": {
        "name": "Full SFT — All Parameters",
        "short": "All model parameters updated. Highest VRAM requirement.",
        "description": (
            "All model parameters are updated during training. "
            "Produces the most capable result but requires very large VRAM "
            "and takes longer. The saved output is a complete model checkpoint."
        ),
        "output_type": "full_model",
        "saves_full_weights": True,
        "supported_architectures": "Most causal LMs (large VRAM required)",
        "memory_note": "~8 bytes/param including optimizer states",
    },
}
