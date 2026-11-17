"""
src/model_management/model_compatibility.py
────────────────────────────────────────────
Determine whether a Hugging Face model is compatible with the
existing training pipeline (SFT via TRL/PEFT) and inference backend.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Architectures known to work with the existing pipeline ────
SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "MistralForCausalLM",
    "Phi3ForCausalLM",
    "PhiForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "FalconForCausalLM",
    "GPTNeoXForCausalLM",
    "GPT2LMHeadModel",
    "OPTForCausalLM",
    "BloomForCausalLM",
    "CodeLlamaForCausalLM",
    "StableLmForCausalLM",
    "MixtralForCausalLM",
    "DeepseekV2ForCausalLM",
}

# Standard LoRA attention/MLP target modules by architecture family
LORA_TARGETS: dict[str, list[str]] = {
    "qwen":    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi":     ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "gemma":   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "falcon":  ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "gpt-neo": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "gpt2":    ["c_attn", "c_proj"],
    "opt":     ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "bloom":   ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "default": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


@dataclass
class CompatibilityReport:
    compatible: bool
    architecture: str = ""
    loading_class: str = "AutoModelForCausalLM"
    lora_targets: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_auth: bool = False
    requires_custom_code: bool = False
    quantization_ok: bool = True
    merge_ok: bool = True
    inference_ok: bool = True


def check_model_compatibility(config: dict) -> CompatibilityReport:
    """
    Evaluate a model's config dict (from AutoConfig or Hub API)
    and return a CompatibilityReport.
    """
    report = CompatibilityReport(compatible=True)

    arch_list: list[str] = config.get("architectures") or []
    if not arch_list:
        report.issues.append("No 'architectures' field in model config.")
        report.compatible = False
        return report

    arch = arch_list[0]
    report.architecture = arch

    if arch not in SUPPORTED_ARCHITECTURES:
        report.issues.append(
            f"Architecture '{arch}' is not in the supported list. "
            "Training may still work but has not been tested."
        )
        report.warnings.append(
            "Inspect LoRA target modules manually before training."
        )
        report.compatible = False  # warn user; don't silently proceed

    # Determine LoRA targets
    report.lora_targets = _lora_targets_for(arch)

    # Check if custom code is required
    if config.get("auto_map"):
        report.requires_custom_code = True
        report.warnings.append(
            "Model requires custom code (trust_remote_code=True). "
            "Review the repository before training."
        )

    # Multimodal check
    model_type: str = config.get("model_type", "").lower()
    if any(k in model_type for k in ("vision", "image", "vl", "llava", "clip")):
        report.issues.append(
            "Multimodal model detected. The current pipeline only supports "
            "text-only causal LMs. Multimodal training is not yet supported."
        )
        report.compatible = False
        report.merge_ok = False

    return report


def _lora_targets_for(arch: str) -> list[str]:
    """Return LoRA target module names for the given architecture class."""
    arch_lower = arch.lower()
    for key, targets in LORA_TARGETS.items():
        if key in arch_lower:
            return targets
    return LORA_TARGETS["default"]


def estimate_vram_gb(
    param_count: Optional[int],
    method: str = "qlora",
    sequence_length: int = 2048,
) -> float:
    """
    Rough VRAM estimate in GB.
    Very approximate — actual usage depends on batch size, optimizer, etc.
    """
    if not param_count:
        return 0.0
    params_b = param_count / 1e9
    if method in ("qlora_sft", "qlora"):
        # 4-bit base (~0.5 bytes/param) + adapter (~2% * 2 bytes/param) + activations
        return params_b * 0.6 + 2.0
    elif method in ("lora_sft", "lora"):
        # bf16 base (2 bytes/param) + adapter (small) + optimizer states
        return params_b * 2.5 + 2.0
    else:
        # Full SFT: bf16 model + optimizer states (~4x model size)
        return params_b * 8.0 + 2.0
