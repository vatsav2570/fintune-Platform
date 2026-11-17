"""
src/inference/model_loader.py
──────────────────────────────
Load a fine-tuned model (adapter or full) for inference.
Wraps the logic from inference.py / inference_9800.py.
"""

from __future__ import annotations
import gc
import logging
import pathlib
from typing import Optional

logger = logging.getLogger(__name__)

# Singleton: currently loaded model
_loaded: dict = {}


def is_torch_available() -> bool:
    """Return True if torch is importable in the current environment."""
    try:
        import torch
        return True
    except ImportError:
        return False


def is_loaded() -> bool:
    return bool(_loaded.get("model"))


def get_loaded_model_id() -> str:
    return _loaded.get("ft_model_id", "")


def load_model(
    base_model_path: str,
    adapter_path: str = "",
    model_type: str = "adapter",   # "adapter" | "full_model"
    ft_model_id: str = "",
    load_in_4bit: bool = True,
    bf16: bool = True,
    device_map: str = "cuda:0",
    progress_callback=None,
) -> tuple[bool, str]:
    """
    Load a model for inference.
    For adapters: loads base model (4-bit) + PeftModel.
    For full models: loads the saved checkpoint directly.
    """
    global _loaded

    if _loaded.get("model") and _loaded.get("ft_model_id") == ft_model_id:
        return True, "Model already loaded."

    # Unload previous model first
    unload_model()

    def _cb(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed in the active environment. Please run: pip install torch"

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        _cb(f"Loading tokenizer from {base_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        if model_type == "adapter" and adapter_path:
            _cb("Loading base model in 4-bit...")
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_cfg,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            # PeftModel needs the adapter DIRECTORY (containing adapter_config.json),
            # not the individual .safetensors file path.
            adapter_dir = adapter_path
            if pathlib.Path(adapter_path).is_file():
                adapter_dir = str(pathlib.Path(adapter_path).parent)
            _cb(f"Loading LoRA adapter from {adapter_dir}...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_dir)
        else:
            # Full model or merged model
            _cb(f"Loading model from {base_model_path}...")
            dtype = torch.bfloat16 if bf16 else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="eager",
            )

        model.eval()
        _loaded = {
            "model": model,
            "tokenizer": tokenizer,
            "ft_model_id": ft_model_id,
            "base_model_path": base_model_path,
            "adapter_path": adapter_path,
        }
        _cb("✅ Model ready!")
        return True, "Model loaded successfully."

    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        unload_model()
        return False, f"Failed to load model: {exc}"


def unload_model():
    """Free GPU memory from the loaded model."""
    global _loaded
    model = _loaded.get("model")
    if model is not None:
        try:
            del model
            _loaded.clear()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Model unloaded and VRAM freed.")
        except Exception as exc:
            logger.warning("Unload warning: %s", exc)
    _loaded = {}


def get_loaded() -> tuple:
    """Return (model, tokenizer) or (None, None)."""
    return _loaded.get("model"), _loaded.get("tokenizer")
