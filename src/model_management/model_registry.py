"""
src/model_management/model_registry.py
──────────────────────────────────────
Auto-discover existing adapters in the project root and in workspace/
and register them in the SQLite registry on first launch.
"""

import json
import logging
import pathlib
from src.config import ROOT_DIR, FINE_TUNED_MODELS_DIR
from src.storage.project_repository import (
    upsert_base_model,
    register_fine_tuned_model,
    list_fine_tuned_models,
    update_fine_tuned_model,
    get_base_model_by_hf_id,
)

logger = logging.getLogger(__name__)

# Adapter directories that exist in the existing project root
_KNOWN_ADAPTERS = [
    {
        "dir": ROOT_DIR / "qwen25-0.5b-9800-qlora",
        "name": "Qwen2.5-0.5B 9800 Series QLoRA v1",
        "base": "Qwen/Qwen2.5-0.5B-Instruct",
        "method": "qlora_sft",
    },
    {
        "dir": ROOT_DIR / "qwen25-0.5b-9800v2-qlora",
        "name": "Qwen2.5-0.5B 9800 Series QLoRA v2",
        "base": "Qwen/Qwen2.5-0.5B-Instruct",
        "method": "qlora_sft",
    },
]


def _load_adapter_config(adapter_dir: pathlib.Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {}


def auto_discover_adapters():
    """
    Scan for known adapter directories and register them in the DB
    if they are not already present.
    Only runs once (checks existing registry first).
    """
    existing = {m["output_dir"] for m in list_fine_tuned_models()}

    for entry in _KNOWN_ADAPTERS:
        d: pathlib.Path = entry["dir"]
        if not d.exists() or not (d / "adapter_config.json").exists():
            continue
        if str(d) in existing:
            continue

        adapter_cfg = _load_adapter_config(d)
        base_hf_id = entry["base"]

        # Ensure base model is registered (may not be downloaded locally)
        bm = get_base_model_by_hf_id(base_hf_id)
        if bm is None:
            bm_id = upsert_base_model(
                hf_model_id=base_hf_id,
                download_status="unknown",
                architecture="Qwen2ForCausalLM",
            )
        else:
            bm_id = bm["id"]

        ft_id = register_fine_tuned_model(
            display_name=entry["name"],
            base_model_id=bm_id,
            dataset_id="",
            training_method=entry["method"],
            output_dir=str(d),
            adapter_path=str(d / "adapter_model.safetensors"),
            model_type="adapter",
        )
        update_fine_tuned_model(ft_id, training_status="complete")
        logger.info("Auto-discovered adapter: %s → %s", d.name, ft_id)


def scan_workspace_adapters():
    """Scan workspace/fine_tuned_models for any adapter directories."""
    existing_dirs = {m["output_dir"] for m in list_fine_tuned_models()}
    for candidate in FINE_TUNED_MODELS_DIR.iterdir():
        if not candidate.is_dir():
            continue
        cfg_path = candidate / "adapter_config.json"
        if not cfg_path.exists():
            continue
        if str(candidate) in existing_dirs:
            continue
        adapter_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        base_hf = adapter_cfg.get("base_model_name_or_path", "unknown")
        bm = get_base_model_by_hf_id(base_hf)
        bm_id = bm["id"] if bm else upsert_base_model(
            hf_model_id=base_hf, download_status="unknown"
        )
        ft_id = register_fine_tuned_model(
            display_name=candidate.name,
            base_model_id=bm_id,
            dataset_id="",
            training_method="qlora_sft",
            output_dir=str(candidate),
            adapter_path=str(candidate / "adapter_model.safetensors"),
            model_type="adapter",
        )
        update_fine_tuned_model(ft_id, training_status="complete")
        logger.info("Workspace adapter discovered: %s", candidate.name)
