"""
src/model_management/huggingface_search.py
──────────────────────────────────────────
Search Hugging Face Hub for candidate base models.
Returns metadata required for the model selection UI.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Curated suggestions shown before any search ───────────────
CURATED_MODELS = [
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "note": "⭐ Used in existing pipeline — fastest to test",
        "approx_size": "~1 GB",
        "license": "Apache 2.0",
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "note": "Small Qwen2.5 — good balance",
        "approx_size": "~3 GB",
        "license": "Apache 2.0",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "note": "Mid-size, strong instruction following",
        "approx_size": "~15 GB",
        "license": "Apache 2.0",
    },
    {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "note": "Tiny Llama 3.2 — gated",
        "approx_size": "~2.5 GB",
        "license": "Llama 3.2 Community",
    },
    {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "note": "Small Llama 3.2 — gated",
        "approx_size": "~6 GB",
        "license": "Llama 3.2 Community",
    },
    {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "note": "Classic Mistral 7B",
        "approx_size": "~14 GB",
        "license": "Apache 2.0",
    },
    {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "note": "Microsoft Phi-3 Mini",
        "approx_size": "~7.6 GB",
        "license": "MIT",
    },
    {
        "id": "google/gemma-2-2b-it",
        "note": "Google Gemma 2 — gated",
        "approx_size": "~5 GB",
        "license": "Gemma",
    },
]


@dataclass
class ModelInfo:
    model_id: str
    author: str = ""
    model_name: str = ""
    pipeline_tag: str = ""
    library_name: str = ""
    tags: list[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    private: bool = False
    gated: bool = False
    license: str = ""
    card_url: str = ""
    param_count: Optional[int] = None
    architecture: str = ""


def search_models(
    query: str = "",
    limit: int = 20,
    filter_tags: list[str] | None = None,
) -> list[ModelInfo]:
    """
    Search Hugging Face Hub for text generation / causal LM models.
    Returns a list of ModelInfo objects.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        hf_filter = filter_tags or ["text-generation"]
        results = api.list_models(
            search=query or None,
            pipeline_tag="text-generation",
            limit=limit,
            sort="downloads",
            direction=-1,
        )
        infos: list[ModelInfo] = []
        for m in results:
            # Extract license from tags
            lic = ""
            for tag in (m.tags or []):
                if tag.startswith("license:"):
                    lic = tag.replace("license:", "")
                    break
            gated = bool(getattr(m, "gated", False))
            infos.append(ModelInfo(
                model_id=m.modelId or "",
                author=m.author or "",
                model_name=(m.modelId or "").split("/")[-1],
                pipeline_tag=m.pipeline_tag or "text-generation",
                library_name=getattr(m, "library_name", "") or "",
                tags=m.tags or [],
                downloads=m.downloads or 0,
                likes=m.likes or 0,
                private=m.private or False,
                gated=gated,
                license=lic,
                card_url=f"https://huggingface.co/{m.modelId}",
            ))
        return infos
    except Exception as exc:
        logger.warning("HF search failed: %s", exc)
        return []


def fetch_model_info(model_id: str) -> Optional[ModelInfo]:
    """Fetch metadata for a specific model ID."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        m = api.model_info(model_id)
        lic = ""
        for tag in (m.tags or []):
            if tag.startswith("license:"):
                lic = tag.replace("license:", "")
                break
        param_count = None
        safetensors_info = getattr(m, "safetensors", None)
        if safetensors_info and hasattr(safetensors_info, "total"):
            param_count = safetensors_info.total
        gated = bool(getattr(m, "gated", False))
        return ModelInfo(
            model_id=m.modelId or model_id,
            author=m.author or "",
            model_name=(m.modelId or model_id).split("/")[-1],
            pipeline_tag=m.pipeline_tag or "",
            library_name=getattr(m, "library_name", "") or "",
            tags=m.tags or [],
            downloads=m.downloads or 0,
            likes=m.likes or 0,
            private=m.private or False,
            gated=gated,
            license=lic,
            card_url=f"https://huggingface.co/{m.modelId or model_id}",
            param_count=param_count,
        )
    except Exception as exc:
        logger.warning("Could not fetch info for %s: %s", model_id, exc)
        return None


def fetch_model_config(model_id: str, token: str = "") -> Optional[dict]:
    """Download and return the config.json as a dict."""
    try:
        from transformers import AutoConfig
        kwargs: dict = {"trust_remote_code": False}
        if token:
            kwargs["token"] = token
        cfg = AutoConfig.from_pretrained(model_id, **kwargs)
        return cfg.to_dict()
    except Exception as exc:
        logger.warning("Could not load config for %s: %s", model_id, exc)
        return None
