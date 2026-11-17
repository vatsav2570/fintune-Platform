"""
src/model_management/model_downloader.py
─────────────────────────────────────────
Download base models from Hugging Face Hub with progress tracking.
Uses a stable directory under workspace/base_models/.
"""

from __future__ import annotations
import logging
import pathlib
import shutil
from typing import Callable, Optional

from src.config import BASE_MODELS_DIR, HF_TOKEN

logger = logging.getLogger(__name__)


def _model_local_dir(model_id: str, revision: str = "main") -> pathlib.Path:
    """Deterministic local path for a model."""
    safe = model_id.replace("/", "__")
    return BASE_MODELS_DIR / f"{safe}@{revision}"


def is_model_available(model_id: str, revision: str = "main") -> bool:
    """Return True if the model's local directory exists and has a config.json."""
    local_dir = _model_local_dir(model_id, revision)
    return (local_dir / "config.json").exists()


def get_local_path(model_id: str, revision: str = "main") -> str:
    """Return local path (str) for a downloaded model."""
    return str(_model_local_dir(model_id, revision))


def check_disk_space(required_bytes: int) -> tuple[bool, int]:
    """Return (ok, free_bytes) for workspace partition."""
    stat = shutil.disk_usage(str(BASE_MODELS_DIR))
    return stat.free >= required_bytes, stat.free


def download_model(
    model_id: str,
    revision: str = "main",
    progress_callback: Optional[Callable[[str], None]] = None,
    token: str = HF_TOKEN,
) -> tuple[bool, str]:
    """
    Download a model to the local workspace.
    Returns (success, message_or_path).
    Never logs or surfaces the token.
    """
    local_dir = _model_local_dir(model_id, revision)

    if is_model_available(model_id, revision):
        msg = f"Model already available at {local_dir}"
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)
        return True, str(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        if progress_callback:
            progress_callback(f"Starting download of {model_id} ...")

        snap_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(local_dir),
            token=token or None,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )

        if progress_callback:
            progress_callback(f"✅ Downloaded to {snap_path}")

        return True, snap_path

    except Exception as exc:
        err = str(exc)
        # Scrub token from error message
        if token:
            err = err.replace(token, "***")

        if "gated" in err.lower() or "401" in err or "403" in err:
            msg = (
                f"Access denied for '{model_id}'. "
                "This model may be gated. "
                "Set HF_TOKEN in your .env and ensure you have accepted the model license on huggingface.co."
            )
        elif "not found" in err.lower() or "404" in err:
            msg = f"Model '{model_id}' not found on Hugging Face Hub."
        else:
            msg = f"Download failed: {err}"

        logger.error("Model download failed: %s", msg)
        # Clean up partial download
        if local_dir.exists() and not any(local_dir.iterdir()):
            local_dir.rmdir()

        return False, msg


def get_model_disk_usage(model_id: str, revision: str = "main") -> int:
    """Return total bytes used by the local model directory."""
    local_dir = _model_local_dir(model_id, revision)
    if not local_dir.exists():
        return 0
    total = 0
    for f in local_dir.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total
