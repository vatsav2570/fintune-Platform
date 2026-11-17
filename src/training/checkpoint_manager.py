"""
src/training/checkpoint_manager.py
────────────────────────────────────
Utilities for inspecting and managing training checkpoints.
"""

import json
import pathlib
import re
from typing import Optional


def list_checkpoints(output_dir: str | pathlib.Path) -> list[dict]:
    """
    Scan an output directory for checkpoint subdirectories.
    Returns a sorted list of checkpoint info dicts.
    """
    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists():
        return []

    checkpoints = []
    for d in output_dir.iterdir():
        if d.is_dir() and re.match(r"checkpoint-\d+", d.name):
            step = int(d.name.split("-")[1])
            size_mb = sum(
                f.stat().st_size for f in d.rglob("*") if f.is_file()
            ) // 1024 // 1024
            trainer_state = d / "trainer_state.json"
            loss = None
            if trainer_state.exists():
                try:
                    ts = json.loads(trainer_state.read_text(encoding="utf-8"))
                    log_history = ts.get("log_history", [])
                    # Get last training loss from log history
                    for entry in reversed(log_history):
                        if "loss" in entry:
                            loss = round(entry["loss"], 4)
                            break
                except Exception:
                    pass
            checkpoints.append({
                "name": d.name,
                "step": step,
                "path": str(d),
                "size_mb": size_mb,
                "loss": loss,
            })
    return sorted(checkpoints, key=lambda x: x["step"])


def get_latest_checkpoint(output_dir: str | pathlib.Path) -> Optional[str]:
    """Return path to the most recent checkpoint, or None."""
    ckpts = list_checkpoints(output_dir)
    if not ckpts:
        return None
    return ckpts[-1]["path"]


def training_complete_marker(output_dir: str | pathlib.Path) -> bool:
    """Return True if the training completion marker exists."""
    marker = pathlib.Path(output_dir) / "_training_complete.json"
    return marker.exists()


def read_completion_status(output_dir: str | pathlib.Path) -> Optional[dict]:
    """Read and return the completion marker contents."""
    marker = pathlib.Path(output_dir) / "_training_complete.json"
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None


def estimate_training_progress(log_path: str | pathlib.Path) -> dict:
    """
    Parse training log to extract current epoch/step/loss.
    Returns dict with 'epoch', 'step', 'loss' keys where found.
    """
    result: dict = {"epoch": None, "step": None, "loss": None}
    p = pathlib.Path(log_path)
    if not p.exists():
        return result

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            # SFTTrainer logs look like: {'loss': 1.234, 'epoch': 1.5}
            match = re.search(r"\{[^}]*'loss'[^}]*\}", line)
            if match:
                try:
                    data = json.loads(match.group().replace("'", '"'))
                    result["loss"] = data.get("loss")
                    result["epoch"] = data.get("epoch")
                    result["step"] = data.get("step")
                    break
                except Exception:
                    pass
    except Exception:
        pass

    return result
