"""
src/training/training_worker.py
────────────────────────────────
Execute a training job in a background subprocess.
The Streamlit UI reads progress from a shared log file.
Training runs are isolated: each has its own output directory and log.

The training logic is self-contained in this file and mirrors
the existing train.py / train_new.py pipeline closely.
"""

import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import threading
from typing import Optional, Callable

from src.training.trainer_interface import TrainingConfig

logger = logging.getLogger(__name__)

# Per-session tracking of active subprocess
_active_proc: Optional[subprocess.Popen] = None
_active_lock = threading.Lock()


# ── Launch ────────────────────────────────────────────────────

def launch_training(config: TrainingConfig) -> tuple[bool, str]:
    """
    Launch a training job as a subprocess.
    Returns (success, message).
    Prevents duplicate concurrent jobs.
    """
    global _active_proc
    with _active_lock:
        if _active_proc is not None and _active_proc.poll() is None:
            return False, "A training job is already running. Wait for it to finish or cancel it."

    errors = config.validate()
    if errors:
        return False, "Configuration errors:\n" + "\n".join(f"• {e}" for e in errors)

    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.log_path).parent.mkdir(parents=True, exist_ok=True)

    # Serialise config to a temp JSON file the subprocess reads
    cfg_file = pathlib.Path(config.output_dir) / "_train_config.json"
    cfg_file.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    # Build subprocess command (runs inside the project venv)
    venv_python = _find_python()
    script = pathlib.Path(__file__).parent / "_train_subprocess.py"

    cmd = [venv_python, str(script), str(cfg_file)]
    log_file = open(config.log_path, "w", buffering=1)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(pathlib.Path(__file__).resolve().parent.parent.parent),
            env={**os.environ},
        )
    except Exception as exc:
        log_file.close()
        return False, f"Failed to start training process: {exc}"

    # Parent process closes its copy of the log file handle;
    # the subprocess has its own inherited handle.
    log_file.close()

    with _active_lock:
        _active_proc = proc

    logger.info("Training subprocess PID %d started", proc.pid)
    return True, f"Training started (PID {proc.pid}). Logs: {config.log_path}"


def get_process_status() -> dict:
    """Return status of the active training subprocess."""
    with _active_lock:
        if _active_proc is None:
            return {"running": False, "pid": None, "returncode": None}
        rc = _active_proc.poll()
        return {
            "running": rc is None,
            "pid": _active_proc.pid,
            "returncode": rc,
        }


def cancel_training() -> str:
    """Attempt to terminate the active training subprocess."""
    global _active_proc
    with _active_lock:
        if _active_proc is None or _active_proc.poll() is not None:
            return "No active training job to cancel."
        _active_proc.terminate()
        try:
            _active_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _active_proc.kill()
        _active_proc = None
    return "Training job cancelled."


def tail_log(log_path: str, n_lines: int = 100) -> list[str]:
    """Return the last n lines of the training log."""
    p = pathlib.Path(log_path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n_lines:]
    except Exception:
        return []


def _find_python() -> str:
    """Find the Python interpreter (prefer venv)."""
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    candidates = [
        root / "venv" / "Scripts" / "python.exe",   # Windows
        root / "venv" / "bin" / "python",            # Unix/macOS
        root / "venv" / "bin" / "python3",           # Unix/macOS alt
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable
