"""
src/storage/project_repository.py
─────────────────────────────────────────────────────────────────
High-level CRUD helpers used by Streamlit pages.
All DB interaction goes through these functions.
"""

import json
import uuid
import datetime
import pathlib
from typing import Optional

from src.storage.database import get_db, init_db


# ── Utility ───────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()

def _uid() -> str:
    return str(uuid.uuid4())


# ── DATASETS ──────────────────────────────────────────────────

def create_dataset(
    name: str,
    source_files: list[str],
    requested_count: int,
    generation_settings: dict,
    dataset_dir: str,
) -> str:
    ds_id = _uid()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO datasets
               (id,name,created_at,source_files,requested_count,
                generation_settings,dataset_dir)
               VALUES (?,?,?,?,?,?,?)""",
            (ds_id, name, _now(), json.dumps(source_files),
             requested_count, json.dumps(generation_settings), dataset_dir),
        )
    return ds_id


def update_dataset_counts(
    ds_id: str,
    accepted: int,
    duplicates: int,
    rejected: int,
    jsonl_path: str,
    metadata_path: str,
    status: str = "complete",
):
    with get_db() as conn:
        conn.execute(
            """UPDATE datasets SET
               accepted_count=?, duplicate_count=?, rejected_count=?,
               jsonl_path=?, metadata_path=?, validation_status=?
               WHERE id=?""",
            (accepted, duplicates, rejected, jsonl_path, metadata_path, status, ds_id),
        )


def list_datasets() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_dataset(ds_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE id=?", (ds_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_dataset(ds_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


# ── BASE MODELS ───────────────────────────────────────────────

def upsert_base_model(
    hf_model_id: str,
    local_dir: str = "",
    download_status: str = "pending",
    config_json: dict = None,
    disk_bytes: int = 0,
    architecture: str = "",
    param_count: str = "",
    license: str = "",
    revision: str = "main",
) -> str:
    """Insert or update a base model record. Returns model id."""
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM base_models WHERE hf_model_id=? AND revision=?",
            (hf_model_id, revision),
        ).fetchone()
        if existing:
            model_id = existing["id"]
            conn.execute(
                """UPDATE base_models SET
                   local_dir=?, download_status=?, config_json=?,
                   disk_bytes=?, architecture=?, param_count=?, license=?
                   WHERE id=?""",
                (local_dir, download_status,
                 json.dumps(config_json) if config_json else None,
                 disk_bytes, architecture, param_count, license, model_id),
            )
        else:
            model_id = _uid()
            conn.execute(
                """INSERT INTO base_models
                   (id,hf_model_id,revision,local_dir,download_status,
                    downloaded_at,config_json,disk_bytes,architecture,param_count,license)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (model_id, hf_model_id, revision, local_dir, download_status,
                 _now() if download_status == "complete" else None,
                 json.dumps(config_json) if config_json else None,
                 disk_bytes, architecture, param_count, license),
            )
    return model_id


def list_base_models() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM base_models ORDER BY downloaded_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_base_model_by_hf_id(hf_model_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM base_models WHERE hf_model_id=? ORDER BY downloaded_at DESC LIMIT 1",
            (hf_model_id,),
        ).fetchone()
    return dict(row) if row else None


# ── FINE-TUNED MODELS ─────────────────────────────────────────

def register_fine_tuned_model(
    display_name: str,
    base_model_id: str,
    dataset_id: str,
    training_method: str,
    output_dir: str,
    adapter_path: str = "",
    model_type: str = "adapter",
) -> str:
    ft_id = _uid()
    # Convert empty string to None for optional FK
    ds_id_val = dataset_id if dataset_id else None
    with get_db() as conn:
        conn.execute(
            """INSERT INTO fine_tuned_models
               (id,display_name,base_model_id,dataset_id,training_method,
                created_at,training_status,output_dir,adapter_path,model_type)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (ft_id, display_name, base_model_id, ds_id_val, training_method,
             _now(), "pending", output_dir, adapter_path, model_type),
        )
    return ft_id


def update_fine_tuned_model(ft_id: str, **kwargs):
    allowed = {"training_status", "adapter_path", "disk_bytes",
               "final_loss", "training_steps", "training_run_id",
               "export_type", "notes"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    cols = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [ft_id]
    with get_db() as conn:
        conn.execute(f"UPDATE fine_tuned_models SET {cols} WHERE id=?", values)


def list_fine_tuned_models() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT ft.*, bm.hf_model_id as base_hf_id,
                      ds.name as dataset_name
               FROM fine_tuned_models ft
               LEFT JOIN base_models bm ON ft.base_model_id = bm.id
               LEFT JOIN datasets ds ON ft.dataset_id = ds.id
               ORDER BY ft.created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_fine_tuned_model(ft_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            """SELECT ft.*, bm.hf_model_id as base_hf_id,
                      bm.local_dir as base_local_dir,
                      ds.name as dataset_name
               FROM fine_tuned_models ft
               LEFT JOIN base_models bm ON ft.base_model_id = bm.id
               LEFT JOIN datasets ds ON ft.dataset_id = ds.id
               WHERE ft.id=?""",
            (ft_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_fine_tuned_model(ft_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM fine_tuned_models WHERE id=?", (ft_id,))


# ── TRAINING RUNS ─────────────────────────────────────────────

def create_training_run(
    fine_tuned_model_id: str,
    config: dict,
    log_path: str,
    checkpoint_dir: str,
) -> str:
    run_id = _uid()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO training_runs
               (id,fine_tuned_model_id,started_at,status,
                config_json,log_path,checkpoint_dir)
               VALUES (?,?,?,?,?,?,?)""",
            (run_id, fine_tuned_model_id, _now(), "running",
             json.dumps(config), log_path, checkpoint_dir),
        )
    return run_id


def finish_training_run(
    run_id: str,
    status: str,
    error_message: str = "",
    metrics: dict = None,
):
    with get_db() as conn:
        conn.execute(
            """UPDATE training_runs SET
               finished_at=?, status=?, error_message=?, metrics_json=?
               WHERE id=?""",
            (_now(), status, error_message,
             json.dumps(metrics) if metrics else None, run_id),
        )


def list_training_runs() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT tr.*, ft.display_name as model_name
               FROM training_runs tr
               LEFT JOIN fine_tuned_models ft ON tr.fine_tuned_model_id = ft.id
               ORDER BY tr.started_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_training_run(run_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM training_runs WHERE id=?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


# ── STATISTICS ────────────────────────────────────────────────

def get_dashboard_stats() -> dict:
    with get_db() as conn:
        n_datasets = conn.execute(
            "SELECT COUNT(*) FROM datasets"
        ).fetchone()[0]
        n_base_models = conn.execute(
            "SELECT COUNT(*) FROM base_models WHERE download_status='complete'"
        ).fetchone()[0]
        n_runs = conn.execute(
            "SELECT COUNT(*) FROM training_runs WHERE status='complete'"
        ).fetchone()[0]
        n_ft_models = conn.execute(
            "SELECT COUNT(*) FROM fine_tuned_models WHERE training_status='complete'"
        ).fetchone()[0]
        recent_runs = conn.execute(
            """SELECT tr.id, tr.started_at, tr.status, ft.display_name as model_name
               FROM training_runs tr
               LEFT JOIN fine_tuned_models ft ON tr.fine_tuned_model_id = ft.id
               ORDER BY tr.started_at DESC LIMIT 5"""
        ).fetchall()
    return {
        "n_datasets": n_datasets,
        "n_base_models": n_base_models,
        "n_successful_runs": n_runs,
        "n_ft_models": n_ft_models,
        "recent_runs": [dict(r) for r in recent_runs],
    }
