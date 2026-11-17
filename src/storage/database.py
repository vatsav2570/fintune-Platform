"""
src/storage/database.py
─────────────────────────────────────────────────────────────────
SQLite schema initialisation and low-level helpers.
All application persistence goes through this module.
"""

import sqlite3
import contextlib
import pathlib
from src.config import DB_PATH


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextlib.contextmanager
def get_db():
    conn = _conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't already exist."""
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            source_files    TEXT,           -- JSON list
            requested_count INTEGER,
            accepted_count  INTEGER,
            duplicate_count INTEGER,
            rejected_count  INTEGER,
            schema_type     TEXT DEFAULT 'messages',
            generation_settings TEXT,       -- JSON blob
            validation_status TEXT DEFAULT 'pending',
            dataset_dir     TEXT,
            jsonl_path      TEXT,
            metadata_path   TEXT,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS base_models (
            id              TEXT PRIMARY KEY,
            hf_model_id     TEXT NOT NULL,
            revision        TEXT DEFAULT 'main',
            local_dir       TEXT,
            download_status TEXT DEFAULT 'pending',
            downloaded_at   TEXT,
            config_json     TEXT,
            disk_bytes      INTEGER DEFAULT 0,
            architecture    TEXT,
            param_count     TEXT,
            license         TEXT,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS fine_tuned_models (
            id              TEXT PRIMARY KEY,
            display_name    TEXT NOT NULL,
            base_model_id   TEXT REFERENCES base_models(id),
            dataset_id      TEXT REFERENCES datasets(id),
            training_method TEXT,           -- 'qlora_sft' | 'lora_sft' | 'full_sft'
            created_at      TEXT NOT NULL,
            training_status TEXT DEFAULT 'pending',
            output_dir      TEXT,
            adapter_path    TEXT,
            model_type      TEXT,           -- 'adapter' | 'merged' | 'full'
            export_type     TEXT,
            disk_bytes      INTEGER DEFAULT 0,
            final_loss      REAL,
            training_steps  INTEGER,
            training_run_id TEXT,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id              TEXT PRIMARY KEY,
            fine_tuned_model_id TEXT REFERENCES fine_tuned_models(id),
            started_at      TEXT,
            finished_at     TEXT,
            status          TEXT DEFAULT 'pending',
            config_json     TEXT,
            log_path        TEXT,
            checkpoint_dir  TEXT,
            error_message   TEXT,
            metrics_json    TEXT
        );

        CREATE TABLE IF NOT EXISTS app_settings (
            key             TEXT PRIMARY KEY,
            value           TEXT
        );
        """)
