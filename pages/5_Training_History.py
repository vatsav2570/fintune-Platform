"""
pages/5_Training_History.py
────────────────────────────────────────────────────────────────
Training history, log viewer, and application settings.
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import streamlit as st

from src.storage.project_repository import list_training_runs, get_training_run, list_datasets
from src.training.training_worker import tail_log
from src.config import (
    AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION,
    BASE_MODELS_DIR, DATASETS_DIR, FINE_TUNED_MODELS_DIR, EXPORTS_DIR, LOGS_DIR,
)
from src.theme import apply_theme

st.set_page_config(page_title="Training History & Settings", page_icon="📈", layout="wide")

# ── Global CSS & Theme ───────────────────────────────────────
apply_theme()

st.markdown('<div class="page-title">📈 Training History & Settings</div>', unsafe_allow_html=True)

tab_hist, tab_datasets, tab_settings = st.tabs(["📊 Training Runs", "📁 Datasets", "⚙️ Settings"])

# ═══════════════════════════════════════════════════════
# TAB 1: Training History
# ═══════════════════════════════════════════════════════
with tab_hist:
    st.markdown('<div class="section-header">All Training Runs</div>', unsafe_allow_html=True)

    runs = list_training_runs()
    if not runs:
        st.info("No training runs recorded yet.")
    else:
        for run in runs:
            status = run.get("status", "unknown")
            badge_cls = {"complete": "badge-complete", "failed": "badge-failed",
                         "running": "badge-running"}.get(status, "badge-pending")
            model_name = run.get("model_name", "Unknown")
            started = run.get("started_at", "")[:16]
            finished = run.get("finished_at", "")[:16] if run.get("finished_at") else "—"

            with st.expander(
                f"**{model_name}** [{status.upper()}] — {started}",
                expanded=False,
            ):
                st.markdown(
                    f"<span class='{badge_cls}'>{status}</span> &nbsp; "
                    f"Started: **{started}** &nbsp; Finished: **{finished}**",
                    unsafe_allow_html=True,
                )

                if run.get("error_message"):
                    st.error(f"Error: {run['error_message']}")

                if run.get("metrics_json"):
                    try:
                        metrics = json.loads(run["metrics_json"])
                        st.json(metrics)
                    except Exception:
                        pass

                if run.get("config_json"):
                    with st.expander("Training Config"):
                        try:
                            cfg = json.loads(run["config_json"])
                            # Redact sensitive info
                            safe_cfg = {k: v for k, v in cfg.items() if "key" not in k.lower()}
                            st.json(safe_cfg)
                        except Exception:
                            st.text(run["config_json"])

                # Log viewer
                log_path = run.get("log_path")
                if log_path and pathlib.Path(log_path).exists():
                    if st.button("📄 View Log", key=f"log_{run['id']}"):
                        lines = tail_log(log_path, 200)
                        st.code("\n".join(lines), language="text")

# ═══════════════════════════════════════════════════════
# TAB 2: Datasets
# ═══════════════════════════════════════════════════════
with tab_datasets:
    st.markdown('<div class="section-header">Registered Datasets</div>', unsafe_allow_html=True)

    datasets = list_datasets()
    if not datasets:
        st.info("No datasets registered yet.")
    else:
        for ds in datasets:
            with st.expander(f"**{ds['name']}** — {ds.get('accepted_count', '?')} examples"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Accepted", ds.get("accepted_count", 0))
                c2.metric("Duplicates", ds.get("duplicate_count", 0))
                c3.metric("Rejected", ds.get("rejected_count", 0))

                st.caption(f"Created: {ds.get('created_at','')[:16]}")
                st.caption(f"JSONL: `{ds.get('jsonl_path','—')}`")
                st.caption(f"Status: {ds.get('validation_status','—')}")

                if ds.get("source_files"):
                    try:
                        files = json.loads(ds["source_files"]) if isinstance(ds["source_files"], str) else ds["source_files"]
                        st.caption(f"Sources: {', '.join(files)}")
                    except Exception:
                        pass

                if ds.get("generation_settings"):
                    try:
                        gs = json.loads(ds["generation_settings"]) if isinstance(ds["generation_settings"], str) else ds["generation_settings"]
                        st.caption(f"Settings: Q-types={gs.get('question_types')}, difficulty={gs.get('difficulty')}")
                    except Exception:
                        pass

                # Download existing dataset
                jsonl_path = ds.get("jsonl_path")
                if jsonl_path and pathlib.Path(jsonl_path).exists():
                    with open(jsonl_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download JSONL",
                            data=f.read(),
                            file_name=pathlib.Path(jsonl_path).name,
                            mime="application/jsonlines",
                            key=f"dl_ds_{ds['id']}",
                        )

# ═══════════════════════════════════════════════════════
# TAB 3: Settings
# ═══════════════════════════════════════════════════════
with tab_settings:
    st.markdown('<div class="section-header">Application Settings</div>', unsafe_allow_html=True)

    st.markdown("### 🔐 Azure OpenAI")
    st.markdown("""
    > Configuration is loaded from your `.env` file. API keys are **never** displayed here.
    > Edit `.env` and restart Streamlit to apply changes.
    """)
    col_a, col_b = st.columns(2)
    col_a.metric("Endpoint", AZURE_ENDPOINT[:40] + "..." if len(AZURE_ENDPOINT) > 40 else AZURE_ENDPOINT)
    col_b.metric("Deployment", AZURE_DEPLOYMENT)
    col_a.metric("API Version", AZURE_API_VERSION)
    col_b.metric("API Key", "✅ Set" if __import__("os").getenv("AZURE_OPENAI_API_KEY") else "❌ Not set")

    st.markdown("---")
    st.markdown("### 📁 Storage Directories")
    for label, path in [
        ("Datasets",         DATASETS_DIR),
        ("Base Models",      BASE_MODELS_DIR),
        ("Fine-Tuned Models", FINE_TUNED_MODELS_DIR),
        ("Exports",          EXPORTS_DIR),
        ("Logs",             LOGS_DIR),
    ]:
        p = pathlib.Path(path)
        size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) // 1024 // 1024
        st.markdown(f"""
        <div class="info-card">
            <b>{label}</b><br>
            <code>{path}</code><br>
            <span style='color:#64748b;'>Disk usage: {size_mb} MB</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **AI Dataset & Fine-Tuning Platform v1.0**

    Built on top of the existing QLoRA fine-tuning pipeline for Qwen/Llama/Mistral models.

    **Existing files preserved and unchanged:**
    - `train.py` — original QLoRA trainer
    - `train_new.py` — CodeLlama variant
    - `inference.py` / `inference_9800.py` — CLI inference scripts
    - `prepare_data.py` / `generate_dataset.py` — original data scripts
    - All `.jsonl` datasets and adapter directories

    **Dataset schema:** All generated datasets use the `messages` format (system/user/assistant) compatible with the existing `SFTTrainer` pipeline.
    """)
