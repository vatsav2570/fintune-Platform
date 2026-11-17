"""
pages/3_Fine_Tuned_Models.py
────────────────────────────────────────────────────────────────
Browse, load, export, and delete fine-tuned models.
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.storage.project_repository import (
    list_fine_tuned_models, get_fine_tuned_model,
    delete_fine_tuned_model, update_fine_tuned_model,
)
from src.model_management.model_registry import scan_workspace_adapters
from src.inference.model_loader import (
    load_model, unload_model, is_loaded, get_loaded_model_id, is_torch_available
)
from src.export.adapter_export import export_adapter
from src.export.merged_model_export import merge_and_export
from src.config import EXPORTS_DIR
from src.theme import apply_theme

st.set_page_config(page_title="Fine-Tuned Models", page_icon="🗂️", layout="wide")

# ── Global CSS & Theme ───────────────────────────────────────
apply_theme()

st.markdown('<div class="page-title">🗂️ Fine-Tuned Models</div>', unsafe_allow_html=True)
st.markdown("Manage all fine-tuned models and adapters.")

if not is_torch_available():
    st.info("ℹ️ **PyTorch is not installed in the current environment.** You can browse, review, and export model adapters, but loading models into memory for inference requires PyTorch (`pip install torch`).")

# Rescan workspace
if st.button("🔄 Refresh & Rescan Workspace"):
    scan_workspace_adapters()
    st.rerun()

models = list_fine_tuned_models()

if not models:
    st.info(
        "No fine-tuned models registered yet.\n\n"
        "Fine-tune a model on the **Model Fine-Tuning** page, "
        "or make sure your existing adapter directories are in the project root."
    )
else:
    # Status filter
    status_filter = st.multiselect(
        "Filter by status",
        ["complete", "running", "failed", "pending", "cancelled"],
        default=["complete"],
    )
    shown = [m for m in models if m["training_status"] in status_filter] if status_filter else models

    for m in shown:
        status = m.get("training_status", "unknown")
        model_type = m.get("model_type", "adapter")
        badge_status = {
            "complete": "badge-complete", "failed": "badge-failed",
        }.get(status, "badge-pending")
        badge_type = "badge-adapter" if model_type == "adapter" else "badge-full"

        with st.container():
            st.markdown(f"""
            <div class="model-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-size:1.1rem;font-weight:600;color:#e2e8f0;">
                            {"✨" if model_type=="adapter" else "🤖"} {m['display_name']}
                        </span>
                        &nbsp;
                        <span class="badge {badge_status}">{status}</span>
                        &nbsp;
                        <span class="badge {badge_type}">{model_type}</span>
                    </div>
                    <div style="color:#64748b;font-size:0.85rem;">{m.get('created_at','')[:16]}</div>
                </div>
                <div style="color:#94a3b8;margin-top:0.5rem;font-size:0.9rem;">
                    🤗 Base: <code>{m.get('base_hf_id','unknown')}</code> &nbsp;|&nbsp;
                    📊 Dataset: {m.get('dataset_name','—')} &nbsp;|&nbsp;
                    🔧 Method: {m.get('training_method','—')}
                </div>
                <div style="color:#64748b;font-size:0.8rem;margin-top:0.3rem;">
                    📁 {m.get('output_dir','—')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)

            # ── Load for Playground ──────────────────────────
            with col1:
                currently_loaded = is_loaded() and get_loaded_model_id() == m["id"]
                if currently_loaded:
                    st.success("Loaded ✓")
                elif status == "complete":
                    if st.button("💬 Load", key=f"load_{m['id']}"):
                        # Resolve base model path: prefer downloaded local dir, else HF id
                        base_local = m.get("base_local_dir") or ""
                        base_path = (
                            base_local
                            if base_local and pathlib.Path(base_local).exists()
                            else m.get("base_hf_id", "")
                        )

                        # Adapter dir = output_dir (contains adapter_config.json).
                        # The stored adapter_path may be the .safetensors file; output_dir is safer.
                        if model_type == "adapter":
                            output_dir = m.get("output_dir", "")
                            stored_ap  = m.get("adapter_path", "")
                            # Prefer output_dir if it has adapter_config.json
                            if output_dir and (pathlib.Path(output_dir) / "adapter_config.json").exists():
                                adapter_path = output_dir
                            elif stored_ap and pathlib.Path(stored_ap).is_file():
                                adapter_path = str(pathlib.Path(stored_ap).parent)
                            else:
                                adapter_path = stored_ap
                        else:
                            adapter_path = ""

                        msgs = []
                        log_box = st.empty()
                        def _cb(msg):
                            msgs.append(msg)
                            log_box.info("\n".join(msgs[-4:]))

                        ok, msg = load_model(
                            base_model_path=base_path,
                            adapter_path=adapter_path,
                            model_type=model_type,
                            ft_model_id=m["id"],
                            progress_callback=_cb,
                        )
                        if ok:
                            st.session_state["playground_model_id"] = m["id"]
                            st.success("Model loaded! Go to 💬 Model Playground.")
                        else:
                            st.error(msg)


            # ── Unload ───────────────────────────────────────
            with col2:
                if is_loaded() and get_loaded_model_id() == m["id"]:
                    if st.button("🔓 Unload", key=f"unload_{m['id']}"):
                        unload_model()
                        st.success("Model unloaded.")
                        st.rerun()

            # ── Export Adapter ───────────────────────────────
            with col3:
                if status == "complete" and model_type == "adapter":
                    if st.button("📦 Export", key=f"export_{m['id']}"):
                        with st.spinner("Creating export ZIP..."):
                            ok, result = export_adapter(
                                adapter_dir=m.get("output_dir", ""),
                                base_model_id=m.get("base_hf_id", "unknown"),
                                method=m.get("training_method", "qlora_sft"),
                                output_dir=str(EXPORTS_DIR),
                            )
                        if ok:
                            # Offer download
                            with open(result, "rb") as f:
                                st.download_button(
                                    "⬇️ Download ZIP",
                                    data=f.read(),
                                    file_name=pathlib.Path(result).name,
                                    mime="application/zip",
                                    key=f"dl_{m['id']}",
                                )
                        else:
                            st.error(result)

            # ── Merge & Export ───────────────────────────────
            with col4:
                if status == "complete" and model_type == "adapter" and m.get("training_method") != "qlora_sft":
                    if st.button("🔀 Merge", key=f"merge_{m['id']}"):
                        base_path = m.get("base_local_dir") or m.get("base_hf_id", "")
                        adapter_path = m.get("adapter_path", "")
                        with st.spinner("Merging (this may take a while)..."):
                            ok, result = merge_and_export(
                                base_model_path=base_path,
                                adapter_path=adapter_path,
                                output_dir=str(EXPORTS_DIR),
                            )
                        if ok:
                            st.success(f"Merged model saved to: `{result}`")
                        else:
                            st.error(result)
                elif model_type == "adapter" and m.get("training_method") == "qlora_sft":
                    st.caption("Merge not supported for QLoRA (4-bit quantized).")

            # ── Delete ───────────────────────────────────────
            with col5:
                delete_key = f"confirm_delete_{m['id']}"
                if st.button("🗑️ Delete", key=f"del_{m['id']}"):
                    st.session_state[delete_key] = True
                if st.session_state.get(delete_key):
                    st.warning(f"⚠️ Delete **{m['display_name']}** from registry? (Files on disk are NOT deleted.)")
                    c_yes, c_no = st.columns(2)
                    if c_yes.button("Yes, delete", key=f"yes_del_{m['id']}"):
                        delete_fine_tuned_model(m["id"])
                        st.session_state.pop(delete_key, None)
                        st.success("Removed from registry.")
                        st.rerun()
                    if c_no.button("Cancel", key=f"no_del_{m['id']}"):
                        st.session_state.pop(delete_key, None)
                        st.rerun()

            st.markdown("---")
