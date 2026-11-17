"""
pages/2_Model_Fine_Tuning.py
────────────────────────────────────────────────────────────────
Model selection, compatibility check, download, training config,
and training execution with live log streaming.
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import time
import datetime
import streamlit as st

from src.config import (
    FINE_TUNED_MODELS_DIR, LOGS_DIR, HF_TOKEN,
    DEFAULT_MAX_SEQ_LEN, DEFAULT_BATCH_SIZE, DEFAULT_GRAD_ACCUM,
    DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_WARMUP_RATIO,
    DEFAULT_LORA_R, DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT,
)
from src.storage.project_repository import (
    list_datasets, get_dataset,
    upsert_base_model, get_base_model_by_hf_id,
    register_fine_tuned_model, update_fine_tuned_model,
    create_training_run, finish_training_run,
)
from src.model_management.huggingface_search import (
    search_models, fetch_model_config, CURATED_MODELS,
)
from src.model_management.model_compatibility import (
    check_model_compatibility, estimate_vram_gb,
)
from src.model_management.model_downloader import (
    is_model_available, download_model, get_local_path, get_model_disk_usage,
)
from src.training.trainer_interface import (
    TrainingConfig, METHOD_DESCRIPTIONS,
)
from src.training.training_worker import (
    launch_training, get_process_status, cancel_training, tail_log,
)
from src.theme import apply_theme

st.set_page_config(page_title="Model Fine-Tuning", page_icon="🏋️", layout="wide")

# ── Global CSS & Theme ───────────────────────────────────────
apply_theme()

st.markdown('<div class="page-title">🏋️ Model Fine-Tuning</div>', unsafe_allow_html=True)
st.markdown("Select a base model, configure your training method, and launch fine-tuning.")

# ── Session state ─────────────────────────────────────────────
for k, v in {
    "selected_hf_id": "",
    "selected_local_path": "",
    "selected_dataset_id": "",
    "training_method": "qlora_sft",
    "compat_report": None,
    "ft_model_id": None,
    "training_run_id": None,
    "log_path": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

tabs = st.tabs(["1️⃣ Select Dataset", "2️⃣ Select Base Model", "3️⃣ Configure & Train", "📊 Live Training"])

# ══════════════════════════════════════════════════════════════
# TAB 1: Dataset selection
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Select a Dataset</div>', unsafe_allow_html=True)

    datasets = list_datasets()
    complete_ds = [d for d in datasets if d.get("validation_status") == "complete" and d.get("jsonl_path")]

    if not complete_ds:
        st.info("No saved datasets found. Generate one on the **Dataset Generation** page first.")
        st.markdown("Or enter a custom path to an existing JSONL file:")
        custom_path = st.text_input("Custom JSONL path", placeholder="/path/to/train.jsonl")
        if custom_path and pathlib.Path(custom_path).exists():
            st.session_state["selected_dataset_id"] = "__custom__"
            st.session_state["custom_dataset_path"] = custom_path
            st.success(f"Will use: `{custom_path}`")
    else:
        ds_options = {f"{d['name']} ({d['accepted_count']} examples)": d["id"] for d in complete_ds}
        # Also allow custom path
        ds_options["📂 Custom JSONL path..."] = "__custom__"
        choice = st.selectbox("Select dataset", list(ds_options.keys()))
        ds_id = ds_options[choice]
        if ds_id == "__custom__":
            custom_path = st.text_input("Custom JSONL path", placeholder="/path/to/train.jsonl")
            if custom_path:
                st.session_state["custom_dataset_path"] = custom_path
        st.session_state["selected_dataset_id"] = ds_id

        if ds_id != "__custom__":
            ds = get_dataset(ds_id)
            if ds:
                st.json({
                    "name": ds["name"],
                    "examples": ds["accepted_count"],
                    "created": ds["created_at"][:16],
                    "path": ds["jsonl_path"],
                })

# ══════════════════════════════════════════════════════════════
# TAB 2: Model selection
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Search or Select Base Model</div>', unsafe_allow_html=True)

    col_search, col_curated = st.columns([0.6, 0.4])
    with col_search:
        search_query = st.text_input("🔍 Search Hugging Face Hub", placeholder="mistral, llama, qwen...")
        if st.button("Search", key="btn_hf_search"):
            with st.spinner("Searching..."):
                results = search_models(search_query, limit=15)
                st.session_state["hf_search_results"] = results

        results = st.session_state.get("hf_search_results", [])
        if results:
            for r in results:
                gated_tag = " 🔒" if r.gated else ""
                col_info, col_btn = st.columns([0.8, 0.2])
                with col_info:
                    st.markdown(
                        f"**{r.model_id}**{gated_tag} — {r.license or 'unknown license'} — "
                        f"⬇️ {r.downloads:,}"
                    )
                with col_btn:
                    if st.button("Select", key=f"sel_{r.model_id}"):
                        st.session_state["selected_hf_id"] = r.model_id

    with col_curated:
        st.markdown("**⭐ Curated suggestions**")
        for cm in CURATED_MODELS:
            col_m, col_btn = st.columns([0.75, 0.25])
            with col_m:
                st.caption(f"**{cm['id'].split('/')[-1]}** {cm['note']}")
            with col_btn:
                if st.button("Pick", key=f"pick_{cm['id']}"):
                    st.session_state["selected_hf_id"] = cm["id"]

    selected_hf = st.session_state.get("selected_hf_id", "")
    if selected_hf:
        st.markdown(f"---\n**Selected:** `{selected_hf}`")

        # Compatibility check
        if st.button("🔍 Check Compatibility", key="btn_compat"):
            with st.spinner("Loading model config..."):
                cfg = fetch_model_config(selected_hf, token=HF_TOKEN)
                if cfg is None:
                    st.error("Could not load model config. Check model ID and ensure you have access.")
                else:
                    report = check_model_compatibility(cfg)
                    st.session_state["compat_report"] = report
                    st.session_state["selected_local_path"] = (
                        get_local_path(selected_hf) if is_model_available(selected_hf)
                        else selected_hf
                    )

        report = st.session_state.get("compat_report")
        if report:
            if report.compatible:
                st.success(f"✅ Compatible — Architecture: `{report.architecture}`")
            else:
                st.error(f"❌ Incompatible")
            for issue in report.issues:
                st.error(f"• {issue}")
            for warn in report.warnings:
                st.warning(f"• {warn}")
            if report.lora_targets:
                st.caption(f"LoRA targets: `{', '.join(report.lora_targets)}`")

        # Download
        available = is_model_available(selected_hf)
        if available:
            sz = get_model_disk_usage(selected_hf) // 1024 // 1024
            st.success(f"✅ Model available locally ({sz} MB)")
            st.session_state["selected_local_path"] = get_local_path(selected_hf)
        else:
            st.warning("Model not downloaded yet.")
            if st.button("⬇️ Download Model", key="btn_download"):
                log_area = st.empty()
                prog = st.progress(0)
                msgs = []

                def _dl_progress(msg):
                    msgs.append(msg)
                    log_area.text("\n".join(msgs[-8:]))

                with st.spinner("Downloading..."):
                    ok, result = download_model(selected_hf, progress_callback=_dl_progress)

                if ok:
                    st.success(f"✅ Downloaded to: `{result}`")
                    st.session_state["selected_local_path"] = result
                    upsert_base_model(
                        hf_model_id=selected_hf,
                        local_dir=result,
                        download_status="complete",
                        disk_bytes=get_model_disk_usage(selected_hf),
                    )
                else:
                    st.error(f"Download failed: {result}")

# ══════════════════════════════════════════════════════════════
# TAB 3: Configure & Train
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Training Configuration</div>', unsafe_allow_html=True)

    # Method selector
    method_labels = {
        "qlora_sft": "QLoRA SFT (4-bit quantized, recommended)",
        "lora_sft":  "LoRA SFT (bf16 base model)",
        "full_sft":  "Full SFT (all parameters, max VRAM)",
    }
    method = st.selectbox("Fine-tuning method", list(method_labels.values()))
    method_key = [k for k, v in method_labels.items() if v == method][0]
    st.session_state["training_method"] = method_key

    desc = METHOD_DESCRIPTIONS[method_key]
    st.info(
        f"**{desc['name']}**\n\n{desc['description']}\n\n"
        f"Output type: **{desc['output_type']}** · "
        f"Memory note: *{desc['memory_note']}*"
    )

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Training Hyperparameters**")
        epochs     = st.number_input("Epochs",          1, 100, DEFAULT_EPOCHS)
        batch_size = st.number_input("Batch Size",       1, 64,  DEFAULT_BATCH_SIZE)
        grad_accum = st.number_input("Gradient Accum",   1, 128, DEFAULT_GRAD_ACCUM)
        lr         = st.number_input("Learning Rate",    value=DEFAULT_LR, format="%.2e", step=1e-5)
        warmup     = st.slider("Warmup Ratio",           0.0, 0.5, DEFAULT_WARMUP_RATIO, 0.01)
        max_seq    = st.number_input("Max Sequence Length", 128, 8192, DEFAULT_MAX_SEQ_LEN, 128)
        save_steps = st.number_input("Save Steps",       10, 1000, 50)
        log_steps  = st.number_input("Log Steps",        1, 100,  5)

    with col_r:
        st.markdown("**LoRA Settings**" if method_key != "full_sft" else "**Model Settings**")
        lora_r      = st.number_input("LoRA Rank (r)",    1, 256, DEFAULT_LORA_R,    disabled=method_key == "full_sft")
        lora_alpha  = st.number_input("LoRA Alpha",       1, 512, DEFAULT_LORA_ALPHA, disabled=method_key == "full_sft")
        lora_drop   = st.slider("LoRA Dropout",           0.0, 0.5, DEFAULT_LORA_DROPOUT, disabled=method_key == "full_sft")
        st.markdown("**Precision**")
        bf16 = st.checkbox("BF16", value=True)
        fp16 = st.checkbox("FP16", value=False, disabled=bf16)
        grad_ckpt = st.checkbox("Gradient Checkpointing", value=True)

    display_name = st.text_input("Output Model Name", placeholder="e.g. my-qwen-domain-expert")
    output_base  = st.text_input(
        "Output Directory",
        value=str(FINE_TUNED_MODELS_DIR / (display_name or "output")),
    )

    st.markdown("---")

    # Derive lora targets from compat report
    report = st.session_state.get("compat_report")
    lora_targets = report.lora_targets if report else None

    # Validate pre-flight
    hf_id     = st.session_state.get("selected_hf_id", "")
    local_path = st.session_state.get("selected_local_path", "")
    model_path = local_path if local_path and pathlib.Path(local_path).exists() else hf_id

    ds_id = st.session_state.get("selected_dataset_id", "")
    if ds_id == "__custom__":
        dataset_path = st.session_state.get("custom_dataset_path", "")
    elif ds_id:
        ds = get_dataset(ds_id)
        dataset_path = ds["jsonl_path"] if ds else ""
    else:
        dataset_path = ""

    preflight_ok = bool(model_path and dataset_path and display_name)
    if not preflight_ok:
        missing = []
        if not model_path:    missing.append("base model")
        if not dataset_path:  missing.append("dataset")
        if not display_name:  missing.append("output model name")
        st.warning(f"⚠️ Not ready to train — missing: {', '.join(missing)}")

    # VRAM estimate
    if report and hf_id:
        param_count = getattr(report, "param_count", None)
        vram = estimate_vram_gb(param_count, method_key, max_seq)
        vram_str = f"~{vram:.1f} GB" if vram > 0 else "unknown (param count not available)"
        st.caption(f"💾 Estimated VRAM: **{vram_str}** (rough estimate). Your system has 130 GB.")

    if st.button(
        "🚀 Launch Training", key="btn_train",
        type="primary", disabled=not preflight_ok,
    ):
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir   = str(FINE_TUNED_MODELS_DIR / f"{display_name}_{ts}")
        log_path  = str(LOGS_DIR / f"train_{display_name}_{ts}.log")
        ckpt_dir  = out_dir

        # Register in DB
        bm = get_base_model_by_hf_id(hf_id) if hf_id else None
        bm_id = bm["id"] if bm else upsert_base_model(
            hf_model_id=hf_id or model_path,
            local_dir=local_path, download_status="complete"
        )

        ft_id = register_fine_tuned_model(
            display_name=display_name,
            base_model_id=bm_id,
            dataset_id=ds_id if ds_id != "__custom__" else "",
            training_method=method_key,
            output_dir=out_dir,
            adapter_path=str(pathlib.Path(out_dir) / "adapter_model.safetensors"),
            model_type="adapter" if method_key != "full_sft" else "full_model",
        )
        update_fine_tuned_model(ft_id, training_status="running")

        cfg = TrainingConfig(
            method=method_key,
            fine_tuned_model_id=ft_id,
            display_name=display_name,
            base_model_path=model_path,
            dataset_path=dataset_path,
            output_dir=out_dir,
            log_path=log_path,
            checkpoint_dir=ckpt_dir,
            max_seq_len=max_seq,
            batch_size=batch_size,
            grad_accum=grad_accum,
            epochs=epochs,
            lr=lr,
            warmup_ratio=warmup,
            save_steps=save_steps,
            log_steps=log_steps,
            bf16=bf16,
            fp16=fp16 and not bf16,
            gradient_checkpointing=grad_ckpt,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_drop,
            lora_targets=lora_targets,
        )

        run_id = create_training_run(
            fine_tuned_model_id=ft_id,
            config=cfg.to_dict(),
            log_path=log_path,
            checkpoint_dir=ckpt_dir,
        )

        ok, msg = launch_training(cfg)
        if ok:
            update_fine_tuned_model(ft_id, training_run_id=run_id)
            st.session_state["ft_model_id"] = ft_id
            st.session_state["training_run_id"] = run_id
            st.session_state["log_path"] = log_path
            st.success(f"✅ {msg}")
            st.info("Switch to the **📊 Live Training** tab to monitor progress.")
        else:
            finish_training_run(run_id, "failed", error_message=msg)
            update_fine_tuned_model(ft_id, training_status="failed")
            st.error(f"Failed to start training: {msg}")

# ══════════════════════════════════════════════════════════════
# TAB 4: Live Training
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">📊 Live Training Monitor</div>', unsafe_allow_html=True)

    status = get_process_status()
    log_path = st.session_state.get("log_path")

    c1, c2 = st.columns(2)
    with c1:
        if status["running"]:
            st.success(f"🟢 Training running (PID {status['pid']})")
        elif status["returncode"] is not None:
            if status["returncode"] == 0:
                st.success("✅ Training completed successfully.")
                # Mark as complete
                ft_id = st.session_state.get("ft_model_id")
                run_id = st.session_state.get("training_run_id")
                if ft_id:
                    update_fine_tuned_model(ft_id, training_status="complete")
                if run_id:
                    finish_training_run(run_id, "complete")
            else:
                st.error(f"❌ Training exited with code {status['returncode']}")
                ft_id = st.session_state.get("ft_model_id")
                run_id = st.session_state.get("training_run_id")
                if ft_id:
                    update_fine_tuned_model(ft_id, training_status="failed")
                if run_id:
                    finish_training_run(run_id, "failed")
        else:
            st.info("No training job active.")

    with c2:
        if status["running"]:
            if st.button("🛑 Cancel Training", key="btn_cancel"):
                msg = cancel_training()
                run_id = st.session_state.get("training_run_id")
                ft_id  = st.session_state.get("ft_model_id")
                if run_id:
                    finish_training_run(run_id, "cancelled")
                if ft_id:
                    update_fine_tuned_model(ft_id, training_status="cancelled")
                st.warning(msg)

    if log_path:
        st.markdown("**Training Logs (last 80 lines)**")
        lines = tail_log(log_path, 80)
        log_text = "\n".join(lines) if lines else "No log output yet..."
        st.code(log_text, language="text")
        if status["running"]:
            time.sleep(3)
            st.rerun()
    else:
        st.info("Start a training job in the **Configure & Train** tab to see logs here.")
