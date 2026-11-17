"""
pages/1_Dataset_Generation.py
────────────────────────────────────────────────────────────────
Upload documents → extract text → generate Q&A with Azure OpenAI
→ review, edit, download, and save dataset for fine-tuning.
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import tempfile
import time
import datetime
import streamlit as st

from src.config import DATASETS_DIR, azure_configured, azure_config_errors, GENERATION_BATCH_SIZE
from src.storage.project_repository import (
    create_dataset, update_dataset_counts, get_dataset, list_datasets
)
from src.document_processing.pdf_processor import extract_pdf
from src.document_processing.txt_processor import extract_txt
from src.document_processing.docx_processor import extract_docx
from src.document_processing.text_chunker import chunk_documents
from src.dataset_generation.azure_client import validate_azure_connection
from src.dataset_generation.dataset_adapters import (
    save_jsonl, load_jsonl, validate_jsonl_schema
)
from src.dataset_generation.validators import _fingerprint
from src.theme import apply_theme

st.set_page_config(page_title="Dataset Generation", page_icon="📄", layout="wide")

# ── Global CSS & Theme ───────────────────────────────────────
apply_theme()

st.markdown('<div class="page-title">📄 Dataset Generation</div>', unsafe_allow_html=True)
st.markdown("Upload documents, extract text, and generate a fine-tuning dataset using Azure OpenAI.")

# ── Session state init ────────────────────────────────────────
for k, v in {
    "extracted_docs": [],        # [{filename, text, word_count, error, is_scanned}]
    "generated_examples": [],    # [{question, answer}]
    "dataset_id": None,
    "generation_done": False,
    "azure_ok": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══ STEP 1: Azure validation ══════════════════════════════════
with st.expander("🔐 Azure OpenAI Configuration", expanded=not azure_configured()):
    if not azure_configured():
        missing = azure_config_errors()
        st.error(
            f"⚠️ Missing environment variables: `{'`, `'.join(missing)}`\n\n"
            "Add them to your `.env` file and restart Streamlit.\n"
            "**The API key must never be entered in the UI.**"
        )
    else:
        if st.button("✅ Validate Azure Connection", key="btn_validate_azure"):
            with st.spinner("Connecting to Azure OpenAI..."):
                ok, msg = validate_azure_connection()
                st.session_state["azure_ok"] = ok
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        if st.session_state.get("azure_ok"):
            st.success("Azure connection verified.")
        elif st.session_state.get("azure_ok") is False:
            st.error("Azure connection failed. Check your .env variables.")

st.markdown("---")

# ══ STEP 2: Document upload ═══════════════════════════════════
st.markdown('<div class="section-header">📁 Step 1 — Upload Documents</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload PDF, TXT, or DOCX files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
    key="file_uploader",
)

MAX_FILE_SIZE_MB = 50

if uploaded:
    if st.button("🔍 Extract Text from All Files", key="btn_extract"):
        st.session_state["extracted_docs"] = []
        progress = st.progress(0)
        for i, uf in enumerate(uploaded):
            # Security: sanitize filename
            safe_name = pathlib.Path(uf.name).name
            ext = safe_name.rsplit(".", 1)[-1].lower()
            size_mb = uf.size / 1024 / 1024

            if size_mb > MAX_FILE_SIZE_MB:
                st.session_state["extracted_docs"].append({
                    "filename": safe_name, "text": "", "word_count": 0,
                    "error": f"File too large ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB limit)",
                    "is_scanned": False, "ext": ext, "size_mb": size_mb,
                })
                progress.progress((i + 1) / len(uploaded))
                continue

            # Save to temp file for extraction
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name

            try:
                if ext == "pdf":
                    result = extract_pdf(tmp_path)
                elif ext == "txt":
                    result = extract_txt(tmp_path)
                elif ext == "docx":
                    result = extract_docx(tmp_path)
                else:
                    result = None

                if result:
                    st.session_state["extracted_docs"].append({
                        "filename": safe_name,
                        "text": result.text,
                        "word_count": result.word_count,
                        "error": result.error,
                        "is_scanned": result.is_scanned,
                        "ext": ext,
                        "size_mb": size_mb,
                        "page_count": result.page_count,
                    })
            finally:
                os.unlink(tmp_path)
            progress.progress((i + 1) / len(uploaded))
        st.success(f"Extracted text from {len(st.session_state['extracted_docs'])} file(s).")

# Show extraction results
if st.session_state["extracted_docs"]:
    st.markdown('<div class="section-header">📋 Extraction Results</div>', unsafe_allow_html=True)
    for i, doc in enumerate(st.session_state["extracted_docs"]):
        col_icon, col_info, col_action = st.columns([0.08, 0.75, 0.17])
        with col_icon:
            st.markdown({"pdf": "📕", "txt": "📄", "docx": "📘"}.get(doc["ext"], "📄"))
        with col_info:
            if doc["error"]:
                st.markdown(f"**{doc['filename']}** — ⚠️ {doc['error']}")
            else:
                st.markdown(
                    f"**{doc['filename']}** — {doc['word_count']:,} words | "
                    f"{doc.get('page_count', 1)} pages | {doc['size_mb']:.2f} MB"
                )
        with col_action:
            if st.button("🗑️", key=f"remove_doc_{i}", help="Remove this file"):
                st.session_state["extracted_docs"].pop(i)
                st.rerun()

    # Preview
    with st.expander("🔍 Preview Extracted Text"):
        valid_docs = [d for d in st.session_state["extracted_docs"] if d["text"]]
        if valid_docs:
            selected = st.selectbox("Select file to preview:", [d["filename"] for d in valid_docs])
            doc = next(d for d in valid_docs if d["filename"] == selected)
            st.text_area("Extracted Text (first 3000 chars):", doc["text"][:3000], height=250)
        else:
            st.info("No successfully extracted documents.")

st.markdown("---")

# ══ STEP 3: Generation settings ══════════════════════════════
st.markdown('<div class="section-header">⚙️ Step 2 — Generation Settings</div>', unsafe_allow_html=True)

valid_docs = [d for d in st.session_state.get("extracted_docs", []) if d.get("text")]

col_a, col_b = st.columns(2)
with col_a:
    dataset_name = st.text_input("Dataset Name", placeholder="e.g. Electronics Training Dataset")
    n_questions = st.number_input(
        "Number of Questions to Generate",
        min_value=1, max_value=10000, value=50, step=10,
        help="How many training examples to generate from your documents."
    )
    difficulty = st.selectbox("Difficulty Level", ["Mixed", "Basic", "Intermediate", "Advanced"])

with col_b:
    question_types = st.multiselect(
        "Question Types",
        ["Conceptual", "Technical", "Analytical", "Comparative", "Procedural", "Troubleshooting"],
        default=["Conceptual", "Technical", "Analytical"],
    )
    custom_instructions = st.text_area(
        "Custom Instructions (optional)",
        placeholder="e.g. Focus on safety specifications. Include numerical values where relevant.",
        height=80,
    )

system_prompt_override = st.text_area(
    "System Prompt for Training (shown to model at inference)",
    value=(
        "You are a knowledgeable and precise assistant. "
        "Answer questions accurately based on the provided training material. "
        "Be concise and factual."
    ),
    height=80,
)

st.markdown("---")

# ══ STEP 4: Generate ═════════════════════════════════════════
st.markdown('<div class="section-header">🚀 Step 3 — Generate Dataset</div>', unsafe_allow_html=True)

if not valid_docs:
    st.info("Upload and extract at least one document first.")
elif not azure_configured():
    st.warning("Configure Azure OpenAI first.")
elif not dataset_name.strip():
    st.warning("Enter a dataset name.")
elif not question_types:
    st.warning("Select at least one question type.")
else:
    col_gen, col_info = st.columns([0.4, 0.6])
    with col_info:
        total_words = sum(d["word_count"] for d in valid_docs)
        st.info(
            f"📊 **{len(valid_docs)} doc(s)** · {total_words:,} words total · "
            f"**{n_questions} questions** requested"
        )
        st.caption("Document content will be sent to your Azure OpenAI deployment for generation.")

    with col_gen:
        generate_btn = st.button(
            "🧠 Generate Dataset", key="btn_generate",
            type="primary", use_container_width=True,
        )

    if generate_btn:
        st.session_state["generated_examples"] = []
        st.session_state["generation_done"] = False

        # Setup output directory
        safe_ds_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name.strip())
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        ds_dir = DATASETS_DIR / f"{safe_ds_name}_{ts}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = ds_dir / "generated_dataset.jsonl"
        meta_path = ds_dir / "generation_metadata.json"

        # Register in DB
        ds_id = create_dataset(
            name=dataset_name.strip(),
            source_files=[d["filename"] for d in valid_docs],
            requested_count=n_questions,
            generation_settings={
                "question_types": question_types,
                "difficulty": difficulty,
                "custom_instructions": custom_instructions,
            },
            dataset_dir=str(ds_dir),
        )
        st.session_state["dataset_id"] = ds_id

        # Chunk documents
        chunks = chunk_documents([{"filename": d["filename"], "text": d["text"]} for d in valid_docs])

        # Progress UI
        prog_bar = st.progress(0)
        status_box = st.empty()
        counter = st.empty()

        def _progress(accepted, total, msg):
            pct = min(accepted / max(total, 1), 1.0)
            prog_bar.progress(pct)
            status_box.info(f"🔄 {msg}")
            counter.markdown(
                f'<div class="gen-counter">{accepted}/{total}</div> accepted',
                unsafe_allow_html=True
            )

        from src.dataset_generation.generator import generate_dataset

        with st.spinner("Generating... (this may take several minutes for large counts)"):
            result = generate_dataset(
                chunks=chunks,
                total_questions=n_questions,
                question_types=question_types,
                difficulty=difficulty,
                output_jsonl_path=jsonl_path,
                output_metadata_path=meta_path,
                system_prompt=system_prompt_override,
                custom_instructions=custom_instructions,
                batch_size=GENERATION_BATCH_SIZE,
                progress_callback=_progress,
            )

        if "error" in result:
            st.error(result["error"])
        else:
            # Load generated examples for review
            records = load_jsonl(jsonl_path)
            examples = []
            for rec in records:
                msgs = rec.get("messages", [])
                q = next((m["content"] for m in msgs if m["role"] == "user"), "")
                a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                examples.append({"question": q, "answer": a})
            st.session_state["generated_examples"] = examples
            st.session_state["generation_done"] = True

            # Update DB
            update_dataset_counts(
                ds_id=ds_id,
                accepted=result["accepted"],
                duplicates=result["duplicates"],
                rejected=result["rejected"],
                jsonl_path=str(jsonl_path),
                metadata_path=str(meta_path),
            )

            st.success(
                f"✅ Generation complete! "
                f"**{result['accepted']}** accepted · "
                f"**{result['duplicates']}** duplicates · "
                f"**{result['rejected']}** rejected"
            )
            if result.get("parse_errors"):
                with st.expander(f"⚠️ {len(result['parse_errors'])} parse warnings"):
                    for e in result["parse_errors"][:20]:
                        st.caption(e)

# ══ STEP 5: Review & Edit ═════════════════════════════════════
if st.session_state["generated_examples"]:
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Step 4 — Review & Edit Dataset</div>', unsafe_allow_html=True)

    examples = st.session_state["generated_examples"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Examples", len(examples))
    c2.metric("Questions Generated", len(examples))

    # Paginated view
    page_size = 10
    n_pages = max(1, (len(examples) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=n_pages, value=1) - 1
    start = page * page_size
    shown = examples[start:start + page_size]

    for idx, ex in enumerate(shown):
        real_idx = start + idx
        with st.expander(f"#{real_idx + 1} — {ex['question'][:80]}..."):
            new_q = st.text_area("Question", ex["question"], key=f"q_{real_idx}", height=60)
            new_a = st.text_area("Answer", ex["answer"], key=f"a_{real_idx}", height=120)
            col_save, col_del = st.columns(2)
            if col_save.button("💾 Save Edit", key=f"save_{real_idx}"):
                st.session_state["generated_examples"][real_idx]["question"] = new_q
                st.session_state["generated_examples"][real_idx]["answer"] = new_a
                st.success("Saved.")
            if col_del.button("🗑️ Delete", key=f"del_{real_idx}"):
                st.session_state["generated_examples"].pop(real_idx)
                st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-header">💾 Step 5 — Save & Download</div>', unsafe_allow_html=True)

    col_dl, col_save_ft = st.columns(2)

    with col_dl:
        # Convert to JSONL for download
        from src.dataset_generation.dataset_adapters import examples_to_jsonl
        records = examples_to_jsonl(
            examples, system_prompt=system_prompt_override
        )
        jsonl_str = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        st.download_button(
            "⬇️ Download JSONL",
            data=jsonl_str.encode("utf-8"),
            file_name=f"{dataset_name or 'dataset'}.jsonl",
            mime="application/jsonlines",
            use_container_width=True,
        )

    with col_save_ft:
        if st.button("✅ Save Dataset for Fine-Tuning", type="primary", use_container_width=True):
            ds_id = st.session_state.get("dataset_id")
            if ds_id:
                ds = get_dataset(ds_id)
                if ds and ds.get("dataset_dir"):
                    final_path = pathlib.Path(ds["dataset_dir"]) / "generated_dataset.jsonl"
                    save_jsonl(records, final_path)
                    update_dataset_counts(
                        ds_id=ds_id,
                        accepted=len(records),
                        duplicates=0,
                        rejected=0,
                        jsonl_path=str(final_path),
                        metadata_path=ds.get("metadata_path", ""),
                        status="complete",
                    )
                    st.success(f"✅ Dataset saved to: `{final_path}`\nReady for fine-tuning on the **Model Fine-Tuning** page.")
                    st.session_state["saved_dataset_path"] = str(final_path)
            else:
                st.warning("Generate a dataset first.")
