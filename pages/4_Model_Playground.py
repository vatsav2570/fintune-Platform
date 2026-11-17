"""
pages/4_Model_Playground.py
────────────────────────────────────────────────────────────────
Chat with the loaded fine-tuned model.
Streaming responses via TextIteratorStreamer.
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from src.inference.model_loader import is_loaded, get_loaded_model_id, is_torch_available
from src.inference.chat_engine import generate_streaming
from src.storage.project_repository import list_fine_tuned_models, get_fine_tuned_model
from src.theme import apply_theme

st.set_page_config(page_title="Model Playground", page_icon="💬", layout="wide")

# ── Global CSS & Theme ───────────────────────────────────────
apply_theme()

st.markdown('<div class="page-title">💬 Model Playground</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = (
        "You are a helpful and knowledgeable assistant. "
        "Answer questions accurately and concisely."
    )

# ── Sidebar: model info + gen settings ───────────────────────
with st.sidebar:
    st.markdown("### 🤖 Model Status")
    if is_loaded():
        ft_id = get_loaded_model_id()
        ft = get_fine_tuned_model(ft_id) if ft_id else None
        if ft:
            st.markdown(f"""
            <div class="model-badge">
                ✨ <b>{ft['display_name']}</b><br>
                <span style='color:#64748b;font-size:0.8rem;'>
                Base: {ft.get('base_hf_id','?')}<br>
                Method: {ft.get('training_method','?')}
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("Model loaded.")
    else:
        st.warning("No model loaded.\nGo to **🗂️ Fine-Tuned Models** to load one.")

    st.markdown("---")
    st.markdown("### ⚙️ Generation Settings")
    temperature = st.slider("Temperature",     0.0, 2.0, 0.7, 0.05)
    top_p       = st.slider("Top-p",           0.0, 1.0, 0.9, 0.05)
    max_tokens  = st.number_input("Max New Tokens", 32, 4096, 512)
    rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.05)
    do_sample   = temperature > 0.01

    st.markdown("---")
    st.markdown("### 📝 System Prompt")
    st.session_state["system_prompt"] = st.text_area(
        "System Prompt", st.session_state["system_prompt"], height=120, label_visibility="collapsed"
    )

    if st.button("🗑️ Clear Conversation"):
        st.session_state["chat_history"] = []
        st.rerun()

# ── Chat display ──────────────────────────────────────────────
if not is_torch_available():
    st.warning("⚠️ **PyTorch is not installed in the active environment.** To chat with fine-tuned models in the playground, install PyTorch: `pip install torch`")
elif not is_loaded():
    st.info(
        "⬅️ No model is currently loaded. "
        "Go to the **🗂️ Fine-Tuned Models** page, find your model, and click **💬 Load**."
    )
else:
    # Show history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if user_input := st.chat_input("Type your message..."):
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Build messages list
        messages = [{"role": "system", "content": st.session_state["system_prompt"]}]
        messages.extend(st.session_state["chat_history"])

        # Streaming response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                for token in generate_streaming(
                    messages=messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                    do_sample=do_sample,
                ):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            except Exception as exc:
                full_response = f"❌ Error: {exc}"
                response_placeholder.error(full_response)

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": full_response}
        )
