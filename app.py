"""
app.py — Main entry point for the AI Dataset & Fine-Tuning Platform.
Run with: streamlit run app.py
"""

import sys
import pathlib

# Ensure src/ is importable
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="AI Fine-Tuning Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap DB and auto-discover existing adapters ──────────
from src.storage.database import init_db
from src.model_management.model_registry import auto_discover_adapters, scan_workspace_adapters
from src.storage.project_repository import get_dashboard_stats
from src.theme import apply_theme

init_db()

# ── Global CSS & Dark Theme ───────────────────────────────────
apply_theme()

if "adapters_discovered" not in st.session_state:
    auto_discover_adapters()
    scan_workspace_adapters()
    st.session_state["adapters_discovered"] = True

# ── Dashboard ─────────────────────────────────────────────────
st.markdown('<div class="page-title">🧠 AI Fine-Tuning Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">End-to-end dataset generation, model fine-tuning, and inference — powered by your existing QLoRA pipeline</div>', unsafe_allow_html=True)

# Stats row
stats = get_dashboard_stats()
col1, col2, col3, col4 = st.columns(4)

def stat_card(col, number, label, icon):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size:1.8rem;">{icon}</div>
            <div class="stat-number">{number}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

stat_card(col1, stats["n_datasets"],       "Datasets Created",     "📊")
stat_card(col2, stats["n_base_models"],    "Base Models Ready",    "🤗")
stat_card(col3, stats["n_successful_runs"],"Training Runs Done",   "🏋️")
stat_card(col4, stats["n_ft_models"],      "Fine-Tuned Models",    "✨")

st.markdown("---")

# Navigation cards
st.markdown('<div class="section-header">🗺️ Navigation</div>', unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3 = st.columns(3)

def nav_card(col, icon, title, desc, page):
    with col:
        st.markdown(f"""
        <div class="stat-card" style="min-height:130px; margin-bottom: 0.5rem;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-weight:600;color:#f1f5f9;margin:0.5rem 0 0.25rem;font-size:1.05rem;">{title}</div>
            <div style="color:#94a3b8;font-size:0.85rem;line-height:1.4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(page, label=f"Open {title} →")

nav_card(nav_col1, "📄", "Dataset Generation",
         "Upload docs, generate Q&A with Azure GPT",
         "pages/1_Dataset_Generation.py")
nav_card(nav_col2, "🏋️", "Model Fine-Tuning",
         "Select model, configure & run training",
         "pages/2_Model_Fine_Tuning.py")
nav_card(nav_col3, "🗂️", "Fine-Tuned Models",
         "Browse, load, export your models",
         "pages/3_Fine_Tuned_Models.py")

nav_col4, nav_col5, nav_col6 = st.columns(3)
nav_card(nav_col4, "💬", "Model Playground",
         "Chat with your fine-tuned model",
         "pages/4_Model_Playground.py")
nav_card(nav_col5, "📈", "Training History",
         "View logs and past training runs",
         "pages/5_Training_History.py")
nav_card(nav_col6, "⚙️", "Application Settings",
         "Configure Azure, HF token, directories",
         "pages/5_Training_History.py")  # Settings live in the ⚙️ tab of Training History

# Recent training runs
if stats["recent_runs"]:
    st.markdown("---")
    st.markdown('<div class="section-header">🕐 Recent Training Activity</div>', unsafe_allow_html=True)
    for run in stats["recent_runs"]:
        status = run.get("status", "unknown")
        badge_class = {
            "complete": "badge-complete",
            "running": "badge-running",
            "failed": "badge-failed",
        }.get(status, "badge-pending")
        st.markdown(
            f"**{run.get('model_name','Unknown')}** &nbsp; "
            f"<span class='{badge_class}'>{status}</span> &nbsp; "
            f"<span style='color:#64748b;font-size:0.85rem;'>{run.get('started_at','')[:16]}</span>",
            unsafe_allow_html=True,
        )
