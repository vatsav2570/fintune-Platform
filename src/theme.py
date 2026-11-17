"""
src/theme.py
─────────────────────────────────────────────────────────────────
Central UI Theme & CSS for AI Fine-Tuning Platform.
Provides a modern, high-contrast dark theme across all pages.
"""

import streamlit as st

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #f1f5f9;
}

/* Force dark background on whole application */
.stApp,
[data-testid="stAppViewContainer"],
.main {
    background: radial-gradient(ellipse at 50% 0%, #171f38 0%, #0b0f19 70%, #060911 100%) !important;
    color: #f1f5f9 !important;
}

/* Header bar */
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stToolbar"] {
    color: #94a3b8 !important;
}

/* Main block padding */
.main .block-container {
    padding: 1.75rem 2.5rem;
    max-width: 1300px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f19 0%, #0f172a 60%, #080d1a 100%) !important;
    border-right: 1px solid rgba(99, 179, 237, 0.15) !important;
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] p {
    color: #cbd5e1 !important;
}

/* Sidebar navigation links */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] a {
    color: #94a3b8 !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.75rem !important;
    margin-bottom: 2px !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a span,
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] a span {
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] a:hover {
    background: rgba(255, 255, 255, 0.06) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover span,
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] a:hover span {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"],
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] a[aria-current="page"] {
    background: rgba(99, 179, 237, 0.15) !important;
    border-left: 3px solid #63b3ed !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] span,
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] a[aria-current="page"] span {
    color: #63b3ed !important;
    font-weight: 600 !important;
}

/* Title & Subtitle */
.page-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed 0%, #a78bfa 50%, #f687b3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0.35rem;
}
.page-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 1.75rem;
    line-height: 1.5;
}

/* Section headers */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #f1f5f9;
    border-bottom: 1px solid rgba(99, 179, 237, 0.2);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Cards */
.stat-card {
    background: linear-gradient(135deg, rgba(22, 30, 52, 0.85) 0%, rgba(13, 18, 33, 0.9) 100%);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.stat-card:hover {
    border-color: rgba(99, 179, 237, 0.5);
    transform: translateY(-3px);
    box-shadow: 0 12px 28px -4px rgba(99, 179, 237, 0.2);
}
.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #63b3ed 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-top: 0.25rem;
}
.stat-label {
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

/* Other Cards */
.model-card, .file-card, .method-card, .info-card {
    background: linear-gradient(135deg, rgba(22, 30, 52, 0.75) 0%, rgba(13, 18, 33, 0.85) 100%) !important;
    border: 1px solid rgba(99, 179, 237, 0.2) !important;
    border-radius: 16px !important;
    padding: 1.25rem !important;
    margin-bottom: 1rem !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35) !important;
}

/* Streamlit page links */
div[data-testid="stPageLink-NavLink"],
div[data-testid="stPageLink"] a {
    background: rgba(99, 179, 237, 0.1) !important;
    border: 1px solid rgba(99, 179, 237, 0.25) !important;
    border-radius: 10px !important;
    padding: 0.45rem 0.9rem !important;
    color: #93c5fd !important;
    text-decoration: none !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    margin-top: 0.35rem !important;
}
div[data-testid="stPageLink-NavLink"]:hover,
div[data-testid="stPageLink"] a:hover {
    background: rgba(99, 179, 237, 0.22) !important;
    border-color: rgba(99, 179, 237, 0.55) !important;
    color: #ffffff !important;
    transform: translateX(2px) !important;
}

/* Status badges */
.badge {
    padding: 3px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 500;
    display: inline-block;
}
.badge-complete {
    background: rgba(16, 185, 129, 0.18);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.badge-running {
    background: rgba(59, 130, 246, 0.18);
    color: #93c5fd;
    border: 1px solid rgba(59, 130, 246, 0.35);
}
.badge-failed {
    background: rgba(239, 68, 68, 0.18);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.badge-pending {
    background: rgba(107, 114, 128, 0.18);
    color: #d1d5db;
    border: 1px solid rgba(107, 114, 128, 0.35);
}
.badge-adapter {
    background: rgba(167, 139, 250, 0.18);
    color: #c4b5fd;
    border: 1px solid rgba(167, 139, 250, 0.35);
}
.badge-full {
    background: rgba(16, 185, 129, 0.18);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

/* Compatibility colors */
.compat-ok { color: #6ee7b7; font-weight: 500; }
.compat-warn { color: #fbbf24; font-weight: 500; }
.compat-err { color: #f87171; font-weight: 500; }

/* Buttons */
div[data-testid="stButton"] > button {
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.2s;
    border: 1px solid rgba(99, 179, 237, 0.25);
    background: rgba(26, 34, 58, 0.8);
    color: #e2e8f0;
}
div[data-testid="stButton"] > button:hover {
    border-color: rgba(99, 179, 237, 0.5);
    background: rgba(37, 48, 80, 0.9);
    color: #ffffff;
    transform: translateY(-1px);
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6) !important;
    transform: translateY(-2px);
}

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(17, 24, 39, 0.55) !important;
    border: 1px solid rgba(99, 179, 237, 0.18) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stExpander"] summary {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #63b3ed !important;
    font-weight: 600 !important;
    border-bottom-color: #63b3ed !important;
}

/* Inputs & Form controls */
div[data-baseweb="input"],
div[data-baseweb="textarea"],
div[data-baseweb="select"] > div {
    background-color: rgba(17, 24, 39, 0.75) !important;
    border-color: rgba(99, 179, 237, 0.22) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}
[data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

/* Horizontal rule */
hr {
    border-color: rgba(99, 179, 237, 0.15) !important;
    margin: 1.5rem 0 !important;
}

/* Chat styles */
.chat-user {
    background: rgba(99, 179, 237, 0.12);
    border: 1px solid rgba(99, 179, 237, 0.25);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    text-align: right;
}
.chat-bot {
    background: rgba(167, 139, 250, 0.12);
    border: 1px solid rgba(167, 139, 250, 0.25);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}
.model-badge {
    background: rgba(22, 30, 52, 0.9);
    border: 1px solid rgba(99, 179, 237, 0.35);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    display: inline-block;
    font-size: 0.85rem;
    color: #93c5fd;
}
.gen-counter {
    font-size: 1.8rem;
    font-weight: 700;
    color: #63b3ed;
}
</style>
"""

def apply_theme():
    """Inject the global dark theme CSS into the current Streamlit page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
