"""
src/config.py
─────────────────────────────────────────────────────────────────
Central application configuration.
All secrets are loaded from .env using python-dotenv.
Nothing here is hardcoded; never expose credentials in logs/UI.
"""

import os
import pathlib
from dotenv import load_dotenv

# ── Locate project root and .env ──────────────────────────────
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"
load_dotenv(ENV_FILE)

# ── Workspace directories ─────────────────────────────────────
WORKSPACE_DIR         = ROOT_DIR / "workspace"
DATASETS_DIR          = WORKSPACE_DIR / "datasets"
BASE_MODELS_DIR       = WORKSPACE_DIR / "base_models"
FINE_TUNED_MODELS_DIR = WORKSPACE_DIR / "fine_tuned_models"
EXPORTS_DIR           = WORKSPACE_DIR / "exports"
LOGS_DIR              = WORKSPACE_DIR / "logs"
DB_PATH               = WORKSPACE_DIR / "registry.db"

for _d in (DATASETS_DIR, BASE_MODELS_DIR, FINE_TUNED_MODELS_DIR, EXPORTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Azure OpenAI ──────────────────────────────────────────────
AZURE_API_KEY         = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT        = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT      = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "Astra_IDPT")
AZURE_API_VERSION     = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

# ── Hugging Face ──────────────────────────────────────────────
HF_TOKEN              = os.getenv("HF_TOKEN", "")
HF_CACHE_DIR          = str(BASE_MODELS_DIR)

# ── Training defaults ─────────────────────────────────────────
DEFAULT_MAX_SEQ_LEN   = 2048
DEFAULT_BATCH_SIZE    = 4
DEFAULT_GRAD_ACCUM    = 4
DEFAULT_EPOCHS        = 3
DEFAULT_LR            = 2e-4
DEFAULT_WARMUP_RATIO  = 0.05
DEFAULT_SAVE_STEPS    = 50
DEFAULT_LOG_STEPS     = 5
DEFAULT_LORA_R        = 16
DEFAULT_LORA_ALPHA    = 32
DEFAULT_LORA_DROPOUT  = 0.05

# ── Dataset generation ────────────────────────────────────────
GENERATION_BATCH_SIZE    = 10   # questions per API call
MAX_CHUNK_TOKENS         = 6000  # tokens per document chunk
CHUNK_OVERLAP_CHARS      = 200

# ── Application ────────────────────────────────────────────────
APP_TITLE    = "AI Dataset & Fine-Tuning Platform"
APP_VERSION  = "1.0.0"


def azure_configured() -> bool:
    """Return True if all required Azure env vars are non-empty."""
    return all([AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT, AZURE_API_VERSION])


def azure_config_errors() -> list[str]:
    """Return a list of missing Azure environment variable names."""
    errors = []
    if not AZURE_API_KEY:
        errors.append("AZURE_OPENAI_API_KEY")
    if not AZURE_ENDPOINT:
        errors.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_DEPLOYMENT:
        errors.append("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not AZURE_API_VERSION:
        errors.append("AZURE_OPENAI_API_VERSION")
    return errors
