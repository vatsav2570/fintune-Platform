# AI Dataset Generation & Model Fine-Tuning Platform

> A complete Streamlit application extending the existing QLoRA fine-tuning pipeline.
> Upload documents → generate datasets via Azure OpenAI → fine-tune models → chat with results → export.

---

## 🖥️ System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Linux (tested on Ubuntu 24.04, aarch64) |
| GPU | NVIDIA CUDA-capable (GB10 / DGX Spark tested) |
| VRAM | ≥ 8 GB for QLoRA on small models;  |
| Python | 3.12.3 (inside `venv/`) |
| CUDA | 13.0 (Driver 580.159.03) |

---

## 🚀 Setup & Installation

### Step 1 — Activate the existing virtual environment

```bash
cd ~/Sreevatsav/qlora-qwen3
source venv/bin/activate
```

### Step 2 — Install application dependencies

```bash
pip install -r requirements_app.txt
```

> The ML stack (`requirements.txt`) is already installed in `venv/`.
> `requirements_app.txt` adds only Streamlit, Azure OpenAI SDK, python-docx, and python-dotenv.

### Step 3 — Configure environment variables

Your `.env` file already contains the required variables. Verify it contains:

```bash
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://azure.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=model_name
AZURE_OPENAI_API_VERSION=xxxx
```

For gated Hugging Face models (e.g. Llama 3), also add:

```bash
HF_TOKEN=your_huggingface_token_here
```

**Never commit `.env` to Git.** See `.env.example` for the variable names.

### Step 4 — Launch the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📁 Project Structure

```
qlora-qwen3/
├── app.py                          ← Streamlit dashboard (entry point)
│
├── pages/
│   ├── 1_Dataset_Generation.py     ← Upload docs, generate Q&A via Azure
│   ├── 2_Model_Fine_Tuning.py      ← HF model search, download, train
│   ├── 3_Fine_Tuned_Models.py      ← Browse, load, export models
│   ├── 4_Model_Playground.py       ← Chat with fine-tuned model
│   └── 5_Training_History.py       ← Logs, datasets, settings
│
├── src/
│   ├── config.py                   ← Loads .env, all app constants
│   ├── document_processing/        ← PDF, TXT, DOCX extraction + chunking
│   ├── dataset_generation/         ← Azure client, prompts, validator, adapter
│   ├── model_management/           ← HF search, compatibility, downloader, registry
│   ├── training/                   ← Trainer interface, subprocess worker, checkpoint mgr
│   ├── inference/                  ← Model loader (base+adapter), streaming chat engine
│   ├── export/                     ← Adapter ZIP, merged model, full model export
│   └── storage/                    ← SQLite schema + CRUD repository
│
├── workspace/                      ← Runtime data (gitignored)
│   ├── datasets/                   ← Generated JSONL datasets
│   ├── base_models/                ← Downloaded HF models
│   ├── fine_tuned_models/          ← Training outputs
│   ├── exports/                    ← ZIP exports
│   └── logs/                       ← Training logs
│
├── tests/                          ← Unit tests (pytest)
│
│ ── ORIGINAL FILES (unchanged) ────────────────────────────────
├── train.py                        ← Original QLoRA trainer
├── train_new.py                    ← CodeLlama variant
├── inference.py                    ← CLI inference script
├── inference_9800.py               ← 9800-domain inference CLI
├── prepare_data.py                 ← PDF extraction + JSONL writer
├── generate_dataset.py             ← Hand-crafted 300-sample QA bank
├── requirements.txt                ← Original ML dependencies
├── train.jsonl                     ← Starter dataset (3 examples)
├── train_9800.jsonl                ← Domain dataset
├── train_9800_300.jsonl            ← Full 300-sample domain dataset
├── qwen25-0.5b-9800-qlora/        ← Existing trained adapter v1
└── qwen25-0.5b-9800v2-qlora/      ← Existing trained adapter v2
```

---

## 🗺️ Application Pages

### Page 1 — Dashboard (`app.py`)
- Statistics: datasets, models, training runs
- Navigation cards to all pages
- Recent training activity feed

### Page 2 — Dataset Generation
1. Upload PDF / TXT / DOCX files (multiple at once)
2. Text extraction with preview
3. Configure: dataset name, question count, types, difficulty, custom instructions
4. Azure OpenAI generates questions in batches with retry/backoff
5. Review, edit, delete individual examples
6. Download as JSONL or save for training

### Page 3 — Model Fine-Tuning
1. **Tab 1** — Select a saved dataset
2. **Tab 2** — Search HuggingFace, check compatibility, download model
3. **Tab 3** — Configure QLoRA / LoRA / Full SFT, launch training
4. **Tab 4** — Live log stream, cancel training

### Page 4 — Fine-Tuned Models
- All registered models with status badges
- Load for playground, unload, export (ZIP), merge (LoRA only), delete

### Page 5 — Model Playground
- Select and load a fine-tuned model
- Streaming chat interface
- Configurable: temperature, top-p, max tokens, repetition penalty
- Separate system prompt

### Page 6 — Training History & Settings
- All training runs with log viewer
- All datasets with download
- Azure and storage configuration display

---

## 🔐 Azure OpenAI Configuration

| Variable | Value |
|----------|-------|
| `AZURE_OPENAI_API_KEY` | Your Azure key (never logged or displayed) |
| `AZURE_OPENAI_ENDPOINT` | `https://sanjeet.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | `Astra_IDPT` |
| `AZURE_OPENAI_API_VERSION` | `2025-03-01-preview` |

Validate the connection from the Dataset Generation page → **Azure OpenAI Configuration** expander → **Validate Azure Connection**.

---

## 🤗 Hugging Face Authentication

For gated models (Llama, Gemma, etc.):

1. Accept the model license on [huggingface.co](https://huggingface.co)
2. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Add to `.env`: `HF_TOKEN=hf_...`
4. Restart Streamlit

---

## 🏋️ Training Methods

| Method | VRAM | Output | Notes |
|--------|------|--------|-------|
| **QLoRA SFT** | ~0.6 bytes/param | Adapter (~17 MB) | Recommended; uses 4-bit NF4 |
| **LoRA SFT** | ~2 bytes/param | Adapter | Better quality, more VRAM |
| **Full SFT** | ~8 bytes/param | Full weights | All params updated |

**Dataset schema** (compatible with all existing trainers):
```jsonl
{"messages":[
  {"role":"system","content":"..."},
  {"role":"user","content":"..."},
  {"role":"assistant","content":"..."}
]}
```

---

## 📦 Export Options

| Type | When Available | Output |
|------|---------------|--------|
| **Adapter ZIP** | QLoRA or LoRA | `adapter_model.safetensors` + config + README |
| **Merged Model** | LoRA (not QLoRA) | Full bf16 weights — no base model needed |
| **Full Model ZIP** | Full SFT | Complete checkpoint + tokenizer |

> **Note:** QLoRA adapters cannot be merged reliably from a 4-bit quantized base. Use LoRA SFT if you need a merged model.

---

## 🔬 Running Tests

```bash
source venv/bin/activate
cd ~/Sreevatsav/qlora-qwen3
python -m pytest tests/ -v
```

Tests cover:
- Document extraction (TXT, chunking)
- LLM response parsing and deduplication
- Dataset schema validation
- Model compatibility checking
- Training configuration validation
- Export ZIP creation

---

## ⚡ Quick Smoke Test (5-question workflow)

```bash
# 1. Start the app
streamlit run app.py

# 2. Open http://localhost:8501
# 3. Go to Dataset Generation
# 4. Upload a small TXT file
# 5. Set "Number of Questions" to 5
# 6. Click Generate
# 7. Review and Save dataset

# 8. Go to Model Fine-Tuning
# 9. Select the dataset
# 10. Select Qwen/Qwen2.5-0.5B-Instruct (already downloaded)
# 11. Configure QLoRA, set epochs=1
# 12. Launch training

# 13. Go to Fine-Tuned Models → Load the adapter
# 14. Go to Model Playground → chat
# 15. Go to Fine-Tuned Models → Export ZIP
```

---

## 🔧 What Was Modified / Added

**No existing files were changed.** All new code is additive:

| New File | Purpose |
|----------|---------|
| `app.py` | Streamlit entry point + dashboard |
| `pages/1–5_*.py` | Five feature pages |
| `src/config.py` | Central configuration |
| `src/storage/` | SQLite schema + repository |
| `src/document_processing/` | PDF/TXT/DOCX extraction + chunking |
| `src/dataset_generation/` | Azure client + prompts + validator + adapter |
| `src/model_management/` | HF search + compatibility + downloader + registry |
| `src/training/` | Trainer interface + subprocess worker + checkpoint manager |
| `src/inference/` | Model loader + streaming chat engine |
| `src/export/` | Adapter / merged / full model export |
| `tests/` | 5 test modules |
| `requirements_app.txt` | App-only deps (streamlit, openai, python-docx, dotenv) |
| `.env.example` | Credential template |
| `.gitignore` | Updated to exclude model weights, datasets, secrets |

---

## 🛡️ Security Notes

- API key is **never** shown in the UI, logged, or embedded in datasets
- Uploaded document filenames are sanitized before filesystem use
- Document content is sandboxed in prompts (labelled as data, not instructions)
- Model weights and datasets are excluded from Git via `.gitignore`
- Gated model tokens are scrubbed from error messages

---

*Built on: DGX Spark (NVIDIA GB10, aarch64, CUDA 13.0, 130 GB VRAM)*
