"""Single source of truth for settings shared by upload and search.

Both flows MUST use the same collection and embedding model, otherwise vectors
from different models end up in one index and similarity scores are meaningless.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = str(ROOT / "chroma_store")
COLLECTION_NAME = "lab_reports"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Original uploaded PDFs live here (NOT under data/reports, which is the seed corpus).
UPLOADS_DIR = ROOT / "data" / "uploads"

MAX_UPLOAD_MB = 10
MAX_PDF_PAGES = 50
DEFAULT_TOP_K = 5
MIN_SIMILARITY = 0.15  # cosine similarity below this is treated as "not relevant"

# ----------------------------------------------------------------------------- answer generation
# Seed corpus (also the only non-upload location a "View Original Report" file may come from).
SEED_REPORTS_DIR = ROOT / "data" / "reports"

import os as _os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # so LLM_* / OPENAI_API_KEY in .env are visible below

# Which model writes the final answer. Switch with env vars / .env - no code change:
#   openai             -> OpenAI chat model (default; same model RAGSearch already uses: gpt-4.1)
#   openai_compatible  -> any OpenAI-compatible server (vLLM / Ollama / LM Studio) serving Qwen or a
#                         fine-tuned Qwen; set LLM_BASE_URL and LLM_MODEL
#   qwen_local         -> Hugging Face Qwen loaded in-process, optional LoRA adapter (QWEN_ADAPTER_PATH)
#   none               -> no LLM: deterministic extraction only
LLM_PROVIDER = _os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = _os.getenv("LLM_MODEL", "gpt-4.1")
LLM_BASE_URL = _os.getenv("LLM_BASE_URL", "")
QWEN_MODEL_NAME = _os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
QWEN_ADAPTER_PATH = _os.getenv("QWEN_ADAPTER_PATH", "")
