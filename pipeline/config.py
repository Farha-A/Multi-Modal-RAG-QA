"""
Pipeline Configuration
======================
Centralises all environment variables, directory paths, and default
constants used across the pipeline modules.

Variables are loaded from a ``.env`` file in the project root via
``python-dotenv`` and fall back to sensible defaults when the key is
absent.
"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# ── Suppress noisy library warnings ──────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ── Load .env from project root ─────────────────────────────────────────────
load_dotenv()

# ── Directory paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # project root (one level above pipeline/)
PDF_DIR = BASE_DIR / "Data" / "PDFs"
IMAGE_DIR = BASE_DIR / "Data" / "images"

# ── Model settings ──────────────────────────────────────────────────────────
COLPALI_MODEL_NAME = os.getenv("COLPALI_MODEL_NAME", "vidore/colqwen2-v1.0-hf")

# ── Qdrant settings ────────────────────────────────────────────────────────
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "docbank_colpali")

# ── Groq / LLM settings ────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.2-11b-vision-preview")

# ── Retrieval settings ─────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "3"))
