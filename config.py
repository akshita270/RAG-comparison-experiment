"""
config.py — Central configuration for the RAG comparison experiment.

All tunable parameters live here so any architecture file can import
a single source of truth. Use environment variables to override values
without touching this file.
"""
import os
from pathlib import Path

# Load a .env file if present — lets you keep API keys out of the shell config.
# This runs at import time so every module that imports config gets the keys.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv is optional; keys can still come from the shell environment

# ─────────────────────────────────────────────────────────────────────────────
# API Keys — pulled from environment so secrets never live in source code
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Which LLM provider to use for answer generation.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Model used by all 4 architectures to generate answers.
# gpt-4o-mini is fast and cheap — good for running 100 evaluations.
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini" if LLM_PROVIDER == "openai" else "claude-haiku-4-5-20251001"
)

# Stronger model used only for the LLM-as-judge step.
# We use a better model here because evaluation quality matters more.
JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    "gpt-4o" if LLM_PROVIDER == "openai" else "claude-sonnet-4-6"
)

# ─────────────────────────────────────────────────────────────────────────────
# Embedding Model
# ─────────────────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, 22M params, fast CPU inference, good quality.
# No API key needed — runs locally via sentence-transformers.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# Vector Store (ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_PATH      = "./data/chroma_db"   # Persistent storage path
COLLECTION_NAME  = "arxiv_papers"       # Name of the ChromaDB collection

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Parameters
# ─────────────────────────────────────────────────────────────────────────────
TOP_K        = 5    # Number of chunks to retrieve per query
CHUNK_SIZE   = 512  # Characters per chunk (~128 tokens for most encoders)
CHUNK_OVERLAP = 50  # Overlap prevents losing context at chunk boundaries

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Parameters
# ─────────────────────────────────────────────────────────────────────────────
NUM_PAPERS    = 25   # ArXiv papers to fetch
NUM_QUESTIONS = 25   # Evaluation questions to generate

# ArXiv search query — focuses on recent ML/AI papers
ARXIV_QUERY    = "machine learning large language models"
ARXIV_CATEGORY = "cs.LG"  # Machine Learning category

# ─────────────────────────────────────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
PAPERS_FILE   = DATA_DIR / "papers.json"
QUESTIONS_FILE = DATA_DIR / "questions.json"
RESULTS_FILE  = DATA_DIR / "rag_comparison_results.csv"
VIZ_DIR       = BASE_DIR / "visualizations" / "output"

# ─────────────────────────────────────────────────────────────────────────────
# Agentic RAG Parameters
# ─────────────────────────────────────────────────────────────────────────────
AGENT_MAX_STEPS = 5  # Maximum tool calls before forcing a final answer

# ─────────────────────────────────────────────────────────────────────────────
# Validation — warn early if no API key is set
# ─────────────────────────────────────────────────────────────────────────────
def validate():
    """Call this at startup to catch missing config before running experiments."""
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise EnvironmentError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise EnvironmentError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
