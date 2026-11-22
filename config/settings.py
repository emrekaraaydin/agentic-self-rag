"""Centralized application settings loaded from environment variables."""

import os
from pathlib import Path
from typing import Final
from dotenv import load_dotenv

# --- Environment Loading ---
load_dotenv()

# --- Workspace Paths & Local Model Cache ---
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
FLASHRANK_CACHE_DIR: Final[Path] = BASE_DIR / ".cache" / "flashrank"
FLASHRANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Ollama LLM, SLM & Embedding Parameters ---
OLLAMA_BASE_URL: Final[str] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL: Final[str] = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
VECTOR_DIMENSION: Final[int] = int(os.getenv("VECTOR_DIMENSION", "768"))

LLM_MODEL: Final[str] = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
LLM_NUM_CTX: Final[int] = int(os.getenv("LLM_NUM_CTX", "16384"))
LLM_TEMPERATURE: Final[float] = float(os.getenv("LLM_TEMPERATURE", "0.0"))

SLM_MODEL: Final[str] = os.getenv("SLM_MODEL", "qwen2.5:3b-instruct")
SLM_NUM_CTX: Final[int] = int(os.getenv("SLM_NUM_CTX", "2048"))
SLM_MIN_TEMPERATURE: Final[float] = float(os.getenv("SLM_MIN_TEMPERATURE", "0.0"))

REQUEST_TIMEOUT: Final[float] = float(os.getenv("REQUEST_TIMEOUT", "60.0"))

# --- FlashRank Cross-Encoder Reranking ---
RERANK_TOP_N: Final[int] = int(os.getenv("RERANK_TOP_N", "8"))
RERANK_MIN_DOCS: Final[int] = int(os.getenv("RERANK_MIN_DOCS", "3"))
RERANK_DEFAULT_THRESHOLD: Final[float] = float(os.getenv("RERANK_DEFAULT_THRESHOLD", "0.03"))
RERANK_REWRITE_THRESHOLD: Final[float] = float(os.getenv("RERANK_REWRITE_THRESHOLD", "0.05"))

# --- ModernBERT-large-nli Hallucination Verification ---
NLI_MODEL_NAME: Final[str] = os.getenv("NLI_MODEL_NAME", "tasksource/ModernBERT-large-nli")
NLI_MAX_LENGTH: Final[int] = int(os.getenv("NLI_MAX_LENGTH", "8192"))
NLI_CONTRA_THRESHOLD: Final[float] = float(os.getenv("NLI_CONTRA_THRESHOLD", "0.25"))
NLI_ENTAIL_THRESHOLD: Final[float] = float(os.getenv("NLI_ENTAIL_THRESHOLD", "0.40"))

# --- Qdrant Vector Store & Agentic Retry Budgets ---
QDRANT_HOST: Final[str] = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: Final[int] = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME: Final[str] = os.getenv("COLLECTION_NAME", "rag_documents")
RETRIEVAL_TOP_K: Final[int] = int(os.getenv("RETRIEVAL_TOP_K", "50"))
QUERY_REWRITE_MAX_RETRY: Final[int] = int(os.getenv("QUERY_REWRITE_MAX_RETRY", "1"))
HALLUCINATION_MAX_RETRY: Final[int] = int(os.getenv("HALLUCINATION_MAX_RETRY", "2"))
MAX_SUB_QUERIES: Final[int] = int(os.getenv("MAX_SUB_QUERIES", "3"))
RETRIEVAL_REFLECTION_MAX_RETRY: Final[int] = int(os.getenv("RETRIEVAL_REFLECTION_MAX_RETRY", "1"))