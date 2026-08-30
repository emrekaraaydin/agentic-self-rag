import os
from typing import Final , Dict
from dotenv import load_dotenv

# .env dosyasindaki degiskenleri yukleme
load_dotenv()

# Ollama ve model parametreleri
OLLAMA_BASE_URL: Final[str] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL: Final[str] = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
VECTOR_DIMENSION: Final[int] = int(os.getenv("VECTOR_DIMENSION", "768"))

LLM_MODEL: Final[str] = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
LLM_NUM_CTX: Final[int] = int(os.getenv("LLM_NUM_CTX", "16384"))
LLM_TEMPERATURE: Final[float] = float(os.getenv("LLM_TEMPERATURE", "0.0"))

SLM_MODEL: Final[str] = os.getenv("SLM_MODEL", "qwen2.5:3b-instruct")
SLM_NUM_CTX: Final[int] = int(os.getenv("SLM_NUM_CTX", "2048"))
SLM_MIN_TEMPERATURE: Final[float] = float(os.getenv("SLM_MIN_TEMPERATURE", "0.0"))
SLM_MAX_TEMPERATURE: Final[float] = float(os.getenv("SLM_MAX_TEMPERATURE", "0.2"))

REQUEST_TIMEOUT: Final[float] = float(os.getenv("REQUEST_TIMEOUT", "60.0"))
RERANK_TOP_N: Final[int] = 5

RERANK_DEFAULT_THRESHOLD: Final[float] = 0.45
RERANK_THRESHOLD_MAP: Final[Dict[str, float]] = {
    "factoid": 0.70,
    "conceptual": 0.45,
    "tabular": 0.35,
}

# Vektor veritabani parametreleri
QDRANT_HOST: Final[str] = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: Final[int] = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME: Final[str] = os.getenv("COLLECTION_NAME", "RAW_DOCUMENTS")
RETRIEVAL_TOP_K: Final[int] = int(os.getenv("RETRIEVAL_TOP_K", "30"))


HALLUCINATION_MAX_RETRY: Final[int] = 2
SATISFACTION_MAX_RETRY:Final[int] = 2
RETRIEVAL_MAX_RETRY:Final[int] = 2