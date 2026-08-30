import logging
from functools import lru_cache
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from config.settings import (
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    QDRANT_HOST,
    QDRANT_PORT,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> OllamaEmbeddings:
    # Tekil embedding modeli ornegi
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_sync_qdrant_client() -> QdrantClient:
    # LangChain QdrantVectorStore icin gRPC destekli senkron istemci
    # LangChain asenkron sorgulari bu istemci uzerinden thread havuzuyla yonetir
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=True,
    )


@lru_cache(maxsize=1)
def get_async_qdrant_client() -> AsyncQdrantClient:
    # Veri yukleme (seed) islemleri icin asenkron istemci
    return AsyncQdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=True,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    # Calisma zamani (runtime) aramalar icin tekil vektor deposu ornegi
    return QdrantVectorStore(
        client=get_sync_qdrant_client(),
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=get_embedding_model(),
    )