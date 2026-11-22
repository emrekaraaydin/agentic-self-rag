"""Cached Ollama embedding model and sync/async Qdrant client singletons."""

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


@lru_cache(maxsize=1)
def get_embedding_model() -> OllamaEmbeddings:
    """Return a cached OllamaEmbeddings client for dense vector encoding."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_sync_qdrant_client() -> QdrantClient:
    """Return a cached gRPC-enabled sync Qdrant client (thread-pooled for async LangChain searches)."""
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=True,
    )


@lru_cache(maxsize=1)
def get_async_qdrant_client() -> AsyncQdrantClient:
    """Return a cached async Qdrant client used during collection creation and bulk seeding."""
    return AsyncQdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=True,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    """Return the singleton QdrantVectorStore bound to the active dense collection."""
    return QdrantVectorStore(
        client=get_sync_qdrant_client(),
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=get_embedding_model(),
    )