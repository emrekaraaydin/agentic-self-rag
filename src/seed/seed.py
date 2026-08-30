import asyncio
import logging
from typing import Any

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models

import config.settings
from src.indexing.splitter import chunk_documents
from src.indexing.vector_store import get_async_qdrant_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def seed_vector_store() -> None:
    # Qdrant koleksiyonunu hazirlar, dokumanlari parcalar, embedding alir ve yukler.
    client: AsyncQdrantClient = get_async_qdrant_client()
    collection_name = config.settings.QDRANT_COLLECTION_NAME

    logger.info("Connecting to Qdrant via gRPC...")

    # 1. Koleksiyon var mi kontrol et, yoksa olustur
    collections_response = await client.get_collections()
    existing_collections = [c.name for c in collections_response.collections]

    # Eger koleksiyon varsa tamamen sil
    if collection_name in existing_collections:
        logger.info(f"Collection '{collection_name}' already exists. Deleting for a fresh start...")
        await client.delete_collection(collection_name=collection_name)

    # Koleksiyonu temiz bir sekilde (yeniden) olustur
    logger.info(f"Creating collection '{collection_name}'...")
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=rest_models.VectorParams(
            size=config.settings.VECTOR_DIMENSION,
            distance=rest_models.Distance.COSINE,
        ),
    )
    logger.info(f"Collection '{collection_name}' successfully created and ready for upsert.")

    # 2. Dokumanlari Yukle ve Parcala
    logger.info("Loading documents from 'data/' directory...")
    loader = PyPDFDirectoryLoader("data/")
    raw_documents = loader.load()
    
    if not raw_documents:
        logger.warning("No documents found in 'data/' directory. Exiting.")
        return

    chunked_documents = chunk_documents(raw_documents)

    # 3. Embedding Uretimi (Ollama)
    logger.info(f"Generating embeddings via Ollama ('{config.settings.EMBEDDING_MODEL}')...")
    embedding_client = OllamaEmbeddings(
        model=config.settings.EMBEDDING_MODEL,
        base_url=config.settings.OLLAMA_BASE_URL,
    )

    texts_to_embed = [doc.page_content for doc in chunked_documents]
    vectors = await embedding_client.aembed_documents(texts_to_embed)

    # 4. Qdrant'a Noktalari (Points) Hazirla ve Toplu Yukle
    points = [
        rest_models.PointStruct(
            id=doc.metadata["chunk_id"],
            vector=vector,
            payload={
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            },
        )
        for doc, vector in zip(chunked_documents, vectors)
    ]

    logger.info(f"Upserting {len(points)} points into Qdrant...")
    await client.upsert(
        collection_name=collection_name,
        points=points,
    )
    logger.info(f"Seeding completed successfully! Total points inserted: {len(points)}")


if __name__ == "__main__":
    asyncio.run(seed_vector_store())