import asyncio
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models

import config.settings
from src.indexing.loader import load_all_pdfs
from src.indexing.splitter import chunk_documents
from src.indexing.vector_store import get_async_qdrant_client, get_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 200


async def seed_vector_store() -> None:
    # Qdrant koleksiyonunu sıfırlar, markdown belgeleri yükler, Nomic önekiyle gömer ve Qdrant'a yazar.
    client: AsyncQdrantClient = get_async_qdrant_client()
    collection_name = config.settings.QDRANT_COLLECTION_NAME

    logger.info("Connecting to Qdrant via gRPC...")

    # 1. Koleksiyon kontrolü ve temiz kurulum
    collections_response = await client.get_collections()
    existing_collections = [c.name for c in collections_response.collections]

    if collection_name in existing_collections:
        logger.info("Collection '%s' already exists. Deleting for a fresh start...", collection_name)
        await client.delete_collection(collection_name=collection_name)

    logger.info("Creating collection '%s'...", collection_name)
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=rest_models.VectorParams(
            size=config.settings.VECTOR_DIMENSION,
            distance=rest_models.Distance.COSINE,
        ),
    )
    logger.info("Collection '%s' successfully created and ready for upsert.", collection_name)

    # 2. Dinamik kök dizin tespiti ve dokümanların yüklenmesi
    # src/seed/seed.py -> 2 üst dizin proje köküdür (root/data)
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"

    logger.info("Loading documents from '%s' using custom loader...", data_dir)
    raw_documents: List[Document] = load_all_pdfs(data_dir)

    if not raw_documents:
        logger.warning("No documents found in '%s'. Exiting seed process.", data_dir)
        return

    chunked_documents: List[Document] = chunk_documents(raw_documents)

    # 3. Nomic asimetrik indeksleme için ön ek ekleme ve toplu embedding üretimi
    logger.info("Generating embeddings via Ollama ('%s') in batches of %d...", config.settings.EMBEDDING_MODEL, EMBED_BATCH_SIZE)
    embedding_client = get_embedding_model()

    # Nomic-embed-text asimetrik arama için doküman ön eki
    texts_to_embed = [f"search_document: {doc.page_content}" for doc in chunked_documents]
    vectors: List[List[float]] = []
    total_chunks = len(texts_to_embed)
    for start_idx in range(0, total_chunks, EMBED_BATCH_SIZE):
        batch_texts = texts_to_embed[start_idx : start_idx + EMBED_BATCH_SIZE]
        batch_vectors = await embedding_client.aembed_documents(batch_texts)
        vectors.extend(batch_vectors)
        logger.info("Embedded %d/%d chunks...", min(start_idx + len(batch_texts), total_chunks), total_chunks)

    # 4. Qdrant Points hazırlama (Metadata alanları doğrudan payload köküne açılır)
    points = [
        rest_models.PointStruct(
            id=doc.metadata["chunk_id"],
            vector=vector,
            payload={
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                **doc.metadata,
            },
        )
        for doc, vector in zip(chunked_documents, vectors)
    ]

    # 5. Toplu yükleme (Batched Upsert)
    total_points = len(points)
    logger.info("Upserting %d points into Qdrant in batches of %d...", total_points, UPSERT_BATCH_SIZE)
    for start_idx in range(0, total_points, UPSERT_BATCH_SIZE):
        batch_points = points[start_idx : start_idx + UPSERT_BATCH_SIZE]
        await client.upsert(
            collection_name=collection_name,
            points=batch_points,
        )
        logger.info("Upserted %d/%d points...", min(start_idx + len(batch_points), total_points), total_points)
    logger.info("Seeding completed successfully! Total points inserted: %d", total_points)


if __name__ == "__main__":
    asyncio.run(seed_vector_store())