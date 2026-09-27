import asyncio
import logging
from typing import Any, Dict, List, Set, Tuple
from langchain_core.documents import Document
from config.settings import RETRIEVAL_TOP_K
from src.indexing.vector_store import get_vector_store
from src.state.state import GraphState
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _format_nomic_query(query: str) -> str:
    # Nomic asimetrik arama icin zorunlu gorev oneki kontrolu
    cleaned_query = query.strip()
    if cleaned_query.startswith("search_query:"):
        return cleaned_query
    return f"search_query: {cleaned_query}"


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.2, max=2.0),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _fetch_documents_with_retry(query: str, top_k: int) -> List[Tuple[Document, float]]:
    # Vektor deposundan ilgili sorgu icin benzerlik aramasi yapar
    vector_store = get_vector_store()
    formatted_query = _format_nomic_query(query)
    return await vector_store.asimilarity_search_with_score(
        query=formatted_query, k=top_k
    )


async def retrieve_node(state: GraphState) -> Dict[str, Any]:
    # Nomic onekini uygulayarak ve donusturulen/orijinal sorgulari birlestirerek aday havuzunu toplar
    transformed_query: str = state.get("question", "").strip()
    original_query: str = state.get("original_question", "").strip()

    search_queries: List[str] = [transformed_query]
    if original_query and original_query != transformed_query:
        search_queries.append(original_query)

    logger.info(
        "Executing async vector retrieval for queries: %s with Top-K: %d",
        search_queries,
        RETRIEVAL_TOP_K,
    )

    # Sorgulari eszamanli calistir
    search_tasks = [
        _fetch_documents_with_retry(query=query_text, top_k=RETRIEVAL_TOP_K)
        for query_text in search_queries
    ]
    nested_results: List[List[Tuple[Document, float]]] = await asyncio.gather(*search_tasks)

    # Sonuclari birlestirip skora gore azalan sirala
    all_scored_docs: List[Tuple[Document, float]] = [
        item for sublist in nested_results for item in sublist
    ]
    all_scored_docs.sort(key=lambda item: float(item[1]), reverse=True)

    # Chunk ID uzerinden tekillestirme
    seen_ids: Set[str] = set()
    merged_documents: List[Document] = []

    for doc, score in all_scored_docs:
        chunk_id: str = (
            doc.metadata.get("chunk_id")
            or doc.metadata.get("id")
            or doc.page_content
        )
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            doc.metadata["retrieval_score"] = float(score)
            merged_documents.append(doc)

    selected_documents = merged_documents[:RETRIEVAL_TOP_K]

    logger.info(
        "Retrieved %d unique candidate documents across queries.",
        len(selected_documents),
    )
    for idx, doc in enumerate(selected_documents[:5]):
        content_preview = doc.page_content[:250].replace("\n", " ")
        score = doc.metadata.get("retrieval_score", 0.0)
        logger.info(
            "Retrieved Doc [%d] - Score: %.4f - Content: %s...",
            idx,
            score,
            content_preview,
        )

    log_entry: Dict[str, Any] = {
        "node": "retrieve_node",
        "queries_used": search_queries,
        "retrieved_doc_count": len(selected_documents),
        "success": len(selected_documents) > 0,
    }

    return {
        "documents": selected_documents,
        "audit_logs": [log_entry],
    }