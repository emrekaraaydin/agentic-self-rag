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


def _deduplicate_scored_docs(
    scored_docs: List[Tuple[Document, float]],
    limit: int,
    matched_sub_query: str = "",
) -> List[Document]:
    sorted_docs = sorted(scored_docs, key=lambda item: float(item[1]), reverse=True)
    seen_ids: Set[str] = set()
    unique_docs: List[Document] = []
    for doc, score in sorted_docs:
        chunk_id: str = (
            doc.metadata.get("chunk_id")
            or doc.metadata.get("id")
            or doc.page_content
        )
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            doc_copy = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            doc_copy.metadata["retrieval_score"] = float(score)
            if matched_sub_query:
                doc_copy.metadata["matched_sub_query"] = matched_sub_query
            unique_docs.append(doc_copy)
        if len(unique_docs) >= limit:
            break
    return unique_docs


async def retrieve_node(state: GraphState) -> Dict[str, Any]:
    # Alt sorgulari (veya donusturulen/orijinal sorguyu) eszamanli getirir ve aday havuzunu olusturur
    sub_queries: List[str] = [
        q.strip() for q in (state.get("sub_queries") or []) if q and q.strip()
    ]
    transformed_query: str = state.get("question", "").strip()
    original_query: str = state.get("original_question", "").strip()
    rewrite_retry_count: int = state.get("query_rewrite_retry_count", 0)
    is_reflective_pass: bool = bool(state.get("retained_documents"))

    sub_query_documents: Dict[str, List[Document]] = {}

    if len(sub_queries) > 1:
        per_sq_k = max(25, RETRIEVAL_TOP_K // len(sub_queries))
        logger.info(
            "Executing decomposed async retrieval for %d sub-queries (per_sq_k=%d): %s",
            len(sub_queries),
            per_sq_k,
            sub_queries,
        )
        nested_results: List[List[Tuple[Document, float]]] = await asyncio.gather(
            *[_fetch_documents_with_retry(query=sq, top_k=per_sq_k) for sq in sub_queries]
        )

        merged_documents: List[Document] = []
        seen_union: Set[str] = set()
        for sq, scored_docs in zip(sub_queries, nested_results):
            sq_docs = _deduplicate_scored_docs(scored_docs, limit=per_sq_k, matched_sub_query=sq)
            sub_query_documents[sq] = sq_docs
            for doc in sq_docs:
                cid = doc.metadata.get("chunk_id") or doc.metadata.get("id") or doc.page_content
                if cid not in seen_union:
                    seen_union.add(cid)
                    merged_documents.append(doc)

        selected_documents = merged_documents
        queries_used = sub_queries
        per_sub_query_counts = {sq: len(docs) for sq, docs in sub_query_documents.items()}
    else:
        primary_query = sub_queries[0] if sub_queries else transformed_query
        search_queries: List[str] = [primary_query]
        if (
            not is_reflective_pass
            and rewrite_retry_count > 0
            and original_query
            and original_query != primary_query
        ):
            search_queries.append(original_query)

        logger.info(
            "Executing async vector retrieval for queries: %s with Top-K: %d",
            search_queries,
            RETRIEVAL_TOP_K,
        )
        nested_results = await asyncio.gather(
            *[_fetch_documents_with_retry(query=q, top_k=RETRIEVAL_TOP_K) for q in search_queries]
        )
        all_scored_docs: List[Tuple[Document, float]] = [
            item for sublist in nested_results for item in sublist
        ]
        selected_documents = _deduplicate_scored_docs(
            all_scored_docs, limit=RETRIEVAL_TOP_K, matched_sub_query=primary_query
        )
        if primary_query:
            sub_query_documents[primary_query] = selected_documents
        queries_used = search_queries
        per_sub_query_counts = {primary_query: len(selected_documents)} if primary_query else {}

    logger.info("Retrieved %d unique candidate documents across queries.", len(selected_documents))
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
        "queries_used": queries_used,
        "retrieved_doc_count": len(selected_documents),
        "per_sub_query_counts": per_sub_query_counts,
        "success": len(selected_documents) > 0,
    }

    return {
        "documents": selected_documents,
        "sub_query_documents": sub_query_documents,
        "audit_logs": [log_entry],
    }
