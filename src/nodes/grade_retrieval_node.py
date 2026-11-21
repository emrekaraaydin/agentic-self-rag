import asyncio
import logging
import math
from typing import Any, Dict, List, Set
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from config.settings import (
    FLASHRANK_CACHE_DIR,
    RERANK_DEFAULT_THRESHOLD,
    RERANK_MIN_DOCS,
    RERANK_REWRITE_THRESHOLD,
    RERANK_TOP_N,
)
from src.state.state import GraphState

logger = logging.getLogger(__name__)
ranker_instance = Ranker(
    model_name="ms-marco-MiniLM-L-12-v2",  # 12 katmanlı gerçek Cross-Encoder (~34MB)
    max_length=1024,                       # 1600 karakterlik chunk'ların tamamını okuması için!
    cache_dir=str(FLASHRANK_CACHE_DIR),
)

_SUBQUERY_PAD_FLOOR = 0.005


def _doc_chunk_id(doc: Document) -> str:
    return str(doc.metadata.get("chunk_id") or doc.metadata.get("id") or doc.page_content)


async def _rerank_candidates(query: str, docs: List[Document]) -> List[Dict[str, Any]]:
    if not docs:
        return []
    passages = [
        {"id": idx, "text": doc.page_content, "metadata": doc.metadata}
        for idx, doc in enumerate(docs)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    return await asyncio.to_thread(ranker_instance.rerank, rerank_request)


def _round_robin_merge(
    buckets: List[List[Document]],
    max_docs: int,
) -> List[Document]:
    merged: List[Document] = []
    seen_ids: Set[str] = set()
    max_bucket_len = max((len(b) for b in buckets), default=0)

    for pos in range(max_bucket_len):
        for bucket in buckets:
            if pos >= len(bucket):
                continue
            doc = bucket[pos]
            cid = _doc_chunk_id(doc)
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(doc)
                if len(merged) >= max_docs:
                    return merged
    return merged


def _build_sources(selected_documents: List[Document]) -> List[Dict[str, Any]]:
    return [
        {
            "index": idx + 1,
            "chunk_id": doc.metadata.get("chunk_id") or doc.metadata.get("id") or f"doc_{idx}",
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "rerank_score": doc.metadata.get("rerank_score"),
            "matched_sub_query": doc.metadata.get("matched_sub_query"),
            "content": doc.page_content,
        }
        for idx, doc in enumerate(selected_documents)
    ]


def _item_to_document(item: Dict[str, Any], matched_sub_query: str = "") -> Document:
    doc_metadata = dict(item.get("metadata", {}))
    doc_metadata["rerank_score"] = float(item["score"])
    if matched_sub_query and not doc_metadata.get("matched_sub_query"):
        doc_metadata["matched_sub_query"] = matched_sub_query
    return Document(page_content=item["text"], metadata=doc_metadata)


async def grade_retrieval_node(state: GraphState) -> Dict[str, Any]:
    # Aday dokumanlari Cross-Encoder ile puanlayip tekil veya alt-sorgu dengeli en iyi dokumanlari secer
    query: str = state["question"]
    documents: List[Document] = state.get("documents", [])
    sub_query_documents: Dict[str, List[Document]] = state.get("sub_query_documents") or {}
    retained_documents: List[Document] = state.get("retained_documents") or []
    rewrite_retry_count: int = state.get("query_rewrite_retry_count", 0)

    active_threshold: float = (
        RERANK_REWRITE_THRESHOLD if rewrite_retry_count > 0 else RERANK_DEFAULT_THRESHOLD
    )

    if not documents and not retained_documents:
        logger.warning("No input documents to grade.")
        return {
            "documents": [],
            "sources": [],
            "is_relevant": False,
            "audit_logs": [
                {
                    "node": "grade_retrieval_node",
                    "applied_threshold": active_threshold,
                    "graded_doc_count": 0,
                    "passed_doc_count": 0,
                    "is_relevant": False,
                }
            ],
        }

    # Case B: Reflective Re-Retrieval (merge retained_documents with newly retrieved missing chunks)
    if retained_documents:
        sq_items = list(sub_query_documents.items()) if sub_query_documents else [(query, documents)]
        rerank_tasks = [_rerank_candidates(sq, sq_docs) for sq, sq_docs in sq_items]
        all_reranked = await asyncio.gather(*rerank_tasks)

        new_buckets: List[List[Document]] = []
        per_sq_selected: Dict[str, int] = {}
        for (sq, _), reranked in zip(sq_items, all_reranked):
            passed = [
                _item_to_document(item, sq)
                for item in reranked
                if float(item["score"]) >= active_threshold
            ][:4]
            if not passed:
                passed = [
                    _item_to_document(item, sq)
                    for item in reranked
                    if float(item["score"]) >= _SUBQUERY_PAD_FLOOR
                ][:2]
            new_buckets.append(passed)
            per_sq_selected[sq] = len(passed)

        new_merged = _round_robin_merge(new_buckets, max_docs=4)
        selected_documents = _round_robin_merge(
            [retained_documents[:4], new_merged],
            max_docs=RERANK_TOP_N,
        )
        is_relevant = len(selected_documents) > 0
        sources = _build_sources(selected_documents)

        logger.info(
            "Reflective retrieval merge completed | retained=%d | newly_selected=%d | total=%d",
            len(retained_documents[:4]),
            len(new_merged),
            len(selected_documents),
        )
        return {
            "documents": selected_documents,
            "sources": sources,
            "retained_documents": [],
            "is_relevant": is_relevant,
            "audit_logs": [
                {
                    "node": "grade_retrieval_node",
                    "mode": "reflective_merge",
                    "applied_threshold": active_threshold,
                    "graded_doc_count": sum(len(v) for _, v in sq_items),
                    "passed_doc_count": len(selected_documents),
                    "per_sub_query_selected": per_sq_selected,
                    "is_relevant": is_relevant,
                    "top_scores": [doc.metadata.get("rerank_score") for doc in selected_documents],
                }
            ],
        }

    # Case A: Balanced Per-Subquery Reranking for decomposed multi-part queries
    if len(sub_query_documents) >= 2:
        sq_items = list(sub_query_documents.items())
        n_sq = len(sq_items)
        per_sq_limit = max(2, math.ceil(RERANK_TOP_N / n_sq))
        min_balanced = min(2, per_sq_limit)

        logger.info(
            "Balanced per-subquery grading for %d sub-queries (limit=%d, threshold=%.2f)",
            n_sq,
            per_sq_limit,
            active_threshold,
        )

        all_reranked = await asyncio.gather(
            *[_rerank_candidates(sq, sq_docs) for sq, sq_docs in sq_items]
        )

        any_passed_threshold = any(
            float(item["score"]) >= active_threshold
            for reranked in all_reranked
            for item in reranked
        )

        buckets: List[List[Document]] = []
        per_sq_selected: Dict[str, int] = {}
        total_graded = sum(len(docs) for _, docs in sq_items)

        for (sq, _), reranked in zip(sq_items, all_reranked):
            if reranked:
                logger.info(
                    "Sub-query '%s' Top-1 Cross-Encoder Score: %.6f | Content: %s...",
                    sq[:70],
                    float(reranked[0]["score"]),
                    reranked[0]["text"][:100].replace("\n", " "),
                )
            passed = [
                _item_to_document(item, sq)
                for item in reranked
                if float(item["score"]) >= active_threshold
            ][:per_sq_limit]

            if any_passed_threshold and len(passed) < min_balanced:
                existing_texts = {d.page_content for d in passed}
                for item in reranked:
                    if len(passed) >= min_balanced:
                        break
                    if float(item["score"]) >= _SUBQUERY_PAD_FLOOR and item["text"] not in existing_texts:
                        passed.append(_item_to_document(item, sq))
                        existing_texts.add(item["text"])

            buckets.append(passed)
            per_sq_selected[sq] = len(passed)

        selected_documents = _round_robin_merge(buckets, max_docs=RERANK_TOP_N) if any_passed_threshold else []
        is_relevant = any_passed_threshold and len(selected_documents) > 0
        sources = _build_sources(selected_documents)

        logger.info(
            "Balanced grading completed | selected=%d/%d | per_sub_query=%s",
            len(selected_documents),
            total_graded,
            per_sq_selected,
        )

        return {
            "documents": selected_documents,
            "sources": sources,
            "is_relevant": is_relevant,
            "audit_logs": [
                {
                    "node": "grade_retrieval_node",
                    "mode": "balanced_subquery",
                    "applied_threshold": active_threshold,
                    "graded_doc_count": total_graded,
                    "passed_doc_count": len(selected_documents),
                    "per_sub_query_selected": per_sq_selected,
                    "is_relevant": is_relevant,
                    "top_scores": [doc.metadata.get("rerank_score") for doc in selected_documents],
                }
            ],
        }

    # Case C: Single query standard reranking
    single_query = next(iter(sub_query_documents.keys()), query) if sub_query_documents else query
    single_docs = next(iter(sub_query_documents.values()), documents) if sub_query_documents else documents

    logger.info(
        "Grading %d documents | Active Threshold: %.2f | Query: '%s'",
        len(single_docs),
        active_threshold,
        single_query,
    )

    rerank_results = await _rerank_candidates(single_query, single_docs)

    logger.info("--- Top 5 FlashRank Scores ---")
    for idx, item in enumerate(rerank_results[:5]):
        raw_score: float = float(item["score"])
        metadata: Dict[str, Any] = item.get("metadata", {})
        qdrant_score: float = float(metadata.get("retrieval_score", 0.0))
        content_preview: str = item["text"][:120].replace("\n", " ")
        logger.info(
            "Rank [%d] - Cross-Encoder Score: %.6f | Qdrant Cosine: %.4f | Content: %s...",
            idx,
            raw_score,
            qdrant_score,
            content_preview,
        )

    filtered_documents: List[Document] = [
        _item_to_document(item, single_query)
        for item in rerank_results
        if float(item["score"]) >= active_threshold
    ]

    is_relevant: bool = len(filtered_documents) > 0
    if is_relevant and len(filtered_documents) < RERANK_MIN_DOCS:
        target_count = min(RERANK_MIN_DOCS, len(rerank_results), RERANK_TOP_N)
        for item in rerank_results[len(filtered_documents) : target_count]:
            filtered_documents.append(_item_to_document(item, single_query))

    selected_documents = filtered_documents[:RERANK_TOP_N]
    sources = _build_sources(selected_documents)
    logger.info(
        "Cross-Encoder grading completed. %d/%d passed the threshold (%.2f).",
        len(selected_documents),
        len(single_docs),
        active_threshold,
    )

    log_entry: Dict[str, Any] = {
        "node": "grade_retrieval_node",
        "mode": "single_query",
        "applied_threshold": active_threshold,
        "graded_doc_count": len(single_docs),
        "passed_doc_count": len(selected_documents),
        "is_relevant": is_relevant,
        "top_scores": [doc.metadata.get("rerank_score") for doc in selected_documents],
    }

    return {
        "documents": selected_documents,
        "sources": sources,
        "is_relevant": is_relevant,
        "audit_logs": [log_entry],
    }
