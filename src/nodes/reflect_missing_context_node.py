"""Reflective retrieval planner that generates targeted search queries for missing multi-part context."""

import logging
from typing import Any, Dict, List
from langchain_core.documents import Document
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from src.chains.factory import create_missing_context_reflector_chain
from src.chains.models import MissingContextPlan
from src.state.state import GraphState

logger = logging.getLogger(__name__)


# --- Reflection Invocation & Document Retention ---
@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.2, max=1.5),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _invoke_reflector(question: str, previous_queries: str, draft_answer: str) -> MissingContextPlan:
    """Invoke the structured missing-context reflector chain with backoff retries."""
    chain = create_missing_context_reflector_chain()
    return await chain.ainvoke(
        {
            "question": question,
            "previous_queries": previous_queries,
            "draft_answer": draft_answer,
        }
    )


def _select_retained_documents(documents: List[Document], max_docs: int = 4) -> List[Document]:
    """Keep top unique context documents to merge with newly retrieved evidence."""
    retained: List[Document] = []
    seen = set()
    for doc in documents:
        cid = doc.metadata.get("chunk_id") or doc.metadata.get("id") or doc.page_content
        if cid not in seen:
            seen.add(cid)
            retained.append(doc)
        if len(retained) >= max_docs:
            break
    return retained


# --- Graph Node ---
async def reflect_missing_context_node(state: GraphState) -> Dict[str, Any]:
    """Produce follow-up search queries for statutes or facts flagged missing in a draft answer."""
    original_question: str = (state.get("original_question") or state.get("question") or "").strip()
    draft_answer: str = (state.get("generation") or "").strip()
    prev_queries: List[str] = list(state.get("sub_queries") or [original_question])
    documents: List[Document] = list(state.get("documents") or [])
    current_retry: int = state.get("retrieval_reflection_retry_count", 0)

    missing_sub_queries: List[str] = []
    success = True
    try:
        plan = await _invoke_reflector(
            question=original_question,
            previous_queries="\n".join(prev_queries),
            draft_answer=draft_answer,
        )
        seen = set()
        for q in plan.missing_sub_queries:
            cleaned = (q or "").strip()
            low = cleaned.lower()
            if cleaned and low not in seen:
                seen.add(low)
                missing_sub_queries.append(cleaned)
            if len(missing_sub_queries) >= 2:
                break
    except Exception as exc:
        success = False
        logger.warning("Missing context reflection failed (%s); falling back to original question.", exc)

    if not missing_sub_queries:
        missing_sub_queries = [original_question] if original_question else []

    retained_documents = _select_retained_documents(documents, max_docs=4)

    logger.info(
        "Reflective re-retrieval triggered (retry=%d) | missing_sub_queries=%s | retained_docs=%d",
        current_retry + 1,
        missing_sub_queries,
        len(retained_documents),
    )

    log_entry: Dict[str, Any] = {
        "node": "reflect_missing_context_node",
        "missing_sub_queries": missing_sub_queries,
        "retained_doc_count": len(retained_documents),
        "retry_count": current_retry + 1,
        "success": success and bool(missing_sub_queries),
    }

    return {
        "sub_queries": missing_sub_queries,
        "retained_documents": retained_documents,
        "retrieval_reflection_retry_count": current_retry + 1,
        "needs_additional_retrieval": False,
        "audit_logs": [log_entry],
    }
