import logging
from typing import Any, Dict, List
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from config import settings
from src.chains.factory import create_query_decomposer_chain
from src.chains.models import QueryPlan
from src.state.state import GraphState

logger = logging.getLogger(__name__)


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.2, max=1.5),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _invoke_decomposer(question: str) -> QueryPlan:
    chain = create_query_decomposer_chain()
    return await chain.ainvoke({"question": question})


def _dedupe_clean_queries(queries: List[str], limit: int) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for q in queries:
        item = (q or "").strip()
        low = item.lower()
        if item and low not in seen:
            seen.add(low)
            cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


async def decompose_query_node(state: GraphState) -> Dict[str, Any]:
    """Karsilastirmali/coklu mevzuat sorularini bagimsiz alt sorgulara ayirir."""
    question: str = (state.get("question") or "").strip()
    is_multi_part = False
    sub_queries: List[str] = [question] if question else []
    success = True

    try:
        plan = await _invoke_decomposer(question)
        cleaned = _dedupe_clean_queries(plan.sub_queries, settings.MAX_SUB_QUERIES)
        if plan.is_multi_part and len(cleaned) >= 2:
            is_multi_part = True
            sub_queries = cleaned
        else:
            is_multi_part = False
            sub_queries = [question] if question else cleaned[:1]
    except Exception as exc:
        success = False
        logger.warning("Query decomposition failed (%s); falling back to single query.", exc)
        is_multi_part = False
        sub_queries = [question] if question else []

    logger.info(
        "Query decomposition completed | is_multi_part=%s | sub_queries=%s",
        is_multi_part,
        sub_queries,
    )

    log_entry: Dict[str, Any] = {
        "node": "decompose_query_node",
        "is_multi_part": is_multi_part,
        "sub_queries": sub_queries,
        "sub_query_count": len(sub_queries),
        "success": success,
    }

    return {
        "is_multi_part": is_multi_part,
        "sub_queries": sub_queries,
        "retained_documents": [],
        "retrieval_reflection_retry_count": state.get("retrieval_reflection_retry_count", 0),
        "needs_additional_retrieval": False,
        "audit_logs": [log_entry],
    }
