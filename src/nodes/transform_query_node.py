import logging
from typing import Any, Dict
from config import settings
from src.chains.factory import create_query_rewriter_chain
from src.state.state import GraphState

logger = logging.getLogger(__name__)


async def transform_query_node(state: GraphState) -> Dict[str, Any]:
    logger.info("Executing async query transformation...")
    question: str = state.get("question", "")
    transformed_query: str = question

    try:
        rewriter_chain = create_query_rewriter_chain()
        result = await rewriter_chain.ainvoke({"question": transformed_query})

        if result and result.strip():
            transformed_query = result.strip()
            logger.info(
                "Query transformed from '%s' to '%s' (temperature: %.2f)",
                question,
                transformed_query,
                settings.SLM_MIN_TEMPERATURE,
            )
    except Exception as exc:
        logger.error("Query transformation failed: %s", exc, exc_info=True)

    log_entry: Dict[str, Any] = {
        "node": "transform_query_node",
        "original_query": question,
        "transformed_query": transformed_query,
        "temperature": settings.SLM_MIN_TEMPERATURE,
        "success": transformed_query != question,
    }

    return {
        "question": transformed_query,
        "audit_logs": [log_entry],
    }