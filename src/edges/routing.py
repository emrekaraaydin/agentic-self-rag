import logging
from typing import Literal
from config import settings
from src.state.state import GraphState

logger = logging.getLogger(__name__)


def route_after_retrieval(state: GraphState,) -> Literal["generate_node", "transform_query_node", "fallback_node"]:
    # Reranker sonrasi dokumanlarin yeterliligi
    is_relevant: bool = state.get("is_relevant", False)
    query_rewrite_retry_count: int = state.get(
        "query_rewrite_retry_count", 0
    )

    if is_relevant:
        logger.info("Documents passed the rerank threshold. Routing to generate_node.")
        return "generate_node"

    if query_rewrite_retry_count >= settings.QUERY_REWRITE_MAX_RETRY:
        logger.warning(
            "Max query rewrite retries reached (%d). Routing to fallback_node.",
            query_rewrite_retry_count,
        )
        return "fallback_node"

    logger.info("No relevant documents found. Routing to transform_query_node.")
    return "transform_query_node"


def route_after_hallucination(
    state: GraphState,
) -> Literal["reflect_missing_context_node", "generate_node", "fallback_node", "__end__"]:
    # Eksik mevzuat / kismi ret durumunda yeniden arama kontrolu
    if (
        state.get("needs_additional_retrieval", False)
        and state.get("retrieval_reflection_retry_count", 0) < settings.RETRIEVAL_REFLECTION_MAX_RETRY
    ):
        logger.info("Missing context detected in multi-part response. Routing to reflect_missing_context_node.")
        return "reflect_missing_context_node"

    # Halusinasyon denetimi
    has_hallucination: bool = state.get("has_hallucination", False)
    generation_hallucination_retry_count: int = state.get(
        "generation_hallucination_retry_count", 0
    )

    if has_hallucination:
        if generation_hallucination_retry_count >= settings.HALLUCINATION_MAX_RETRY:
            logger.warning(
                "Max hallucination retries reached (%d). Routing to fallback_node.",
                generation_hallucination_retry_count,
            )
            return "fallback_node"

        logger.info("Hallucination detected. Routing back to generate_node.")
        return "generate_node"

    logger.info("No hallucination detected. Ending process.")
    return "__end__"