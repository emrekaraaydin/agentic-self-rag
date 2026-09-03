import logging
from typing import Literal
from config import settings
from src.state.state import GraphState

logger = logging.getLogger(__name__)


def route_after_retrieval(state: GraphState) -> Literal["generate_node", "__end__"]:
    # Reranker sonrasi dokumanlarin yeterliligi
    is_relevant: bool = state.get("is_relevant", False)

    if is_relevant:
        logger.info("Documents passed the rerank threshold. Routing to generate_node.")
        return "generate_node"

    logger.warning("No relevant documents found. Ending graph.")
    return "__end__"


def route_after_hallucination(state: GraphState) -> Literal["generate_node", "__end__"]:
    # Halusinasyon denetimi
    has_hallucination: bool = state.get("has_hallucination", False)
    generation_hallucination_retry_count: int = state.get("generation_hallucination_retry_count", 0)

    if has_hallucination:
        if generation_hallucination_retry_count >= settings.HALLUCINATION_MAX_RETRY:
            logger.warning(
                "Max hallucination retries reached (%d). Ending graph.",
                generation_hallucination_retry_count,
            )
            return "__end__"
        logger.info("Hallucination detected. Routing back to generate_node.")
        return "generate_node"

    logger.info("No hallucination detected. Ending process.")
    return "__end__"