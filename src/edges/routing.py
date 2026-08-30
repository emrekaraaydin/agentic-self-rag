import logging
from typing import Literal
from config import settings
from src.state.state import GraphState

logger = logging.getLogger(__name__)


def route_after_retrieval(state: GraphState) -> Literal["generate_node", "transform_query_node", "__end__"]:
    # Reranker sonrasi dokumanlarin yeterliligi
    is_relevant: bool = state.get("is_relevant", False)
    retrieval_retry_count: int = state.get("retrieval_retry_count", 0)

    if is_relevant:
        logger.info("Documents passed the rerank threshold. Routing to generate_node.")
        return "generate_node"

    if retrieval_retry_count >= settings.RETRIEVAL_MAX_RETRY:
        logger.warning(
            "Max retrieval retries reached (%d). Ending graph without documents.",
            retrieval_retry_count,
        )
        return "__end__"

    logger.info("No relevant documents found. Routing to transform_query_node.")
    return "transform_query_node"


def route_after_hallucination(state: GraphState) -> Literal["generate_node", "grade_answer_node", "__end__"]:
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

    logger.info("No hallucination detected. Routing to grade_answer_node.")
    return "grade_answer_node"


def route_after_answer(state: GraphState) -> Literal["transform_query_node", "__end__"]:
    # Cevap yeterliligi denetimi
    is_satisfactory: bool = state.get("is_answer_satisfactory", False)
    generation_satisfaction_retry_count: int = state.get("generation_satisfaction_retry_count", 0)

    if is_satisfactory:
        logger.info("Answer is satisfactory. Ending process.")
        return "__end__"

    if generation_satisfaction_retry_count >= settings.SATISFACTION_MAX_RETRY:
        logger.warning(
            "Max satisfaction retries reached (%d). Returning best effort answer.",
            generation_satisfaction_retry_count,
        )
        return "__end__"

    logger.info("Answer is unsatisfactory. Routing to transform_query_node.")
    return "transform_query_node"