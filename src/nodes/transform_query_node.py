import logging
from typing import Any, Dict
from config import settings
from src.chains.factory import create_query_rewriter_chain
from src.state.state import GraphState

logger = logging.getLogger(__name__)


async def transform_query_node(state: GraphState) -> Dict[str, Any]:
    # Geri besleme ve basarisiz deneme sayisina gore dinamik sicaklikla sorgu donusturme
    logger.info("Executing async query transformation...")

    # Her zaman asil niyeti korumak adina original_question oncelikli alinir
    question: str = state.get("original_question") or state.get("question", "")
    retrieval_retries: int = state.get("retrieval_retry_count", 0)
    satisfaction_retries: int = state.get("generation_satisfaction_retry_count", 0)

    # State'e geri yazilmak uzere deneme sayacini bir artiriyoruz
    current_retry_count: int = retrieval_retries + 1

    # Toplam basarisiz arama/cevap denemesi uzerinden dinamik sicaklik hesabi
    total_retries: int = current_retry_count + satisfaction_retries
    dynamic_temperature: float = min(
        settings.SLM_MIN_TEMPERATURE + (total_retries * 0.05),
        settings.SLM_MAX_TEMPERATURE,
    )

    transformed_query: str = question

    try:
        rewriter_chain = create_query_rewriter_chain(temperature=dynamic_temperature)
        result = await rewriter_chain.ainvoke({"question": question})
        if result and result.strip():
            transformed_query = result.strip()
            logger.info(
                "Query transformed from '%s' to '%s' (temperature: %.2f)",
                question,
                transformed_query,
                dynamic_temperature,
            )
    except Exception as exc:
        logger.error("Query transformation failed: %s", exc, exc_info=True)

    log_entry: Dict[str, Any] = {
        "node": "transform_query_node",
        "original_query": question,
        "transformed_query": transformed_query,
        "dynamic_temperature": dynamic_temperature,
        "success": transformed_query != question,
    }

    return {
        "question": transformed_query,
        "retrieval_retry_count": current_retry_count,
        "audit_logs": [log_entry],
    }