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
from src.chains.factory import create_hallucination_grader_chain
from src.chains.models import GradeHallucinations
from src.state.state import GraphState

logger = logging.getLogger(__name__)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.5, max=3.0),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _invoke_hallucination_grader(context: str, generation: str) -> GradeHallucinations:
    # LLM cagrisi icin izole edilmis asenkron fonksiyon
    chain = create_hallucination_grader_chain()
    return await chain.ainvoke({"documents": context, "generation": generation})


async def grade_hallucination_node(state: GraphState) -> Dict[str, Any]:
    # Uretilen yanitin yalnizca baglam havuzuna sadik kalip kalmadigini denetler
    logger.info("Executing async hallucination verification...")

    generation: str = state.get("generation") or ""
    documents: List[Document] = state.get("documents", [])
    current_retry_count: int = state.get("generation_hallucination_retry_count", 0)

    if not generation.strip():
        logger.warning("Empty generation received in hallucination grader.")
        return {
            "has_hallucination": True,
            "generation_hallucination_retry_count": current_retry_count + 1,
            "audit_logs": [
                {
                    "node": "grade_hallucination_node",
                    "has_hallucination": True,
                    "reasoning": "Empty generation.",
                    "success": False,
                }
            ],
        }

    context: str = "\n\n".join(doc.page_content for doc in documents)

    result: GradeHallucinations = await _invoke_hallucination_grader(
        context=context, generation=generation
    )

    has_hallucination: bool = result.binary_score.strip().lower() != "yes"

    logger.info("Hallucination check completed. Has hallucination: %s", has_hallucination)

    log_entry: Dict[str, Any] = {
        "node": "grade_hallucination_node",
        "has_hallucination": has_hallucination,
        "reasoning": result.reasoning,
        "success": True,
    }

    return {
        "has_hallucination": has_hallucination,
        "generation_hallucination_retry_count": current_retry_count + 1 if has_hallucination else current_retry_count,
        "audit_logs": [log_entry],
    }