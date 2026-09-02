import logging
from typing import Any, Dict
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from src.chains.factory import create_answer_grader_chain
from src.chains.models import GradeAnswer
from src.state.state import GraphState

logger = logging.getLogger(__name__)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.5, max=3.0),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _invoke_answer_grader(question: str, generation: str) -> GradeAnswer:
    # LLM cagrisi icin izole edilmis asenkron fonksiyon
    chain = create_answer_grader_chain()
    return await chain.ainvoke({"question": question, "generation": generation})


async def grade_answer_node(state: GraphState) -> Dict[str, Any]:
    # Uretilen ve halusinasyondan arinmis yanitin soruyu dogrudan cozup cozmedigini denetler
    logger.info("Executing async answer resolution evaluation...")

    question: str = state.get("question", "")
    generation: str = state.get("generation") or ""
    current_retry_count: int = state.get("generation_satisfaction_retry_count", 0)

    if not generation.strip():
        logger.warning("Empty generation received in answer grader.")
        return {
            "is_answer_satisfactory": False,
            "generation_satisfaction_retry_count": current_retry_count + 1,
            "audit_logs": [
                {
                    "node": "grade_answer_node",
                    "is_answer_satisfactory": False,
                    "reasoning": "Empty generation.",
                    "feedback": "No generation produced to evaluate.",
                    "success": False,
                }
            ],
        }

    result: GradeAnswer = await _invoke_answer_grader(
        question=question, generation=generation
    )

    is_satisfactory: bool = result.is_satisfactory

    logger.info("Answer evaluation completed. Is satisfactory: %s", is_satisfactory)
    if not is_satisfactory and result.feedback:
        logger.info("Feedback for reformulation: %s", result.feedback)

    log_entry: Dict[str, Any] = {
        "node": "grade_answer_node",
        "is_answer_satisfactory": is_satisfactory,
        "reasoning": result.reasoning,
        "feedback": result.feedback,
        "success": True,
    }

    return {
        "is_answer_satisfactory": is_satisfactory,
        "generation_satisfaction_retry_count": current_retry_count + 1 if not is_satisfactory else current_retry_count,
        "audit_logs": [log_entry],
    }