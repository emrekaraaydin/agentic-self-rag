"""Deterministic abstention fallback node when evidence is missing or retries are exhausted."""

from typing import Any, Dict
from src.state.state import GraphState


async def fallback_node(state: GraphState) -> Dict[str, Any]:
    """Return a deterministic fallback message when no documents pass or retry limits expire."""
    documents = state.get("documents", [])
    retry_count = state.get("generation_hallucination_retry_count", 0)

    # 1. No relevant documents retrieved above threshold
    if not documents:
        message = "I could not find any relevant documents in the knowledge base to answer your question."
        reason = "no_relevant_documents"
    # 2. Hallucination verification retry budget exhausted
    else:
        message = (
            "A verified answer could not be generated from the available context "
            "due to factual consistency constraints."
        )
        reason = "max_hallucination_retries_exceeded"

    return {
        "generation": message,
        "has_hallucination": False,  
        "audit_logs": [
            {
                "node": "fallback_node",
                "reason": reason,
                "retry_count": retry_count,
            }
        ],
    }