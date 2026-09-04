from typing import Any, Dict
from src.state.state import GraphState


async def fallback_node(state: GraphState) -> Dict[str, Any]:
    # Belgelerin yetersizligi veya halusinasyon limiti asiminda deterministik yanit atar
    documents = state.get("documents", [])
    retry_count = state.get("generation_hallucination_retry_count", 0)

    # 1. Retrieval hic dokuman bulamadiysa
    if not documents:
        message = "I could not find any relevant documents in the knowledge base to answer your question."
        reason = "no_relevant_documents"
    # 2. Retry limiti asildiysa
    else:
        message = (
            "A verified answer could not be generated from the available context "
            "due to factual consistency constraints."
        )
        reason = "max_hallucination_retries_exceeded"

    return {
        "generation": message,
        "has_hallucination": False,  # Deterministik ret mesaji artik halusinasyon sayilmaz
        "audit_logs": [
            {
                "node": "fallback_node",
                "reason": reason,
                "retry_count": retry_count,
            }
        ],
    }