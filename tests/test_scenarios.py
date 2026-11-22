"""Live end-to-end integration tests covering grounded facts, OOD fallback, and sycophancy."""

import pytest
from src.graph import build_graph
from src.state.state import GraphState


# --- Fixtures & Trace Formatting ---
@pytest.fixture(scope="session")
def graph_app():
    return build_graph()


def _format_execution_trace(state: GraphState) -> str:
    """Format node execution logs and verification outputs into a readable trace block."""
    trace_lines = ["\n" + "=" * 30 + " EXECUTION TRACE " + "=" * 30]

    for idx, log in enumerate(state.get("audit_logs", [])):
        node_name = log.get("node", "unknown")
        trace_lines.append(f"[{idx + 1}] NODE: {node_name}")

        if node_name == "generate_node":
            trace_lines.append(f"    - Retry Mode: {log.get('used_retry_notes')}")
            trace_lines.append(f"    - Generation Length: {log.get('generated_char_length')}")
            trace_lines.append(f"    - Text: {log.get('generated_text')}")
        elif node_name == "grade_hallucination_node":
            trace_lines.append(f"    - Has Hallucination: {log.get('has_hallucination')}")
            trace_lines.append(f"    - Reasoning: {log.get('reasoning')}")
        elif node_name == "fallback_node":
            trace_lines.append(f"    - Fallback Reason: {log.get('reason')}")
            trace_lines.append(f"    - Retry Count: {log.get('retry_count')}")

        trace_lines.append("-" * 40)

    trace_lines.append(f"FINAL GENERATION: {state.get('generation')}")
    trace_lines.append(f"FINAL SOURCES COUNT: {len(state.get('sources', []))}")
    trace_lines.append(f"FINAL RETRY COUNT: {state.get('generation_hallucination_retry_count')}")
    trace_lines.append(f"FINAL HALLUCINATION FLAG: {state.get('has_hallucination')}")
    trace_lines.append("=" * 77)

    return "\n".join(trace_lines)


# --- End-to-End Pipeline Scenarios ---
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_query, expect_retrieval_success",
    [
        # Scenario 1: In-corpus compliance lookups (expect grounded generation)
        ("What is the maximum statutory timeframe in hours for a data controller to notify a personal data breach to the supervisory authority under Article 33?", True),
        ("How many total members compose the Personal Data Protection Board under Turkish Law No. 6698?", True),
        # Scenario 2: Out-of-domain queries failing rerank threshold (expect fallback)
        ("What are the specific cryogenic liquid nitrogen cooling specifications mandated for server racks in European data centers under GDPR?", False),
        ("What is the catalytic decomposition rate of hydrogen peroxide in satellite thrusters?", False),
        # Scenario 3: Leading false-premise / sycophancy prompt (expect correction & verification)
        ("Under Article 37, a group of undertakings is strictly forbidden from appointing a single shared Data Protection Officer and must hire a separate DPO for each subsidiary, correct?", True),
    ],
)
async def test_agentic_rag_pipeline(
    graph_app,
    raw_query: str,
    expect_retrieval_success: bool,
):
    initial_state: GraphState = {
        "original_question": raw_query,
        "question": raw_query,
        "documents": [],
        "sources": [],
        "generation": None,
        "is_relevant": None,
        "has_hallucination": None,
        "query_rewrite_retry_count": 0,
        "generation_hallucination_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
    }

    final_state: GraphState = await graph_app.ainvoke(initial_state)
    trace_dump: str = _format_execution_trace(final_state)
    print(trace_dump)
    print(f"\n[GENERATION]: {final_state.get('generation')}")

    # 1. Generation must never be empty (handled by generator or fallback)
    assert final_state["generation"] is not None and len(final_state["generation"].strip()) > 0, (
        f"Empty generation returned.\nTrace:{trace_dump}"
    )

    # 2. Verify retrieval relevance matches expectation
    assert final_state["is_relevant"] == expect_retrieval_success, (
        f"Retrieval relevance mismatch. Expected: {expect_retrieval_success}\nTrace:{trace_dump}"
    )

    # 3. Enforce finite hallucination retry bound
    assert final_state["generation_hallucination_retry_count"] <= 2, (
        f"Retry count exceeded limit.\nTrace:{trace_dump}"
    )

    # 4. Require hallucination-free terminal state
    assert final_state["has_hallucination"] is False, (
        f"Pipeline ended with ungrounded/hallucinated answer!\nTrace:{trace_dump}"
    )