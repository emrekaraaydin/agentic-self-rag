"""Offline unit tests for conditional graph routing, clean-refusal heuristics, and NLI grading."""

import asyncio
import pytest
from langchain_core.documents import Document

from src.edges.routing import route_after_hallucination, route_after_retrieval
from src.nodes import grade_hallucination_node as gh_mod
from src.nodes.grade_hallucination_node import (
    _detect_clean_refusal,
    _evaluate_with_nli,
    grade_hallucination_node,
)
from src.state.state import GraphState


# --- Retrieval Routing Tests ---
def test_route_after_retrieval_success():
    """Verify routing to generate_node when retrieval meets relevance threshold."""
    mock_state: GraphState = {
        "original_question": "test",
        "question": "test",
        "is_relevant": True,
        "documents": [],
        "sources": [],
        "generation": None,
        "has_hallucination": None,
        "query_rewrite_retry_count": 1,
        "generation_hallucination_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
    }
    assert route_after_retrieval(mock_state) == "generate_node"


def test_route_after_retrieval_failure():
    """Verify routing to fallback_node when retrieval fails after rewrite retry budget."""
    mock_state: GraphState = {
        "original_question": "test",
        "question": "test",
        "is_relevant": False,
        "documents": [],
        "sources": [],
        "generation": None,
        "has_hallucination": None,
        "query_rewrite_retry_count": 1,
        "generation_hallucination_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
    }
    assert route_after_retrieval(mock_state) == "fallback_node"


# --- Hallucination & Reflection Routing Tests ---
@pytest.mark.parametrize(
    "has_hallucination, retry_count, expected_route",
    [
        (False, 0, "__end__"),
        (True, 0, "generate_node"),
        (True, 1, "generate_node"),
        (True, 2, "fallback_node"),
    ],
)
def test_route_after_hallucination_matrix(
    has_hallucination: bool, retry_count: int, expected_route: str
):
    """Verify hallucination retry progression up to fallback cap."""
    mock_state: GraphState = {
        "original_question": "test",
        "question": "test",
        "is_relevant": True,
        "documents": [],
        "sources": [],
        "generation": "sample response",
        "has_hallucination": has_hallucination,
        "query_rewrite_retry_count": 1,
        "generation_hallucination_retry_count": retry_count,
        "hallucination_reasoning": "detected" if has_hallucination else None,
        "audit_logs": [],
    }
    assert route_after_hallucination(mock_state) == expected_route


def test_route_after_hallucination_triggers_reflection():
    mock_state: GraphState = {
        "original_question": "Compare GDPR and KVKK",
        "question": "Compare GDPR and KVKK",
        "is_relevant": True,
        "documents": [],
        "sources": [],
        "generation": "The provided documents do not contain information about KVKK.",
        "has_hallucination": False,
        "needs_additional_retrieval": True,
        "retrieval_reflection_retry_count": 0,
        "query_rewrite_retry_count": 0,
        "generation_hallucination_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
    }
    assert route_after_hallucination(mock_state) == "reflect_missing_context_node"


def test_route_after_hallucination_skips_reflection_after_max_retry():
    mock_state: GraphState = {
        "original_question": "Compare GDPR and KVKK",
        "question": "Compare GDPR and KVKK",
        "is_relevant": True,
        "documents": [],
        "sources": [],
        "generation": "The provided documents do not contain information about KVKK.",
        "has_hallucination": False,
        "needs_additional_retrieval": True,
        "retrieval_reflection_retry_count": 1,
        "query_rewrite_retry_count": 0,
        "generation_hallucination_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
    }
    assert route_after_hallucination(mock_state) == "__end__"


# --- Clean Refusal & NLI Verification Tests ---
def test_clean_refusal_detection():
    assert _detect_clean_refusal("The context does not provide information about this topic.")
    assert _detect_clean_refusal(
        "The provided context does not contain Article 99(6) or any specific details. "
        "Therefore, I cannot confirm or deny the statement based on the given information."
    )
    assert not _detect_clean_refusal(
        "The context does not mention X, however, based on the provided documents, the ceiling is 20M EUR."
    )
    assert not _detect_clean_refusal(
        "Under GDPR Article 33, the controller must notify the supervisory authority within 72 hours."
    )


def test_nli_evaluation_verdicts_and_refusal_skip(monkeypatch):
    docs = [Document(page_content="Controllers must notify breaches within 72 hours.", metadata={"doc_title": "GDPR"})]

    # 1. Grounded verdict (high entailment, low contradiction)
    monkeypatch.setattr(gh_mod, "_score_nli_windows", lambda passages, hyp: (0.91, 0.07, 0.02, 1, 120))
    res_ok, score_ok, meta_ok = _evaluate_with_nli(docs, "Breaches must be reported within 72 hours.")
    assert res_ok.verdict == "grounded"
    assert not res_ok.is_refusal
    assert score_ok == pytest.approx(0.91)
    assert meta_ok["window_count"] == 1

    # 2. High contradiction -> hallucinated even if entailment not near zero
    monkeypatch.setattr(gh_mod, "_score_nli_windows", lambda passages, hyp: (0.45, 0.15, 0.40, 1, 120))
    res_contra, _, _ = _evaluate_with_nli(docs, "Breaches must be reported within 24 hours.")
    assert res_contra.verdict == "hallucinated"

    # 3. Low entailment -> hallucinated
    monkeypatch.setattr(gh_mod, "_score_nli_windows", lambda passages, hyp: (0.20, 0.75, 0.05, 1, 120))
    res_low_e, _, _ = _evaluate_with_nli(docs, "Unverified external rule applies.")
    assert res_low_e.verdict == "hallucinated"

    # 4. Clean refusal skips model inference completely
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("_score_nli_windows should not run on clean refusal")

    monkeypatch.setattr(gh_mod, "_score_nli_windows", _fail_if_called)
    res_ref, score_ref, _ = _evaluate_with_nli(
        docs, "The provided context does not contain specific details, so I cannot confirm or deny."
    )
    assert res_ref.verdict == "grounded"
    assert res_ref.is_refusal
    assert score_ref == pytest.approx(1.0)


def test_grade_hallucination_node_async_flow(monkeypatch):
    docs = [Document(page_content="GDPR Art 33: 72 hours.", metadata={"source": "gdpr.pdf"})]
    base_state: GraphState = {
        "original_question": "Notification deadline?",
        "question": "Notification deadline?",
        "is_relevant": True,
        "documents": docs,
        "sources": [],
        "generation": "Notification is within 72 hours.",
        "has_hallucination": None,
        "query_rewrite_retry_count": 0,
        "generation_hallucination_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
        "is_multi_part": False,
        "sub_queries": [],
        "retrieval_reflection_retry_count": 0,
    }

    # Pass case
    monkeypatch.setattr(gh_mod, "_score_nli_windows", lambda passages, hyp: (0.88, 0.10, 0.02, 1, 80))
    out_ok = asyncio.run(grade_hallucination_node(base_state))
    assert out_ok["has_hallucination"] is False
    assert out_ok["generation_hallucination_retry_count"] == 0
    assert out_ok["hallucination_reasoning"] is None
    assert out_ok["audit_logs"][0]["nli_entailment"] == pytest.approx(0.88)

    # Hallucination increment + reasoning
    monkeypatch.setattr(gh_mod, "_score_nli_windows", lambda passages, hyp: (0.05, 0.05, 0.90, 1, 80))
    out_bad = asyncio.run(grade_hallucination_node({**base_state, "generation": "Notification is within 24 hours."}))
    assert out_bad["has_hallucination"] is True
    assert out_bad["generation_hallucination_retry_count"] == 1
    assert "ModernBERT-large-nli" in out_bad["hallucination_reasoning"]

    # Empty generation
    out_empty = asyncio.run(grade_hallucination_node({**base_state, "generation": "   "}))
    assert out_empty["has_hallucination"] is True
    assert out_empty["generation_hallucination_retry_count"] == 1

    # Multi-part missing context -> reflective retrieval
    multi_state = {
        **base_state,
        "is_multi_part": True,
        "sub_queries": ["Part 1 GDPR", "Part 2 KVKK"],
        "generation": "GDPR specifies 72 hours, however the context does not contain KVKK details.",
    }
    out_refl = asyncio.run(grade_hallucination_node(multi_state))
    assert out_refl["needs_additional_retrieval"] is True
    assert out_refl["has_hallucination"] is False