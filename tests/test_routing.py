import pytest
from src.edges.routing import route_after_hallucination, route_after_retrieval
from src.state.state import GraphState


def test_route_after_retrieval_success():
    # Dokuman esigi gectiginde uretime yonlendirme kontrolu
    mock_state: GraphState = {
        "original_question": "test",
        "question": "test",
        "is_relevant": True,
        "documents": [],
        "generation": None,
        "has_hallucination": None,
        "retrieval_retry_count": 1,
        "generation_hallucination_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
    }
    assert route_after_retrieval(mock_state) == "generate_node"


def test_route_after_retrieval_failure():
    # Dokuman bulunamadiginda sonlanma kontrolu
    mock_state: GraphState = {
        "original_question": "test",
        "question": "test",
        "is_relevant": False,
        "documents": [],
        "generation": None,
        "has_hallucination": None,
        "retrieval_retry_count": 1,
        "generation_hallucination_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
    }
    assert route_after_retrieval(mock_state) == "__end__"


@pytest.mark.parametrize(
    "has_hallucination, retry_count, expected_route",
    [
        (False, 0, "__end__"),
        (True, 0, "generate_node"),
        (True, 1, "generate_node"),
        (True, 2, "__end__"),
    ],
)
def test_route_after_hallucination_matrix(has_hallucination, retry_count, expected_route):
    # Halusinasyon ve tekrar deneme sayaci matrisi kontrolu
    mock_state: GraphState = {
        "original_question": "test",
        "question": "test",
        "is_relevant": True,
        "documents": [],
        "generation": "sample response",
        "has_hallucination": has_hallucination,
        "retrieval_retry_count": 1,
        "generation_hallucination_retry_count": retry_count,
        "hallucination_reasoning": "detected" if has_hallucination else None,
        "audit_logs": [],
    }
    assert route_after_hallucination(mock_state) == expected_route