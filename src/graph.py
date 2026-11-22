"""LangGraph StateGraph construction and edge wiring for Agentic RAG."""

import logging
from langgraph.graph import END, StateGraph
from src.nodes.decompose_query_node import decompose_query_node
from src.nodes.generate_node import generate_node
from src.nodes.grade_hallucination_node import grade_hallucination_node
from src.nodes.grade_retrieval_node import grade_retrieval_node
from src.nodes.reflect_missing_context_node import reflect_missing_context_node
from src.nodes.retrieve_node import retrieve_node
from src.nodes.transform_query_node import transform_query_node
from src.nodes.fallback_node import fallback_node
from src.edges.routing import (
    route_after_hallucination,
    route_after_retrieval,
)
from src.state.state import GraphState

logger = logging.getLogger(__name__)


def build_graph():
    """Construct and compile the bounded cyclic Agentic RAG StateGraph."""
    logger.info("Building Agentic RAG StateGraph...")

    workflow = StateGraph(GraphState)

    # 1. Register Graph Nodes
    workflow.add_node("decompose_query_node", decompose_query_node)
    workflow.add_node("transform_query_node", transform_query_node)
    workflow.add_node("reflect_missing_context_node", reflect_missing_context_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("grade_retrieval_node", grade_retrieval_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("grade_hallucination_node", grade_hallucination_node)
    workflow.add_node("fallback_node", fallback_node)

    # 2. Entry Point
    workflow.set_entry_point("decompose_query_node")

    # 3. Direct Edges
    workflow.add_edge("decompose_query_node", "retrieve_node")
    workflow.add_edge("transform_query_node", "retrieve_node")
    workflow.add_edge("reflect_missing_context_node", "retrieve_node")
    workflow.add_edge("retrieve_node", "grade_retrieval_node")
    workflow.add_edge("generate_node", "grade_hallucination_node")
    workflow.add_edge("fallback_node", END)

    # 4. Conditional Edges
    # Post-rerank sufficiency decision
    workflow.add_conditional_edges(
    "grade_retrieval_node",
    route_after_retrieval,
    {
        "generate_node": "generate_node",
        "transform_query_node": "transform_query_node",
        "fallback_node": "fallback_node",
    },
)

    # Post-NLI grounding, missing-context reflection, and retry cap decision
    workflow.add_conditional_edges(
    "grade_hallucination_node",
    route_after_hallucination,
    {
        "reflect_missing_context_node": "reflect_missing_context_node",
        "generate_node": "generate_node",
        "fallback_node": "fallback_node",
        "__end__": END,
    },
)

    # 5. Compile Graph
    app = workflow.compile()
    logger.info("Agentic RAG StateGraph successfully compiled.")

    return app