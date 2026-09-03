import logging
from langgraph.graph import END, StateGraph
from src.nodes.generate_node import generate_node
from src.nodes.grade_hallucination_node import grade_hallucination_node
from src.nodes.grade_retrieval_node import grade_retrieval_node
from src.nodes.retrieve_node import retrieve_node
from src.nodes.transform_query_node import transform_query_node
from src.edges.routing import (
    route_after_hallucination,
    route_after_retrieval,
)
from src.state.state import GraphState

logger = logging.getLogger(__name__)


def build_graph():
    # Agentic RAG StateGraph olusturma ve derleme adimi
    logger.info("Building Agentic RAG StateGraph...")

    workflow = StateGraph(GraphState)

    # 1. Dugumlerin (Nodes) Eklenmesi
    workflow.add_node("transform_query_node", transform_query_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("grade_retrieval_node", grade_retrieval_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("grade_hallucination_node", grade_hallucination_node)

    # 2. Giris Noktasi (Entry Point)
    workflow.set_entry_point("transform_query_node")

    # 3. Deterministik / Dogrusal Gecisler (Direct Edges)
    workflow.add_edge("transform_query_node", "retrieve_node")
    workflow.add_edge("retrieve_node", "grade_retrieval_node")
    workflow.add_edge("generate_node", "grade_hallucination_node")

    # 4. Kosullu Yonlendirmeler (Conditional Edges)

    # Reranker sonrasi dokuman yeterliligi kontrolu
    workflow.add_conditional_edges(
        "grade_retrieval_node",
        route_after_retrieval,
        {
            "generate_node": "generate_node",
            "__end__": END,
        },
    )

    # Halusinasyon denetimi kontrolu
    workflow.add_conditional_edges(
        "grade_hallucination_node",
        route_after_hallucination,
        {
            "generate_node": "generate_node",
            "__end__": END,
        },
    )

    # 5. Grafigin Derlenmesi
    app = workflow.compile()
    logger.info("Agentic RAG StateGraph successfully compiled.")

    return app