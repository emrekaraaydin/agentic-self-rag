import logging
import os
from typing import Any, Dict, List, Optional
import chainlit as cl
from src.graph import build_graph
from src.state.state import GraphState

logging.getLogger("chainlit").setLevel(logging.ERROR)


def resolve_pdf_path(source_name: str) -> Optional[str]:
    # Dosya adi veya yolunun diskteki konumunu tespit eder
    potential_paths = [
        source_name,
        os.path.join("data", source_name),
        os.path.join(os.path.dirname(__file__), "data", source_name),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    return None


@cl.on_chat_start
async def on_chat_start() -> None:
    app_graph = build_graph()
    cl.user_session.set("graph", app_graph)
    cl.user_session.set("previous_elements", [])

    await cl.Message(
        content="Agentic RAG pipeline initialized. You may submit your query."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    graph = cl.user_session.get("graph")
    user_query: str = message.content

    # Onceki sorgudan kalan elementleri arayuzden ve yan panelden kaldir
    previous_elements: List[Any] = cl.user_session.get("previous_elements", [])
    for element in previous_elements:
        try:
            await element.remove()
        except Exception:
            pass
    cl.user_session.set("previous_elements", [])

    initial_state: GraphState = {
        "original_question": user_query,
        "question": user_query,
        "documents": [],
        "sources": [],
        "generation": None,
        "is_relevant": None,
        "has_hallucination": None,
        "retrieval_retry_count": 0,
        "generation_hallucination_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
    }

    final_generation: str = ""
    final_sources: List[Dict[str, Any]] = []

    # Graf calismasini adim adim stream etme
    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            if node_name == "transform_query_node":
                async with cl.Step(name="Query Transformation") as step:
                    transformed_query: str = node_output.get("question", user_query)
                    step.output = f"Optimized query: '{transformed_query}'"

            elif node_name == "retrieve_node":
                async with cl.Step(name="Vector Retrieval") as step:
                    retrieved_count: int = len(node_output.get("documents", []))
                    step.output = f"Retrieved {retrieved_count} candidate chunks from Qdrant."

            elif node_name == "grade_retrieval_node":
                async with cl.Step(name="Reranking & Filtering") as step:
                    passed_count: int = len(node_output.get("documents", []))
                    is_relevant: bool = node_output.get("is_relevant", False)
                    final_sources = node_output.get("sources", [])
                    step.output = (
                        f"Chunks passed rerank threshold: {passed_count} | "
                        f"Status: {'Relevant' if is_relevant else 'Insufficient'}"
                    )

            elif node_name == "generate_node":
                async with cl.Step(name="Response Generation") as step:
                    step.output = "Draft response generated from context."
                    if node_output.get("generation"):
                        final_generation = node_output["generation"]

            elif node_name == "grade_hallucination_node":
                async with cl.Step(name="Hallucination Verification") as step:
                    has_hallucination: bool = node_output.get("has_hallucination", False)
                    retry_count: int = node_output.get(
                        "generation_hallucination_retry_count", 0
                    )
                    reasoning: str = node_output.get("hallucination_reasoning") or ""

                    if has_hallucination:
                        step.output = (
                            f"Hallucination detected (Attempt {retry_count}). "
                            f"Routing back to generator.\nGrader verdict: {reasoning}"
                        )
                    else:
                        step.output = "Response verified: Grounded in context."

            elif node_name == "fallback_node":
                async with cl.Step(name="Fallback Triggered") as step:
                    final_generation = node_output.get("generation", "")
                    step.output = "Constraint violated or documents missing. Standard fallback applied."

    response_text = final_generation or "Unable to generate a valid response."
    source_elements: List[Any] = []

    # Kaynaklari yan panele baglama
    if final_sources:
        citation_tags: List[str] = []

        for idx, src in enumerate(final_sources):
            source_label = f"Source {idx + 1}"
            citation_tags.append(f"[{source_label}]")

            raw_source_name = src.get("source", "unknown")
            page_num = src.get("page")
            score = src.get("rerank_score", 0.0)
            chunk_id = src.get("chunk_id", f"doc_{idx}")

            pdf_path = resolve_pdf_path(raw_source_name)

            if pdf_path and pdf_path.endswith(".pdf"):
                source_elements.append(
                    cl.Pdf(
                        name=source_label,
                        path=pdf_path,
                        page=int(page_num) if page_num is not None else 1,
                        display="side",
                    )
                )
            else:
                card_content = (
                    f"**Document ID:** `{chunk_id}`\n\n"
                    f"**Source File:** {raw_source_name}\n\n"
                    f"**Page:** {page_num if page_num is not None else 'N/A'}\n\n"
                    f"**FlashRank Score:** {score:.4f}"
                )
                source_elements.append(
                    cl.Text(
                        name=source_label,
                        content=card_content,
                        display="side",
                    )
                )

        response_text += "\n\n**Sources:** " + " ".join(citation_tags)

    # Yaniti gonder ve olusturulan elementleri bir sonraki sorguda temizlemek uzere sakla
    await cl.Message(
        content=response_text,
        elements=source_elements,
    ).send()

    cl.user_session.set("previous_elements", source_elements)