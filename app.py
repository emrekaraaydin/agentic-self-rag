import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import chainlit as cl
from config.settings import RERANK_DEFAULT_THRESHOLD
from src.graph import build_graph
from src.state.state import GraphState
from src.ui.chunk_visualizer import (
    ChunkStyle,
    build_highlighted_chunk_pdf,
    build_sources_markdown_section,
    cleanup_highlighted_pdfs,
    compute_chunk_style,
    format_chunk_detail_card,
    format_chunk_pdf_label,
    format_chunk_text_label,
)

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
    cl.user_session.set("previous_temp_pdfs", [])

    await cl.Message(
        content="Agentic RAG pipeline initialized. You may submit your query."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    graph = cl.user_session.get("graph")
    user_query: str = message.content

    # Onceki sorgudan kalan elementleri ve gecici highlight edilmis PDF dosyalarini temizle
    previous_elements: List[Any] = cl.user_session.get("previous_elements", [])
    for element in previous_elements:
        try:
            await element.remove()
        except Exception:
            pass
    cl.user_session.set("previous_elements", [])

    previous_temp_pdfs: List[str] = cl.user_session.get("previous_temp_pdfs", [])
    if previous_temp_pdfs:
        await asyncio.to_thread(cleanup_highlighted_pdfs, previous_temp_pdfs)
    cl.user_session.set("previous_temp_pdfs", [])

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
                    graded_docs = node_output.get("documents", [])
                    passed_count: int = len(graded_docs)
                    is_relevant: bool = node_output.get("is_relevant", False)
                    raw_sources: List[Dict[str, Any]] = node_output.get("sources", [])

                    # Eksik metin veya metadata alanlarini dokumanlardan tamamla
                    enriched_sources: List[Dict[str, Any]] = []
                    for idx, src_item in enumerate(raw_sources):
                        merged_src = dict(src_item)
                        if idx < len(graded_docs):
                            doc_obj = graded_docs[idx]
                            if not merged_src.get("content"):
                                merged_src["content"] = getattr(doc_obj, "page_content", "")
                            doc_meta = getattr(doc_obj, "metadata", {}) or {}
                            merged_src.setdefault("chunk_index", doc_meta.get("chunk_index", idx))
                            merged_src.setdefault("retrieval_score", doc_meta.get("retrieval_score"))
                        merged_src.setdefault("threshold", RERANK_DEFAULT_THRESHOLD)
                        enriched_sources.append(merged_src)

                    final_sources = enriched_sources
                    step.output = (
                        f"Chunks passed rerank threshold (%{RERANK_DEFAULT_THRESHOLD * 100:.1f}): {passed_count} | "
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
    temp_pdf_paths: List[str] = []

    # Kaynak chunklari reranker skoruna gore dinamik renklendirerek yan panele baglama
    if final_sources:
        batch_scores: List[float] = [
            float(s.get("rerank_score") or 0.0) for s in final_sources
        ]
        active_threshold: float = float(
            final_sources[0].get("threshold") or RERANK_DEFAULT_THRESHOLD
        )

        # Her chunk icin dinamik stil ve diskteki PDF yolunu hazirla
        styled_items: List[Tuple[int, Dict[str, Any], ChunkStyle, Optional[str]]] = []
        pdf_groups: Dict[str, List[Tuple[int, Dict[str, Any], ChunkStyle]]] = {}

        for idx, src in enumerate(final_sources):
            rank_idx = idx + 1
            score = float(src.get("rerank_score") or 0.0)
            style = compute_chunk_style(
                score=score,
                threshold=active_threshold,
                batch_scores=batch_scores,
            )
            raw_source_name = src.get("source", "unknown")
            pdf_path = resolve_pdf_path(raw_source_name)
            styled_items.append((rank_idx, src, style, pdf_path))

            if pdf_path and pdf_path.endswith(".pdf"):
                pdf_groups.setdefault(pdf_path, []).append((rank_idx, src, style))

        source_rows: List[Dict[str, Any]] = []

        for rank_idx, src, style, pdf_path in styled_items:
            raw_source_name = src.get("source", "unknown")
            reported_page = src.get("page")

            if pdf_path and pdf_path.endswith(".pdf"):
                same_pdf_list = pdf_groups.get(pdf_path, [])
                highlighted_pdf_path, resolved_page = await asyncio.to_thread(
                    build_highlighted_chunk_pdf,
                    pdf_path,
                    rank_idx,
                    src,
                    style,
                    same_pdf_list,
                )
                if highlighted_pdf_path != pdf_path:
                    temp_pdf_paths.append(highlighted_pdf_path)

                src_for_card = dict(src)
                src_for_card["page"] = resolved_page

                pdf_label = format_chunk_pdf_label(rank_idx, style, resolved_page)
                text_label = format_chunk_text_label(rank_idx)
                card_content = format_chunk_detail_card(rank_idx, src_for_card, style)

                source_elements.append(
                    cl.Pdf(
                        name=pdf_label,
                        path=highlighted_pdf_path,
                        page=resolved_page,
                        display="side",
                    )
                )
                source_elements.append(
                    cl.Text(
                        name=text_label,
                        content=card_content,
                        display="side",
                    )
                )
                source_rows.append(
                    {
                        "pdf_label": pdf_label,
                        "text_label": text_label,
                        "style": style,
                        "source_name": raw_source_name,
                    }
                )
            else:
                fallback_page = int(reported_page) if reported_page is not None else None
                primary_label = format_chunk_pdf_label(rank_idx, style, fallback_page)
                card_content = format_chunk_detail_card(rank_idx, src, style)

                source_elements.append(
                    cl.Text(
                        name=primary_label,
                        content=card_content,
                        display="side",
                    )
                )
                source_rows.append(
                    {
                        "pdf_label": primary_label,
                        "text_label": None,
                        "style": style,
                        "source_name": raw_source_name,
                    }
                )

        response_text += build_sources_markdown_section(
            source_rows=source_rows,
            threshold=active_threshold,
        )

    # Yaniti gonder ve olusturulan elementleri bir sonraki sorguda temizlemek uzere sakla
    await cl.Message(
        content=response_text,
        elements=source_elements,
    ).send()

    cl.user_session.set("previous_elements", source_elements)
    cl.user_session.set("previous_temp_pdfs", temp_pdf_paths)
