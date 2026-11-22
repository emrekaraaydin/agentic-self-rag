"""Shared LangGraph state definition for the Agentic RAG workflow."""

from operator import add
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import NotRequired, ReadOnly, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """Typed state passed between query planning, retrieval, grading, and verification nodes."""

    # --- Query Inputs ---
    original_question: ReadOnly[str]
    question: str

    # --- Retrieval & Reranking Outputs ---
    is_relevant: Optional[bool]
    documents: List[Document]
    sources: List[Dict[str, Any]]

    # --- Generation & NLI Verification Outputs ---
    generation: Optional[str]
    has_hallucination: Optional[bool]
    hallucination_reasoning: Optional[str]

    # --- Retry Counters & Execution Trace ---
    generation_hallucination_retry_count: int
    query_rewrite_retry_count: int
    audit_logs: Annotated[List[Dict[str, Any]], add]

    # --- Multi-Part Planning & Reflective Re-Retrieval ---
    is_multi_part: NotRequired[bool]
    sub_queries: NotRequired[List[str]]
    sub_query_documents: NotRequired[Dict[str, List[Document]]]
    retained_documents: NotRequired[List[Document]]
    retrieval_reflection_retry_count: NotRequired[int]
    needs_additional_retrieval: NotRequired[bool]