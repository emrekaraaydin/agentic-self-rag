from operator import add
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import NotRequired, ReadOnly, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    original_question: ReadOnly[str]
    question: str
    is_relevant: Optional[bool]
    documents: List[Document]
    generation: Optional[str]
    has_hallucination: Optional[bool]
    generation_hallucination_retry_count: int
    query_rewrite_retry_count: int
    hallucination_reasoning: Optional[str]
    audit_logs: Annotated[List[Dict[str, Any]], add]
    sources: List[Dict[str, Any]]
    is_multi_part: NotRequired[bool]
    sub_queries: NotRequired[List[str]]
    sub_query_documents: NotRequired[Dict[str, List[Document]]]
    retained_documents: NotRequired[List[Document]]
    retrieval_reflection_retry_count: NotRequired[int]
    needs_additional_retrieval: NotRequired[bool]