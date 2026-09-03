from operator import add
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import ReadOnly, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    original_question: ReadOnly[str]
    question: str
    is_relevant: Optional[bool]
    documents: List[Document]
    generation: Optional[str]
    has_hallucination: Optional[bool]
    retrieval_retry_count: int
    generation_hallucination_retry_count: int
    hallucination_reasoning: Optional[str]
    audit_logs: Annotated[List[Dict[str, Any]], add]