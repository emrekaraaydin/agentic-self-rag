"""Pydantic structured-output schemas for verification verdicts, query plans, and reflection."""

from typing import List, Literal
from pydantic import BaseModel, Field


class GradeHallucinations(BaseModel):
    """Structured factual consistency and refusal verdict emitted by the hallucination verifier."""

    is_refusal: bool = Field(
        ...,
        description=(
            "True if the response states that the context lacks the information, "
            "cannot answer, or contains no mention of the topic. False if it makes positive factual claims."
        ),
    )
    reasoning: str = Field(
        ...,
        description=(
            "If is_refusal is True, output 'Clean refusal - no unsupported facts'. "
            "If is_refusal is False, verify whether every claim exists in the context."
        ),
    )
    verdict: Literal["grounded", "hallucinated"] = Field(
        ...,
        description=(
            "'grounded': If is_refusal is True OR if all factual claims are explicitly found in the context. "
            "'hallucinated': If the response invents facts, names, or events not present in the context."
        ),
    )


class QueryPlan(BaseModel):
    """Structured multi-part decomposition plan produced by the query planner."""

    is_multi_part: bool = Field(
        ...,
        description=(
            "True if the question compares two or more distinct statutes, regulations, or separate articles/sections "
            "requiring independent retrieval; False for a single focused question."
        ),
    )
    sub_queries: List[str] = Field(
        ...,
        description=(
            "1 to 3 concise, self-contained English search queries. Return 1 query if is_multi_part is False; "
            "return 2 to 3 standalone queries if is_multi_part is True, each targeting one specific statute/article/topic."
        ),
    )


class MissingContextPlan(BaseModel):
    """Structured follow-up search queries targeting missing statutes or sections in a draft answer."""

    missing_sub_queries: List[str] = Field(
        ...,
        description=(
            "1 to 2 targeted standalone English search queries specifically aimed at retrieving the missing "
            "statute, article, or facts reported as absent in the draft answer (empty list if nothing more to search)."
        ),
    )