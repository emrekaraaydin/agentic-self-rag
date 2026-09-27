from typing import Literal
from pydantic import BaseModel, Field
class GradeHallucinations(BaseModel):
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