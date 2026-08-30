from typing import Literal,Optional
from pydantic import BaseModel, Field

class GradeHallucinations(BaseModel):
    reasoning: str = Field(
        ...,
        description="Concise step-by-step reasoning evaluating factual grounding against context.",
    )
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="'yes' if all claims are strictly grounded in context, 'no' if there is any hallucinated or unsupported claim.",
    )

class GradeAnswer(BaseModel):
    reasoning: str = Field(
        ...,
        description="Step-by-step evaluation of the answer against the user's question, identifying resolved aspects and gaps.",
    )
    is_satisfactory: bool = Field(
        ...,
        description="True if the answer directly and adequately resolves the question, False if essential information is missing or unresolved.",
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Specific critique or missing points to guide query reformulation if the answer is unsatisfactory.",
    )