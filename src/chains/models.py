from typing import Literal,Optional
from pydantic import BaseModel, Field

# class GradeHallucinations(BaseModel):
#     reasoning: str = Field(
#         ...,
#         description="Check if the response makes factual claims about the real world that are absent from the context. Note: Statements explicitly stating that the context lacks information, or that information is unknown, are NOT hallucinations.",
#     )
#     binary_score: Literal["yes", "no"] = Field(
#         ...,
#         description="'yes' ONLY if the response invents real-world facts, numbers, or features not supported by context. 'no' if the response is fully grounded OR if the response simply states that information was not found / missing in context.")
class GradeHallucinations(BaseModel):
    reasoning: str = Field(
        ...,
        description="Brief explanation of whether the claims in the response are grounded in the context.",
    )
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description=(
            "'yes' if the response makes factual claims not found in context. "
            "'no' if all facts are supported by context OR if the response states information is missing."
        ),
    )