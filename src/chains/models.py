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
# class GradeHallucinations(BaseModel):
#     binary_score: Literal["yes", "no"] = Field(
#         ...,
#         description=(
#             "'yes' if the response makes factual claims not found in context. "
#             "'no' if all facts are supported by context OR if the response states information is missing."
#         ),
#     )
#     reasoning: str = Field(
#         ...,
#         description="Brief explanation of whether the claims in the response are grounded in the context.",
#     )
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