from typing import Literal,Optional
from pydantic import BaseModel, Field

class GradeHallucinations(BaseModel):
    reasoning: str = Field(
        ...,
        description="Check if the response makes factual claims about the real world that are absent from the context. Note: Statements explicitly stating that the context lacks information, or that information is unknown, are NOT hallucinations.",
    )
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="'yes' ONLY if the response invents real-world facts, numbers, or features not supported by context. 'no' if the response is fully grounded OR if the response simply states that information was not found / missing in context.")
    