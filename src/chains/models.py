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
    
class GradeAnswer(BaseModel):
    reasoning: str = Field(
        ...,
        description="Brief verification checking ONLY two failure conditions: (1) Does the response state that information is missing/unknown/unfound? (2) Is the response about a completely different entity or topic than asked?",
    )
    is_satisfactory: bool = Field(
        ...,
        description="Default to True. Set to False ONLY if the response states information is unavailable or is completely off-topic. Set to True for any response that provides relevant facts, technical details, or partial information about the requested topic.",
    )
    feedback: str = Field(
        ...,
        description="If unsatisfactory, provide 1 concise sentence explaining what specific topic needs retrieval. If satisfactory, write 'N/A'.",
    )
#class GenerationResult(BaseModel):
    # 1. Model önce cevabı üretir (veya bilginin olmadığını açıklar)
   #  answer: str = Field(
    #     ...,
    #     description="The detailed answer based on context, or a statement explaining that the context lacks the required information.",
    # )

    # 2. Ürettiği cevaba bakarak kararı en son verir
   #  has_sufficient_context: bool = Field(
    #     ...,
   #     description="Set to False if you stated above that context lacks info. Set to True if you provided the actual answer from context.",
    # 