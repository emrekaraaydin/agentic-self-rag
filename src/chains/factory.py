import logging
from typing import Any, Dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import ChatOllama
from config import settings
from src.chains.models import MissingContextPlan, QueryPlan
from src.prompts.system_prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    REWRITER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def get_base_llm() -> ChatOllama:
    # 16k Baglam penceresi ve Apple Silicon icin optimize edilmis ChatOllama ornegi
    return ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        num_ctx=settings.LLM_NUM_CTX,
        client_kwargs={"timeout": settings.REQUEST_TIMEOUT},
    )


def get_base_slm() -> ChatOllama:
    # Sorgu donusturme ve hafif islemler icin kucuk baglamli SLM istemcisi
    return ChatOllama(
        model=settings.SLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.SLM_MIN_TEMPERATURE,
        num_ctx=settings.SLM_NUM_CTX,
        client_kwargs={"timeout": settings.REQUEST_TIMEOUT},
    )


# def create_hallucination_grader_chain() -> RunnableSerializable[Dict[str, Any], GradeHallucinations]:
#     # Uretilen yanitta halusinasyon olup olmadigini denetleyen zincir (LLM tabanli - Yorum satirina alindi, Vectara HHEM-2.1 kullaniliyor)
#     llm = get_base_llm()
#     structured_llm = llm.with_structured_output(GradeHallucinations)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", HALLUCINATION_GRADER_SYSTEM_PROMPT),
#             ("human", "Set of facts:\n\n{documents}\n\nLLM generation: {generation}"),
#         ]
#     )
#     return prompt | structured_llm

def create_generator_chain() -> RunnableSerializable[Dict[str, Any], str]:
    # Alinan baglama dayanarak yanit ureten zincir
    llm = get_base_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATOR_SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\n{retry_notes}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def create_query_rewriter_chain() -> RunnableSerializable[Dict[str, Any],str]:
    # SLM ile sorgu sadelestirme ve donusturme zinciri
    slm = get_base_slm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITER_SYSTEM_PROMPT),
            ("human", "User question:\n\n{question}"),
        ]
    )
    return prompt | slm | StrOutputParser()


def create_query_decomposer_chain() -> RunnableSerializable[Dict[str, Any], QueryPlan]:
    """Coklu mevzuat veya karsilastirmali sorulari alt sorgulara ayiran planlayici zincir."""
    llm = get_base_llm()
    structured_llm = llm.with_structured_output(QueryPlan)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DECOMPOSER_SYSTEM_PROMPT),
            ("human", "User question:\n\n{question}"),
        ]
    )
    return prompt | structured_llm


def create_missing_context_reflector_chain() -> RunnableSerializable[Dict[str, Any], MissingContextPlan]:
    """Taslak yanitta eksik kalan mevzuat/madde icin hedefli yeniden arama sorgusu ureten zincir."""
    llm = get_base_llm()
    structured_llm = llm.with_structured_output(MissingContextPlan)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REFLECTION_SYSTEM_PROMPT),
            (
                "human",
                "Original question:\n{question}\n\n"
                "Previous search queries:\n{previous_queries}\n\n"
                "Draft answer:\n{draft_answer}",
            ),
        ]
    )
    return prompt | structured_llm