import logging
from typing import Any, Dict, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_ollama import ChatOllama
from config import settings
from src.chains.models import (
    GradeHallucinations,
)
from src.prompts.system_prompts import (
    GENERATOR_SYSTEM_PROMPT,
    HALLUCINATION_GRADER_SYSTEM_PROMPT,
    REWRITER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def get_base_llm(temperature: Optional[float] = None) -> ChatOllama:
    # 16k Baglam penceresi ve Apple Silicon icin optimize edilmis ChatOllama ornegi
    return ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        num_ctx=settings.LLM_NUM_CTX,
        client_kwargs={"timeout": settings.REQUEST_TIMEOUT},
    )


def get_base_slm(temperature: Optional[float] = None) -> ChatOllama:
    # Sorgu donusturme ve hafif islemler icin kucuk baglamli SLM istemcisi
    return ChatOllama(
        model=settings.SLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature if temperature is not None else settings.SLM_MIN_TEMPERATURE,
        num_ctx=settings.SLM_NUM_CTX,
        client_kwargs={"timeout": settings.REQUEST_TIMEOUT},
    )


def create_hallucination_grader_chain() -> RunnableSerializable[Dict[str, Any], GradeHallucinations]:
    # Uretilen yanitta halusinasyon olup olmadigini denetleyen zincir
    llm = get_base_llm()
    structured_llm = llm.with_structured_output(GradeHallucinations)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", HALLUCINATION_GRADER_SYSTEM_PROMPT),
            ("human", "Set of facts:\n\n{documents}\n\nLLM generation: {generation}"),
        ]
    )
    return prompt | structured_llm

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


def create_query_rewriter_chain(temperature: Optional[float] = None) -> RunnableSerializable[Dict[str, Any],str]:
    # Dinamik sicaklik ve yapilandirilmis cikti (Pydantic) ile sorgu donusturme zinciri
    slm = get_base_slm(temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITER_SYSTEM_PROMPT),
            ("human", "User question:\n\n{question}"),
        ]
    )
    return prompt | slm | StrOutputParser()