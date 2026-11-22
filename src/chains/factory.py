"""Ollama chat model clients and LCEL chain factories for planning, rewriting, synthesis, and reflection."""

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


# --- Base Model Clients ---
def get_base_llm() -> ChatOllama:
    """Return the primary ChatOllama LLM client configured with the 16K context window."""
    return ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        num_ctx=settings.LLM_NUM_CTX,
        client_kwargs={"timeout": settings.REQUEST_TIMEOUT},
    )


def get_base_slm() -> ChatOllama:
    """Return the lightweight ChatOllama SLM client used for query rewriting."""
    return ChatOllama(
        model=settings.SLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.SLM_MIN_TEMPERATURE,
        num_ctx=settings.SLM_NUM_CTX,
        client_kwargs={"timeout": settings.REQUEST_TIMEOUT},
    )


# --- LCEL Runnable Chains ---
# Note: Hallucination verification runs locally via ModernBERT-large-nli in src/nodes/grade_hallucination_node.py.
def create_generator_chain() -> RunnableSerializable[Dict[str, Any], str]:
    """Create the context-grounded response generation chain."""
    llm = get_base_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATOR_SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\n{retry_notes}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def create_query_rewriter_chain() -> RunnableSerializable[Dict[str, Any],str]:
    """Create the SLM-based query simplification and rewriting chain."""
    slm = get_base_slm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITER_SYSTEM_PROMPT),
            ("human", "User question:\n\n{question}"),
        ]
    )
    return prompt | slm | StrOutputParser()


def create_query_decomposer_chain() -> RunnableSerializable[Dict[str, Any], QueryPlan]:
    """Create the structured query planner chain for comparative or multi-statute questions."""
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
    """Create the structured reflection chain generating follow-up queries for missing context."""
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