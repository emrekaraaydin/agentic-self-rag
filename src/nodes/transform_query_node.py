"""SLM-based query rewriting node with post-processing for repetitive parenthetical expansions."""

import logging
import re
from typing import Any, Dict, List
from config import settings
from src.chains.factory import create_query_rewriter_chain
from src.state.state import GraphState

logger = logging.getLogger(__name__)


# --- Rewritten Query Cleanup ---
def _clean_rewritten_query(raw_query: str) -> str:
    """Remove nested parentheses, phrase repetitions, and redundant terms from SLM outputs."""
    text = re.sub(r"\s+", " ", raw_query).strip()
    if "(" not in text and ")" not in text:
        return text

    first_paren_idx = text.find("(")
    if first_paren_idx == -1:
        return re.sub(r"[()]+", "", text).strip()

    main_part = text[:first_paren_idx].strip()
    paren_tail = text[first_paren_idx:]

    has_qmark = main_part.endswith("?") or paren_tail.rstrip().endswith("?")
    main_part = main_part.rstrip("?").strip()

    # Flatten nested parentheses inside the trailing expansion into comma-separated candidates
    paren_flat = re.sub(r"[()?]+", ",", paren_tail)
    raw_phrases = [p.strip() for p in paren_flat.split(",") if p.strip()]

    main_lower = main_part.lower()
    main_words = set(re.findall(r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]+", main_lower))

    cleaned_phrases: List[str] = []
    seen_lower: set[str] = set()

    for phrase in raw_phrases:
        # Collapse consecutive word/n-gram repetitions within a phrase (e.g., "A B A B" -> "A B")
        deduped = phrase
        for _ in range(3):
            next_deduped = re.sub(
                r"\b([0-9a-zA-ZçğıöşüÇĞİÖŞÜ]+(?:\s+[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]+){0,7}?)\s+\1\b",
                r"\1",
                deduped,
                flags=re.IGNORECASE,
            )
            if next_deduped == deduped:
                break
            deduped = next_deduped

        deduped = re.sub(r"\s+", " ", deduped).strip()
        low = deduped.lower()
        if not low or low in seen_lower:
            continue

        # Discard candidate phrases already verbatim or fully covered in the main question
        phrase_words = re.findall(r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]+", low)
        if not phrase_words:
            continue
        if low in main_lower:
            continue
        if all(w in main_words for w in phrase_words):
            continue

        seen_lower.add(low)
        cleaned_phrases.append(deduped)
        if len(cleaned_phrases) >= 4:
            break

    suffix = "?" if has_qmark else ""
    if cleaned_phrases:
        return f"{main_part} ({', '.join(cleaned_phrases)}){suffix}"
    return f"{main_part}{suffix}"


# --- Graph Node ---
async def transform_query_node(state: GraphState) -> Dict[str, Any]:
    """Rewrite a failing query with the SLM to recover after zero rerank hits."""
    logger.info("Executing async query transformation...")
    question: str = state.get("question", "")
    transformed_query: str = question
    retry_count: int = state.get("query_rewrite_retry_count", 0) + 1
    try:
        rewriter_chain = create_query_rewriter_chain()
        result = await rewriter_chain.ainvoke({"question": transformed_query})

        if result and result.strip():
            transformed_query = _clean_rewritten_query(result.strip())
            logger.info(
                "Query transformed from '%s' to '%s' (temperature: %.2f)",
                question,
                transformed_query,
                settings.SLM_MIN_TEMPERATURE,
            )
    except Exception as exc:
        logger.error("Query transformation failed: %s", exc, exc_info=True)

    log_entry: Dict[str, Any] = {
        "node": "transform_query_node",
        "original_query": question,
        "transformed_query": transformed_query,
        "temperature": settings.SLM_MIN_TEMPERATURE,
        "success": transformed_query != question,
    }

    return {
        "question": transformed_query,
        "sub_queries": [transformed_query],
        "query_rewrite_retry_count": retry_count,
        "audit_logs": [log_entry],
    }