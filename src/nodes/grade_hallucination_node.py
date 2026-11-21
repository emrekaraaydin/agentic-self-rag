import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import (
    NLI_CONTRA_THRESHOLD,
    NLI_ENTAIL_THRESHOLD,
    NLI_MAX_LENGTH,
    NLI_MODEL_NAME,
    RETRIEVAL_REFLECTION_MAX_RETRY,
)
from src.chains.models import GradeHallucinations
from src.state.state import GraphState

logger = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_NLI_TOKENIZER: Any = None
_NLI_MODEL: Any = None
_NLI_DEVICE: str = "cpu"

_REFUSAL_PATTERNS: Tuple[str, ...] = (
    "the context does not provide information about this topic",
    "could not find any relevant documents",
    "verified answer could not be generated",
    "does not contain information",
    "do not contain information",
    "does not contain specific",
    "do not contain specific",
    "does not contain a specific",
    "do not contain a specific",
    "does not provide information",
    "do not provide information",
    "not mentioned in the provided",
    "not directly mentioned in the provided",
    "cannot confirm or deny",
)

_POSITIVE_FACT_INDICATORS: Tuple[str, ...] = (
    "in fact,",
    "according to document",
    "according to the provided",
    "according to the documents",
    "according to article",
    "however, based on the provided",
    "instead, the context",
)

_MISSING_CONTEXT_PATTERNS: Tuple[str, ...] = (
    "does not contain",
    "do not contain",
    "does not provide",
    "do not provide",
    "not mentioned in the provided",
    "not directly mentioned",
    "no information about",
    "does not specify",
    "cannot confirm or deny",
    "knowledge base lacks",
)


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_nli_model() -> Tuple[Any, Any, str]:
    """Thread-safe lazy singleton loader for ModernBERT-large-nli."""
    global _NLI_TOKENIZER, _NLI_MODEL, _NLI_DEVICE

    if _NLI_MODEL is not None and _NLI_TOKENIZER is not None:
        return _NLI_TOKENIZER, _NLI_MODEL, _NLI_DEVICE

    with _MODEL_LOCK:
        if _NLI_MODEL is not None and _NLI_TOKENIZER is not None:
            return _NLI_TOKENIZER, _NLI_MODEL, _NLI_DEVICE

        try:
            tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME, local_files_only=True)
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)

        device = _select_device()
        try:
            model.to(device)
            model.eval()
        except Exception as exc:
            logger.warning("Failed to move %s to %s (%s); falling back to CPU.", NLI_MODEL_NAME, device, exc)
            device = "cpu"
            model.to("cpu")
            model.eval()

        _NLI_TOKENIZER = tokenizer
        _NLI_MODEL = model
        _NLI_DEVICE = device
        logger.info("Loaded NLI hallucination model %s on device=%s.", NLI_MODEL_NAME, _NLI_DEVICE)
        return _NLI_TOKENIZER, _NLI_MODEL, _NLI_DEVICE


def _detect_clean_refusal(generation: str) -> bool:
    """Yanıtın pozitif bir iddia içermeyen temiz bir ret (clean refusal) olup olmadığını denetler."""
    low = (generation or "").strip().lower()
    if not any(pat in low for pat in _REFUSAL_PATTERNS):
        return False
    if any(ind in low for ind in _POSITIVE_FACT_INDICATORS):
        return False
    return True


def _detect_missing_context_claim(generation: str) -> bool:
    """Üretilen yanıtta bağlamda bir bilginin veya mevzuatın eksik kaldığına dair ifade olup olmadığını saptar."""
    low = (generation or "").strip().lower()
    return any(pat in low for pat in _MISSING_CONTEXT_PATTERNS)


def _format_documents(documents: List[Document]) -> List[str]:
    """Belgeleri isteğe bağlı doc_title üst bilgisiyle Passage bloklarına dönüştürür."""
    formatted: List[str] = []
    for doc in documents:
        content = (doc.page_content or "").strip()
        if not content:
            continue
        doc_title = str((doc.metadata or {}).get("doc_title") or "").strip()
        block = f"[Document: {doc_title}]\n{content}" if doc_title else content
        formatted.append(block)
    return [f"Passage [{idx + 1}]: {block}" for idx, block in enumerate(formatted)]


def _split_oversized_passage(
    tokenizer: Any,
    text: str,
    max_tokens: int,
    overlap_tokens: int = 96,
) -> List[str]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return [text]
    stride = max(64, max_tokens - overlap_tokens)
    pieces: List[str] = []
    for start in range(0, len(ids), stride):
        piece_ids = ids[start : start + max_tokens]
        pieces.append(tokenizer.decode(piece_ids, skip_special_tokens=True))
        if start + max_tokens >= len(ids):
            break
    return pieces


def _build_nli_windows(
    passages: List[str],
    hypothesis: str,
    tokenizer: Any,
    max_length: int = NLI_MAX_LENGTH,
    overlap_docs: int = 1,
) -> Tuple[List[str], int]:
    """8K pencereye sığan tek blok veya taşma halinde çakışmalı NLI pencereleri üretir."""
    full_premise = "\n\n".join(passages)
    premise_tokens = len(tokenizer(full_premise, add_special_tokens=False)["input_ids"])
    hyp_tokens = len(tokenizer(hypothesis, add_special_tokens=False)["input_ids"])
    total_tokens = premise_tokens + hyp_tokens + 4

    if total_tokens <= max_length:
        return [full_premise], total_tokens

    passage_budget = max(128, max_length - hyp_tokens - 16)
    normalized: List[str] = []
    for p in passages:
        normalized.extend(
            _split_oversized_passage(tokenizer, p, max_tokens=max(128, passage_budget - 32))
        )

    piece_lens = [len(tokenizer(piece, add_special_tokens=False)["input_ids"]) + 2 for piece in normalized]
    windows: List[str] = []
    n = len(normalized)
    idx = 0

    while idx < n:
        win_pieces: List[str] = []
        used = 0
        j = idx
        while j < n:
            cost = piece_lens[j]
            if win_pieces and used + cost > passage_budget:
                break
            win_pieces.append(normalized[j])
            used += cost
            j += 1
        windows.append("\n\n".join(win_pieces))
        if j >= n:
            break
        step = max(1, (j - idx) - overlap_docs)
        idx += step

    return windows, total_tokens


def _score_nli_windows(
    passages: List[str],
    hypothesis: str,
) -> Tuple[float, float, float, int, int]:
    """ModernBERT-large-nli ile (entailment, neutral, contradiction, window_count, total_tokens) döner."""
    global _NLI_DEVICE
    tokenizer, model, device = _get_nli_model()
    windows, total_tokens = _build_nli_windows(passages, hypothesis, tokenizer, max_length=NLI_MAX_LENGTH)

    triplets: List[Tuple[float, float, float]] = []
    for win_premise in windows:
        encoded = tokenizer(
            win_premise,
            hypothesis,
            truncation="only_first",
            max_length=NLI_MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
        try:
            with torch.inference_mode():
                logits = model(**inputs).logits
        except Exception as exc:
            if _NLI_DEVICE != "cpu":
                logger.warning("NLI inference on %s failed (%s); falling back to CPU.", _NLI_DEVICE, exc)
                _NLI_DEVICE = "cpu"
                device = "cpu"
                model.to("cpu")
                model.eval()
                inputs = {k: v.to("cpu") for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
                with torch.inference_mode():
                    logits = model(**inputs).logits
            else:
                raise

        squeezed = logits[0] if logits.dim() == 2 else logits.squeeze(0)
        probs = torch.softmax(squeezed, dim=-1).detach().cpu().tolist()
        triplets.append((float(probs[0]), float(probs[1]), float(probs[2])))

    p_entail = max(t[0] for t in triplets)
    p_contra = min(t[2] for t in triplets)
    p_neutral = max(0.0, 1.0 - p_entail - p_contra)
    return p_entail, p_neutral, p_contra, len(windows), total_tokens


def _evaluate_with_nli(
    documents: List[Document],
    generation: str,
    question: str = "",
) -> Tuple[GradeHallucinations, float, Dict[str, Any]]:
    """ModernBERT-large-nli ile tam bağlam / 8K pencere olgusal tutarlılık denetimi yapar."""
    del question  # NLI premise/hypothesis arayüzü ile gerekmez; uyumluluk imzası korunur
    is_refusal = _detect_clean_refusal(generation)
    passages = _format_documents(documents)

    if not passages:
        return (
            GradeHallucinations(
                is_refusal=is_refusal,
                reasoning="No context documents available for NLI verification.",
                verdict="grounded" if is_refusal else "hallucinated",
            ),
            0.0,
            {
                "nli_entailment": 0.0,
                "nli_neutral": 0.0,
                "nli_contradiction": 0.0,
                "window_count": 0,
                "total_tokens": 0,
            },
        )

    if is_refusal:
        return (
            GradeHallucinations(
                is_refusal=True,
                reasoning="Clean refusal - no unsupported factual claims.",
                verdict="grounded",
            ),
            1.0,
            {
                "nli_entailment": 1.0,
                "nli_neutral": 0.0,
                "nli_contradiction": 0.0,
                "window_count": 0,
                "total_tokens": 0,
            },
        )

    p_entail, p_neutral, p_contra, window_count, total_tokens = _score_nli_windows(
        passages,
        generation.strip(),
    )
    is_grounded = (p_contra < NLI_CONTRA_THRESHOLD) and (p_entail >= NLI_ENTAIL_THRESHOLD)
    verdict = "grounded" if is_grounded else "hallucinated"
    reasoning = (
        f"ModernBERT-large-nli entailment={p_entail:.4f}, neutral={p_neutral:.4f}, "
        f"contradiction={p_contra:.4f} (windows={window_count}, tokens={total_tokens}, "
        f"thresholds contra<{NLI_CONTRA_THRESHOLD:.2f}, entail>={NLI_ENTAIL_THRESHOLD:.2f})."
    )
    meta = {
        "nli_entailment": round(p_entail, 4),
        "nli_neutral": round(p_neutral, 4),
        "nli_contradiction": round(p_contra, 4),
        "window_count": window_count,
        "total_tokens": total_tokens,
    }

    return (
        GradeHallucinations(
            is_refusal=False,
            reasoning=reasoning,
            verdict=verdict,
        ),
        p_entail,
        meta,
    )


# Mevcut karşılaştırma scriptleri ile geriye dönük uyumluluk takma adı
_evaluate_with_hhem = _evaluate_with_nli


async def grade_hallucination_node(state: GraphState) -> Dict[str, Any]:
    """Üretilen yanıtın bağlam havuzuna sadık kalıp kalmadığını ModernBERT-large-nli ile denetler."""
    logger.info("Executing async ModernBERT-large-nli hallucination verification...")

    generation: str = state.get("generation") or ""
    documents: List[Document] = state.get("documents", [])
    question: str = state.get("question") or ""
    current_retry_count: int = state.get("generation_hallucination_retry_count", 0)
    reflection_retry_count: int = state.get("retrieval_reflection_retry_count", 0)
    is_multi_part: bool = bool(
        state.get("is_multi_part", False) or len(state.get("sub_queries") or []) > 1
    )

    if not generation.strip():
        logger.warning("Empty generation received in hallucination grader.")
        return {
            "has_hallucination": True,
            "needs_additional_retrieval": False,
            "generation_hallucination_retry_count": current_retry_count + 1,
            "hallucination_reasoning": "Empty generation.",
            "audit_logs": [
                {
                    "node": "grade_hallucination_node",
                    "has_hallucination": True,
                    "needs_additional_retrieval": False,
                    "nli_score": 0.0,
                    "hhem_score": 0.0,
                    "contra_threshold": NLI_CONTRA_THRESHOLD,
                    "entail_threshold": NLI_ENTAIL_THRESHOLD,
                    "reasoning": "Empty generation.",
                    "success": False,
                }
            ],
        }

    # Eksik mevzuat / kısmi ret durumunda çoklu-parçalı soruda yeniden arama (Self-Reflective Re-Retrieval)
    if (
        _detect_missing_context_claim(generation)
        and is_multi_part
        and len(documents) > 0
        and reflection_retry_count < RETRIEVAL_REFLECTION_MAX_RETRY
    ):
        logger.info(
            "Missing context claim detected in multi-part answer (reflection_retry=%d/%d). Triggering additional retrieval.",
            reflection_retry_count,
            RETRIEVAL_REFLECTION_MAX_RETRY,
        )
        return {
            "has_hallucination": False,
            "needs_additional_retrieval": True,
            "audit_logs": [
                {
                    "node": "grade_hallucination_node",
                    "has_hallucination": False,
                    "needs_additional_retrieval": True,
                    "reasoning": "Missing context claim detected in multi-part response; routing to reflective re-retrieval.",
                    "success": True,
                }
            ],
        }

    result, nli_score, meta = await asyncio.to_thread(
        _evaluate_with_nli, documents, generation, question
    )

    has_hallucination: bool = result.verdict == "hallucinated" and not result.is_refusal

    logger.info(
        "ModernBERT-large-nli check completed | Entail: %.4f | Contra: %.4f | Windows: %d | Is Refusal: %s | Has Hallucination: %s",
        meta["nli_entailment"],
        meta["nli_contradiction"],
        meta["window_count"],
        result.is_refusal,
        has_hallucination,
    )

    log_entry: Dict[str, Any] = {
        "node": "grade_hallucination_node",
        "has_hallucination": has_hallucination,
        "needs_additional_retrieval": False,
        "nli_score": round(nli_score, 4),
        "nli_entailment": meta["nli_entailment"],
        "nli_neutral": meta["nli_neutral"],
        "nli_contradiction": meta["nli_contradiction"],
        "hhem_score": round(nli_score, 4),
        "contra_threshold": NLI_CONTRA_THRESHOLD,
        "entail_threshold": NLI_ENTAIL_THRESHOLD,
        "window_count": meta["window_count"],
        "is_refusal": result.is_refusal,
        "reasoning": result.reasoning,
        "success": True,
    }

    return {
        "has_hallucination": has_hallucination,
        "needs_additional_retrieval": False,
        "generation_hallucination_retry_count": current_retry_count + 1 if has_hallucination else current_retry_count,
        "hallucination_reasoning": result.reasoning if has_hallucination else None,
        "audit_logs": [log_entry],
    }
