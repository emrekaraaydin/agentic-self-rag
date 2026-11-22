"""Chainlit UI entrypoint with page-deduplicated PDF citations and chunk inspection cards."""

import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
import chainlit as cl
from src.graph import build_graph
from src.state.state import GraphState

logging.getLogger("chainlit").setLevel(logging.ERROR)

# --- Constants & Bilingual Matching Stopwords ---
_STOPWORDS: Set[str] = {
    # English stopwords
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "how", "can", "could", "would", "should", "may",
    "might", "must", "shall", "will", "into", "onto", "upon", "about", "above",
    "below", "under", "over", "between", "among", "through", "during", "before",
    "after", "only", "also", "such", "than", "then", "their", "there", "these",
    "those", "they", "them", "its", "been", "being", "does", "did", "done",
    "other", "more", "most", "some", "any", "all", "each", "both", "either",
    "neither", "because", "unless", "until", "while", "wherever", "whether",
    "provided", "context", "document", "documents", "information", "knowledge",
    "base", "source", "according", "article", "section", "paragraph",
    # Turkish stopwords (kept for bilingual tokenization coverage)
    "bir", "veya", "ile", "icin", "için", "gibi", "kadar", "daha", "olan",
    "olarak", "bu", "su", "şu", "o", "ve", "da", "de", "mi", "mu", "mı", "mü",
    "ne", "neden", "nasil", "nasıl", "kim", "hangi", "ama", "fakat", "ancak",
    "gore", "göre", "uzere", "üzere", "dolayi", "dolayı", "ilgili", "hakkindaki",
    "hakkında", "belge", "belgeler", "kaynak", "paragraf", "madde",
}

_REFUSAL_MARKERS: Tuple[str, ...] = (
    "i could not find any relevant documents",
    "a verified answer could not be generated",
    "the context does not provide information about this topic",
    "unable to generate a valid response",
)


# --- Passage Matching & Bold Highlighting Helpers ---
def _tokenize_for_matching(text: str) -> Set[str]:
    """Extract content terms stripped of stopwords for paragraph-to-chunk matching."""
    tokens = re.findall(r"[0-9a-zA-ZçğıöşüÇĞİÖŞÜ]+", text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in _STOPWORDS}


def _is_pure_refusal_or_fallback(response_text: str) -> bool:
    """Return True when the response is a refusal or fallback containing no positive factual claims."""
    low = response_text.strip().lower()
    if any(marker in low for marker in _REFUSAL_MARKERS):
        return True
    has_no_info = (
        "does not contain" in low
        or "do not contain" in low
        or "does not provide" in low
        or "do not provide" in low
    )
    has_positive_facts = (
        "in fact," in low
        or "according to" in low
        or "however," in low
        or "whereas" in low
    )
    return has_no_info and not has_positive_facts


def _is_skippable_heading_or_metadata(block: str) -> bool:
    """Skip markdown headings, bracketed breadcrumb headers, and tiny non-prose blocks."""
    stripped = block.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    if len(_tokenize_for_matching(stripped)) < 3:
        return True
    return False


_ABBREV_ENDINGS: Tuple[str, ...] = (
    "art.",
    "no.",
    "sec.",
    "reg.",
    "para.",
    "par.",
    "e.g.",
    "i.e.",
    "vs.",
    "p.",
    "pp.",
)


def _split_sentences_preserve_ws(line: str) -> List[str]:
    """Split a long line at sentence boundaries while preserving list numbers, citations, and exact whitespace."""
    parts: List[str] = []
    last_idx = 0
    for match in re.finditer(r"[.!?](\s+)(?=[A-Z(\[])", line):
        before = line[: match.start() + 1]
        # Do not split after list numbers like '2. ' or legal abbreviations like 'No. '
        if re.search(r"(?:^|\s)\d{1,3}\.$", before):
            continue
        if before.lower().endswith(_ABBREV_ENDINGS):
            continue

        ws_start, ws_end = match.span(1)
        parts.append(line[last_idx:ws_start])
        parts.append(line[ws_start:ws_end])
        last_idx = ws_end

    parts.append(line[last_idx:])
    return parts


def _wrap_segment_bold(segment: str) -> str:
    """Wrap segment text in balanced Markdown bold, preserving leading bullet/spaces."""
    if not segment.strip():
        return segment

    lead_match = re.match(r"^(\s*(?:[-*]\s+)?)", segment)
    prefix = lead_match.group(1) if lead_match else ""
    rest = segment[len(prefix):]

    trail_match = re.search(r"(\s*)$", rest)
    trail = trail_match.group(1) if trail_match else ""
    core = rest[: len(rest) - len(trail)] if trail else rest

    cleaned_inner = core.replace("**", "").strip()
    if not cleaned_inner:
        return segment
    return f"{prefix}**{cleaned_inner}**{trail}"


def _find_matching_segments(
    chunk_content: str,
    matched_para_texts: List[str],
) -> Tuple[List[Dict[str, Any]], Set[Tuple[int, int, int]]]:
    """Identify chunk prose segments overlapping with matched answer paragraph texts."""
    if not matched_para_texts or not chunk_content.strip():
        return [], set()

    answer_tokens: Set[str] = set()
    for para_text in matched_para_texts:
        answer_tokens.update(_tokenize_for_matching(para_text))

    if not answer_tokens:
        return [], set()

    blocks = re.split(r"(\n\s*\n)", chunk_content)
    parsed_blocks: List[Dict[str, Any]] = []
    candidates: List[Tuple[int, int, int, int, float]] = []

    for b_idx, block in enumerate(blocks):
        if b_idx % 2 == 1 or _is_skippable_heading_or_metadata(block):
            parsed_blocks.append({"raw": block, "lines": None})
            continue

        raw_lines = block.split("\n")
        split_lines: List[List[str]] = []
        for l_idx, line in enumerate(raw_lines):
            if len(line) > 220:
                # Preserve exact whitespace separators at odd indices
                segs = _split_sentences_preserve_ws(line)
            else:
                segs = [line]
            split_lines.append(segs)

            for s_idx, seg in enumerate(segs):
                if s_idx % 2 == 1 or not seg.strip():
                    continue
                seg_tokens = _tokenize_for_matching(seg)
                if len(seg_tokens) < 3:
                    continue
                overlap = len(seg_tokens & answer_tokens)
                if overlap > 0:
                    coverage = overlap / max(len(seg_tokens), 1)
                    candidates.append((b_idx, l_idx, s_idx, overlap, coverage))

        parsed_blocks.append({"raw": block, "lines": split_lines})

    best_overlap = max((c[3] for c in candidates), default=0)
    if best_overlap < 2:
        return parsed_blocks, set()

    highlight_keys: Set[Tuple[int, int, int]] = set()
    for b_idx, l_idx, s_idx, overlap, coverage in candidates:
        if (overlap >= 2 and overlap >= 0.55 * best_overlap) or (
            overlap >= 3 and coverage >= 0.45
        ):
            highlight_keys.add((b_idx, l_idx, s_idx))

    return parsed_blocks, highlight_keys


def highlight_matching_chunk_passages(
    chunk_content: str,
    matched_para_texts: List[str],
) -> str:
    """Bold matching passages in matched chunks; leave General Context chunks plain."""
    parsed_blocks, highlight_keys = _find_matching_segments(
        chunk_content, matched_para_texts
    )
    if not highlight_keys:
        return chunk_content

    rebuilt_blocks: List[str] = []
    for b_idx, item in enumerate(parsed_blocks):
        split_lines = item["lines"]
        if split_lines is None:
            rebuilt_blocks.append(item["raw"])
            continue

        rebuilt_lines: List[str] = []
        for l_idx, segs in enumerate(split_lines):
            rebuilt_segs = [
                _wrap_segment_bold(seg) if (b_idx, l_idx, s_idx) in highlight_keys else seg
                for s_idx, seg in enumerate(segs)
            ]
            rebuilt_lines.append("".join(rebuilt_segs))
        rebuilt_blocks.append("\n".join(rebuilt_lines))

    return "".join(rebuilt_blocks)


# --- Citation Deduplication & PDF Resolution ---
def _normalize_page_number(page_val: Any) -> Optional[int]:
    """Coerce numeric page metadata values into an integer page index."""
    if page_val is None or isinstance(page_val, bool):
        return None
    if isinstance(page_val, (int, float)):
        return int(page_val)
    text = str(page_val).strip()
    if text.isdigit():
        return int(text)
    return None


def build_pdf_and_chunk_citation_plan(
    response_text: str,
    sources: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match paragraphs to chunks, deduplicate PDF citations per page, and prepare chunk cards."""
    if not response_text.strip() or not sources:
        return response_text, [], []

    if _is_pure_refusal_or_fallback(response_text):
        return response_text, [], []

    num_sources = len(sources)
    chunk_tokens: Dict[int, Set[str]] = {
        idx: _tokenize_for_matching(str(src.get("content") or ""))
        for idx, src in enumerate(sources, start=1)
    }

    raw_parts = re.split(r"(\n\s*\n)", response_text.strip())
    paragraph_texts: Dict[int, str] = {}
    chunk_to_paragraphs: Dict[int, List[int]] = {}
    part_plans: List[Tuple[bool, Optional[int], str, List[int], bool]] = []
    logical_para_num = 0

    for part in raw_parts:
        if not part.strip():
            part_plans.append((False, None, part, [], False))
            continue

        logical_para_num += 1
        paragraph_texts[logical_para_num] = part

        existing_tags = [
            int(m)
            for m in re.findall(r"\[(\d+)\]", part)
            if 1 <= int(m) <= num_sources
        ]
        if existing_tags:
            unique_tags = list(dict.fromkeys(existing_tags))
            for k in unique_tags:
                chunk_to_paragraphs.setdefault(k, []).append(logical_para_num)
            part_plans.append((True, logical_para_num, part, unique_tags, True))
            continue

        para_tokens = _tokenize_for_matching(part)
        if not para_tokens:
            part_plans.append((True, logical_para_num, part, [], False))
            continue

        scored_chunks: List[Tuple[int, int, float]] = []
        for idx in range(1, num_sources + 1):
            c_tokens = chunk_tokens.get(idx, set())
            if not c_tokens:
                continue
            overlap_count = len(para_tokens & c_tokens)
            if overlap_count > 0:
                jaccard_like = overlap_count / max(len(para_tokens), 1)
                scored_chunks.append((overlap_count, -idx, jaccard_like))

        if not scored_chunks:
            part_plans.append((True, logical_para_num, part, [], False))
            continue

        scored_chunks.sort(reverse=True)
        best_overlap, neg_best_idx, _ = scored_chunks[0]
        best_idx = -neg_best_idx

        matched_chunk_indices: List[int] = [best_idx]
        if len(scored_chunks) > 1:
            second_overlap, neg_second_idx, _ = scored_chunks[1]
            second_idx = -neg_second_idx
            if second_overlap >= 2 and second_overlap >= 0.6 * best_overlap:
                matched_chunk_indices.append(second_idx)

        matched_chunk_indices.sort()
        for k in matched_chunk_indices:
            chunk_to_paragraphs.setdefault(k, []).append(logical_para_num)

        part_plans.append((True, logical_para_num, part, matched_chunk_indices, False))

    # Deduplicate sources by (source, page) so each PDF page appears once in primary citations
    grouped_pdfs: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for idx, src in enumerate(sources, start=1):
        raw_source = str(src.get("source") or "unknown")
        norm_page = _normalize_page_number(src.get("page"))
        page_val = norm_page if norm_page is not None else src.get("page")
        score = float(src.get("rerank_score") or 0.0)
        key = (raw_source, page_val)

        entry = grouped_pdfs.get(key)
        if entry is None:
            entry = {
                "source": raw_source,
                "page": page_val,
                "matched_paras": [],
                "max_score": score,
                "chunk_nums": [],
            }
            grouped_pdfs[key] = entry

        entry["chunk_nums"].append(idx)
        if score > entry["max_score"]:
            entry["max_score"] = score
        entry["matched_paras"].extend(chunk_to_paragraphs.get(idx, []))

    pdf_entries = list(grouped_pdfs.values())
    for entry in pdf_entries:
        entry["matched_paras"] = sorted(set(entry["matched_paras"]))

    # Matched PDF pages first (ordered by paragraph number, then score), then General Context
    pdf_entries.sort(
        key=lambda e: (
            0 if e["matched_paras"] else 1,
            e["matched_paras"][0] if e["matched_paras"] else 0,
            -e["max_score"],
            e["source"],
            e["page"] if isinstance(e["page"], int) else 0,
        )
    )

    chunk_to_pdf_num: Dict[int, int] = {}
    for pdf_num, entry in enumerate(pdf_entries, start=1):
        entry["pdf_num"] = pdf_num
        for orig_chunk_num in entry["chunk_nums"]:
            chunk_to_pdf_num[orig_chunk_num] = pdf_num

    # Annotate paragraphs using deduplicated PDF citation numbers [1], [2], ...
    annotated_parts: List[str] = []
    for is_para, _, part, matched_chunks, had_existing in part_plans:
        if not is_para or not matched_chunks:
            annotated_parts.append(part)
            continue

        if had_existing:
            def _remap_tag(match: re.Match[str]) -> str:
                num = int(match.group(1))
                if 1 <= num <= num_sources:
                    return f"[{chunk_to_pdf_num.get(num, num)}]"
                return match.group(0)

            annotated_parts.append(re.sub(r"\[(\d+)\]", _remap_tag, part))
            continue

        pdf_nums = sorted(dict.fromkeys(chunk_to_pdf_num[k] for k in matched_chunks))
        citation_suffix = " ".join(f"[{p}]" for p in pdf_nums)
        annotated_parts.append(f"{part.rstrip()} {citation_suffix}")

    # Build chunk entries (matched chunks first with bold passage highlights, then General Context)
    chunk_entries: List[Dict[str, Any]] = []
    for idx, src in enumerate(sources, start=1):
        raw_source = str(src.get("source") or "unknown")
        norm_page = _normalize_page_number(src.get("page"))
        page_val = norm_page if norm_page is not None else src.get("page")
        score = float(src.get("rerank_score") or 0.0)
        chunk_id = str(src.get("chunk_id") or f"doc_{idx - 1}")
        raw_content = str(src.get("content") or "").strip()

        matched_paras = sorted(set(chunk_to_paragraphs.get(idx, [])))
        matched_para_texts = [
            paragraph_texts[p] for p in matched_paras if p in paragraph_texts
        ]
        highlighted_content = (
            highlight_matching_chunk_passages(raw_content, matched_para_texts)
            if matched_paras
            else raw_content
        )

        chunk_entries.append(
            {
                "orig_idx": idx,
                "pdf_num": chunk_to_pdf_num.get(idx, idx),
                "source": raw_source,
                "page": page_val,
                "score": score,
                "chunk_id": chunk_id,
                "matched_paras": matched_paras,
                "matched_para_texts": matched_para_texts,
                "raw_content": raw_content,
                "highlighted_content": highlighted_content,
            }
        )

    chunk_entries.sort(
        key=lambda c: (
            0 if c["matched_paras"] else 1,
            c["matched_paras"][0] if c["matched_paras"] else 0,
            -c["score"],
            c["orig_idx"],
        )
    )
    for display_idx, entry in enumerate(chunk_entries, start=1):
        entry["display_idx"] = display_idx

    return "".join(annotated_parts), pdf_entries, chunk_entries


def resolve_pdf_path(source_name: str) -> Optional[str]:
    """Locate the PDF file on disk relative to the workspace."""
    potential_paths = [
        source_name,
        os.path.join("data", source_name),
        os.path.join(os.path.dirname(__file__), "data", source_name),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    return None


def _format_match_label(matched_paras: List[int]) -> str:
    """Format matched paragraph badges or fall back to 'General Context'."""
    if matched_paras:
        return ", ".join(f"**Paragraph {p}**" for p in matched_paras)
    return "General Context"


# --- Chainlit Session & Event Callbacks ---
@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the compiled graph and session state containers for a new chat."""
    app_graph = build_graph()
    cl.user_session.set("graph", app_graph)
    cl.user_session.set("previous_elements", [])
    cl.user_session.set("chunk_details_store", {})
    cl.user_session.set("shown_chunk_keys", set())

    await cl.Message(
        content="Agentic RAG pipeline initialized. You may submit your query."
    ).send()


@cl.action_callback("show_chunk_details")
async def on_show_chunk_details(action: cl.Action) -> None:
    """Render side chunk inspection cards once per message and retire the action button."""
    store: Dict[str, List[Dict[str, Any]]] = (
        cl.user_session.get("chunk_details_store") or {}
    )
    key = (action.payload or {}).get("key")
    chunk_entries = store.get(key) if key else None

    if not chunk_entries:
        await cl.Message(
            content="Chunk details are no longer available for this message."
        ).send()
        return

    shown_keys: Set[str] = cl.user_session.get("shown_chunk_keys") or set()
    if key in shown_keys:
        return
    shown_keys.add(key)
    cl.user_session.set("shown_chunk_keys", shown_keys)

    chunk_elements: List[Any] = []
    summary_lines: List[str] = []

    for c in chunk_entries:
        display_idx = c["display_idx"]
        chunk_tag = f"Chunk [{display_idx}]"
        raw_source = c["source"]
        page_num = c["page"]
        page_str = f"p.{page_num}" if page_num is not None else "p.N/A"
        score = float(c["score"])
        chunk_id = c["chunk_id"]
        match_label = _format_match_label(c["matched_paras"])
        body = c["highlighted_content"] or "_Content not available._"

        card_content = (
            f"### {chunk_tag} — {raw_source}\n\n"
            f"**Page:** {page_num if page_num is not None else 'N/A'} | "
            f"**FlashRank Score:** {score:.4f}\n\n"
            f"**Chunk ID:** `{chunk_id}`\n\n"
            f"**Matched Paragraphs:** {match_label}\n\n"
            f"---\n\n"
            f"#### Chunk Content:\n\n"
            f"{body}"
        )
        chunk_elements.append(
            cl.Text(
                name=chunk_tag,
                content=card_content,
                display="side",
            )
        )
        summary_lines.append(
            f"- {chunk_tag} `{raw_source}` ({page_str}) | Score: `{score:.4f}` — {match_label}"
        )

    await cl.Message(
        content=f"**Retrieved Chunks ({len(chunk_entries)}):**\n" + "\n".join(summary_lines),
        elements=chunk_elements,
    ).send()

    prev_elems: List[Any] = cl.user_session.get("previous_elements") or []
    prev_elems.extend(chunk_elements)
    cl.user_session.set("previous_elements", prev_elems)

    try:
        await action.remove()
    except Exception:
        pass


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Stream Agentic RAG steps, deduplicate PDF citations, and attach side PDF viewers."""
    graph = cl.user_session.get("graph")
    user_query: str = message.content

    # Remove side elements attached by the previous turn
    previous_elements: List[Any] = cl.user_session.get("previous_elements", [])
    for element in previous_elements:
        try:
            await element.remove()
        except Exception:
            pass
    cl.user_session.set("previous_elements", [])

    initial_state: GraphState = {
        "original_question": user_query,
        "question": user_query,
        "documents": [],
        "sources": [],
        "generation": None,
        "is_relevant": None,
        "has_hallucination": None,
        "query_rewrite_retry_count": 0,
        "generation_hallucination_retry_count": 0,
        "retrieval_reflection_retry_count": 0,
        "hallucination_reasoning": None,
        "audit_logs": [],
    }

    final_generation: str = ""
    final_sources: List[Dict[str, Any]] = []
    final_documents: List[Any] = []

    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            if node_name == "decompose_query_node":
                async with cl.Step(name="Query Planning") as step:
                    is_multi = node_output.get("is_multi_part", False)
                    sub_queries = node_output.get("sub_queries") or [user_query]
                    if is_multi:
                        step.output = f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}"
                    else:
                        step.output = "Single-part query plan."

            elif node_name == "transform_query_node":
                async with cl.Step(name="Query Transformation") as step:
                    transformed_query: str = node_output.get("question", user_query)
                    step.output = f"Optimized query: '{transformed_query}'"

            elif node_name == "reflect_missing_context_node":
                async with cl.Step(name="Reflective Re-Retrieval") as step:
                    sub_queries = node_output.get("sub_queries") or []
                    step.output = f"Targeting missing context with sub-queries: {sub_queries}"

            elif node_name == "retrieve_node":
                async with cl.Step(name="Vector Retrieval") as step:
                    retrieved_count: int = len(node_output.get("documents", []))
                    step.output = f"Retrieved {retrieved_count} candidate chunks from Qdrant."

            elif node_name == "grade_retrieval_node":
                async with cl.Step(name="Reranking & Filtering") as step:
                    final_documents = node_output.get("documents", [])
                    passed_count: int = len(final_documents)
                    is_relevant: bool = node_output.get("is_relevant", False)
                    final_sources = node_output.get("sources", [])
                    for idx, src in enumerate(final_sources):
                        if not src.get("content") and idx < len(final_documents):
                            src["content"] = getattr(final_documents[idx], "page_content", "")
                    step.output = (
                        f"Chunks passed rerank threshold: {passed_count} | "
                        f"Status: {'Relevant' if is_relevant else 'Insufficient'}"
                    )

            elif node_name == "generate_node":
                async with cl.Step(name="Response Generation") as step:
                    step.output = "Draft response generated from context."
                    if node_output.get("generation"):
                        final_generation = node_output["generation"]

            elif node_name == "grade_hallucination_node":
                async with cl.Step(name="Hallucination Verification") as step:
                    has_hallucination: bool = node_output.get("has_hallucination", False)
                    needs_reflection: bool = node_output.get("needs_additional_retrieval", False)
                    retry_count: int = node_output.get(
                        "generation_hallucination_retry_count", 0
                    )
                    reasoning: str = node_output.get("hallucination_reasoning") or ""

                    if needs_reflection:
                        step.output = "Missing context detected in multi-part answer. Routing to reflective retrieval."
                    elif has_hallucination:
                        step.output = (
                            f"Hallucination detected (Attempt {retry_count}). "
                            f"Routing back to generator.\nGrader verdict: {reasoning}"
                        )
                    else:
                        step.output = "Response verified: Grounded in context."

            elif node_name == "fallback_node":
                async with cl.Step(name="Fallback Triggered") as step:
                    final_generation = node_output.get("generation", "")
                    step.output = "Constraint violated or documents missing. Standard fallback applied."

    response_text = final_generation or "Unable to generate a valid response."
    pdf_elements: List[Any] = []
    actions: List[cl.Action] = []

    if final_sources and not _is_pure_refusal_or_fallback(response_text):
        response_text, pdf_entries, chunk_entries = build_pdf_and_chunk_citation_plan(
            response_text, final_sources
        )
        citation_lines: List[str] = []

        for entry in pdf_entries:
            pdf_num = entry["pdf_num"]
            pdf_tag = f"[{pdf_num}]"
            raw_source = entry["source"]
            page_num = entry["page"]
            page_str = f"p.{page_num}" if page_num is not None else "p.N/A"
            match_label = _format_match_label(entry["matched_paras"])
            pdf_path = resolve_pdf_path(raw_source)

            if pdf_path and pdf_path.endswith(".pdf"):
                pdf_elements.append(
                    cl.Pdf(
                        name=pdf_tag,
                        path=pdf_path,
                        page=int(page_num) if page_num is not None else 1,
                        display="side",
                    )
                )
            else:
                pdf_elements.append(
                    cl.Text(
                        name=pdf_tag,
                        content=f"### {pdf_tag} — {raw_source}\n\n**Page:** {page_num if page_num is not None else 'N/A'}\n\nFile not found on disk.",
                        display="side",
                    )
                )

            citation_lines.append(
                f"- {pdf_tag} `{raw_source}` ({page_str}) — {match_label}"
            )

        if citation_lines:
            response_text += "\n\n**Sources (PDF Pages):**\n" + "\n".join(citation_lines)

        if chunk_entries:
            detail_key = uuid.uuid4().hex
            store: Dict[str, List[Dict[str, Any]]] = (
                cl.user_session.get("chunk_details_store") or {}
            )
            store[detail_key] = chunk_entries
            if len(store) > 20:
                oldest_key = next(iter(store))
                del store[oldest_key]
            cl.user_session.set("chunk_details_store", store)

            actions.append(
                cl.Action(
                    name="show_chunk_details",
                    label=f"Show Retrieved Chunks ({len(chunk_entries)})",
                    payload={"key": detail_key},
                    tooltip="View retrieved chunk text, FlashRank scores, and highlighted matching passages",
                )
            )

    await cl.Message(
        content=response_text,
        elements=pdf_elements,
        actions=actions,
    ).send()

    cl.user_session.set("previous_elements", pdf_elements)
