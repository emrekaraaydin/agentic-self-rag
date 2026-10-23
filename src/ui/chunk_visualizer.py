from __future__ import annotations

import difflib
import logging
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

from config.settings import BASE_DIR, RERANK_DEFAULT_THRESHOLD

logger = logging.getLogger(__name__)

HIGHLIGHT_CACHE_DIR: Path = BASE_DIR / ".cache" / "highlighted_pdfs"
HIGHLIGHT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ChunkStyle:
    badge: str
    tier_label: str
    score_pct: float
    normalized_intensity: float
    highlight_rgb: Tuple[float, float, float]
    accent_rgb: Tuple[float, float, float]
    badge_bg_rgb: Tuple[float, float, float]
    highlight_opacity: float
    border_width: float
    progress_bar: str


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _lerp_color(
    color_low: Tuple[float, float, float],
    color_high: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    factor = _clamp(t, 0.0, 1.0)
    return (
        round(color_low[0] + (color_high[0] - color_low[0]) * factor, 4),
        round(color_low[1] + (color_high[1] - color_low[1]) * factor, 4),
        round(color_low[2] + (color_high[2] - color_low[2]) * factor, 4),
    )


def _build_progress_bar(score: float, width: int = 10) -> str:
    # Skoru gorsel ilerleme cubuguna donusturur (0.0 - 1.0 arasi)
    clamped = _clamp(score, 0.0, 1.0)
    filled = int(round(clamped * width))
    filled = max(1 if score > 0 else 0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def compute_chunk_style(
    score: float,
    threshold: float = RERANK_DEFAULT_THRESHOLD,
    batch_scores: Optional[Sequence[float]] = None,
) -> ChunkStyle:
    """
    Reranker skoruna ve belirlenen esige (threshold) gore goz yormayan,
    profesyonel pastel renk paleti ve vurgu siddeti hesaplar.
    """
    safe_score = _clamp(float(score or 0.0), 0.0, 1.0)
    safe_threshold = _clamp(float(threshold or RERANK_DEFAULT_THRESHOLD), 0.0, 0.95)
    score_pct = round(safe_score * 100.0, 1)

    # Esik uzerindeki mutlak normallestirme (karekok egrisi ile orta skorlari da ayirt edici yapar)
    abs_norm = _clamp((safe_score - safe_threshold) / max(1e-6, 1.0 - safe_threshold))
    perceptual_abs = abs_norm ** 0.65

    # Sorgudaki diger chunklara gore bagil konum
    if batch_scores and len(batch_scores) > 1:
        min_s = min(batch_scores)
        max_s = max(batch_scores)
        if max_s - min_s > 1e-5:
            rel_norm = _clamp((safe_score - min_s) / (max_s - min_s))
        else:
            rel_norm = perceptual_abs
        intensity = _clamp(0.72 * perceptual_abs + 0.28 * rel_norm)
    else:
        intensity = perceptual_abs

    # Goz yormayan profesyonel 4 kademeli renk paleti (PyMuPDF Multiply karisim moduna uygun)
    if safe_score >= 0.70:
        # Kademe 1: Yuksek Eslesme - Yumusak Zumrut / Adaçayı Yesili
        tier_t = _clamp((safe_score - 0.70) / 0.30)
        badge = "🟢"
        tier_label = "Yüksek Eşleşme"
        highlight_rgb = _lerp_color((0.62, 0.90, 0.72), (0.46, 0.84, 0.60), tier_t)
        accent_rgb = _lerp_color((0.16, 0.64, 0.40), (0.10, 0.54, 0.32), tier_t)
        badge_bg_rgb = (0.90, 0.97, 0.93)
        highlight_opacity = round(0.36 + 0.12 * tier_t, 3)
        border_width = round(2.8 + 0.9 * tier_t, 2)
    elif safe_score >= 0.35:
        # Kademe 2: Iyi Eslesme - Sicak Kehribar / Altin Sarisi
        tier_t = _clamp((safe_score - 0.35) / 0.35)
        badge = "🟡"
        tier_label = "İyi Eşleşme"
        highlight_rgb = _lerp_color((0.97, 0.89, 0.58), (0.95, 0.82, 0.44), tier_t)
        accent_rgb = _lerp_color((0.82, 0.58, 0.14), (0.76, 0.50, 0.08), tier_t)
        badge_bg_rgb = (0.99, 0.96, 0.88)
        highlight_opacity = round(0.27 + 0.09 * tier_t, 3)
        border_width = round(2.1 + 0.6 * tier_t, 2)
    elif safe_score >= 0.12:
        # Kademe 3: Orta Eslesme - Yumusak Somon / Kum Tonu
        tier_t = _clamp((safe_score - 0.12) / 0.23)
        badge = "🟠"
        tier_label = "Orta Eşleşme"
        highlight_rgb = _lerp_color((0.96, 0.82, 0.66), (0.95, 0.75, 0.52), tier_t)
        accent_rgb = _lerp_color((0.80, 0.48, 0.26), (0.76, 0.42, 0.18), tier_t)
        badge_bg_rgb = (0.99, 0.94, 0.89)
        highlight_opacity = round(0.20 + 0.07 * tier_t, 3)
        border_width = round(1.6 + 0.4 * tier_t, 2)
    else:
        # Kademe 4: Sinirda / Esik Ustu - Dingin Buz / Arduvaz Mavisi
        tier_t = _clamp(
            (safe_score - safe_threshold) / max(1e-4, 0.12 - safe_threshold)
        )
        badge = "🔵"
        tier_label = "Sınırda (Eşik Üstü)"
        highlight_rgb = _lerp_color((0.78, 0.87, 0.96), (0.68, 0.81, 0.94), tier_t)
        accent_rgb = _lerp_color((0.44, 0.60, 0.78), (0.34, 0.52, 0.72), tier_t)
        badge_bg_rgb = (0.93, 0.96, 0.99)
        highlight_opacity = round(0.14 + 0.06 * tier_t, 3)
        border_width = round(1.2 + 0.3 * tier_t, 2)

    return ChunkStyle(
        badge=badge,
        tier_label=tier_label,
        score_pct=score_pct,
        normalized_intensity=round(intensity, 3),
        highlight_rgb=highlight_rgb,
        accent_rgb=accent_rgb,
        badge_bg_rgb=badge_bg_rgb,
        highlight_opacity=highlight_opacity,
        border_width=border_width,
        progress_bar=_build_progress_bar(safe_score),
    )


def format_chunk_pdf_label(
    rank_idx: int,
    style: ChunkStyle,
    page_num: Optional[int],
) -> str:
    """
    Chainlit mesajinda tiklanabilir PDF kaynak etiketi olusturur.
    Not: CommonMark link ayristirmasinda ic ice koseli parantez cakismasini
    onlemek icin yuvarlak parantez kullanilir.
    """
    page_str = f"S.{page_num}" if page_num is not None else "S.?"
    return f"{style.badge} Chunk {rank_idx} (%{style.score_pct:.1f} | {page_str})"


def format_chunk_text_label(rank_idx: int) -> str:
    """Chainlit yan panelinde chunk metin kartini acan tiklanabilir etiket."""
    return f"📄 Chunk {rank_idx} Metni"


def _strip_injected_header_prefix(chunk_text: str) -> Tuple[Optional[str], str]:
    """
    splitter.py tarafindan metnin basina eklenen '[Header 1 > Header 2]\\n'
    baglam on ekini ayirir.
    """
    text = (chunk_text or "").strip()
    match = re.match(r"^\[([^\[\]\n]+)\]\s*\n(.*)$", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, text


def _normalize_token(word: str) -> str:
    return re.sub(r"[^\w]", "", word, flags=re.UNICODE).lower()


def _find_chunk_line_rects_on_page(
    page: fitz.Page,
    chunk_text: str,
) -> List[fitz.Rect]:
    """
    Verilen PDF sayfasinda chunk metnine karsilik gelen satir dikdortgenlerini (fitz.Rect)
    kelime dizisi eslestirmesi (SequenceMatcher) ve arama yedegi ile tespit eder.
    """
    _, body_text = _strip_injected_header_prefix(chunk_text)
    if not body_text:
        return []

    chunk_tokens = [_normalize_token(w) for w in body_text.split()]
    chunk_tokens = [t for t in chunk_tokens if t]
    if not chunk_tokens:
        return []

    raw_words = page.get_text("words")
    if not raw_words:
        return []

    # (page_word_index, rect, block_no, line_no, norm_token)
    indexed_page_words: List[Tuple[int, fitz.Rect, int, int, str]] = []
    page_tokens: List[str] = []

    for idx, item in enumerate(raw_words):
        x0, y0, x1, y1, word_str, block_no, line_no, _ = item[:8]
        norm = _normalize_token(str(word_str))
        if norm:
            indexed_page_words.append(
                (idx, fitz.Rect(x0, y0, x1, y1), int(block_no), int(line_no), norm)
            )
            page_tokens.append(norm)

    if not page_tokens:
        return []

    matcher = difflib.SequenceMatcher(None, page_tokens, chunk_tokens, autojunk=False)
    matching_blocks = [
        block for block in matcher.get_matching_blocks() if block.size >= 3
    ]

    # Eger 3+ kelimelik blok azsa 2+ kelimelik bloklari da degerlendir
    if sum(b.size for b in matching_blocks) < min(6, len(chunk_tokens)):
        matching_blocks = [
            block for block in matcher.get_matching_blocks() if block.size >= 2
        ]

    matched_page_indices: set[int] = set()
    if matching_blocks:
        # Ardisik eslesen bloklar arasindaki kucuk (1-5 kelimelik) tireleme/OCR bosluklarini doldur
        for idx_b, block in enumerate(matching_blocks):
            for pos in range(block.a, block.a + block.size):
                matched_page_indices.add(pos)

            if idx_b + 1 < len(matching_blocks):
                next_block = matching_blocks[idx_b + 1]
                page_gap = next_block.a - (block.a + block.size)
                chunk_gap = next_block.b - (block.b + block.size)
                if 0 < page_gap <= 6 and 0 <= chunk_gap <= 6:
                    for pos in range(block.a + block.size, next_block.a):
                        matched_page_indices.add(pos)

    # Kelime eslesmesi yeterli degilse cumle/ifade bazli search_for yedegi uygula
    if len(matched_page_indices) < min(4, len(chunk_tokens)):
        clean_plain = re.sub(r"[#*`_>|~-]+", " ", body_text)
        clean_plain = re.sub(r"\s+", " ", clean_plain).strip()
        words_list = clean_plain.split()
        fallback_rects: List[fitz.Rect] = []
        step = 6
        for i in range(0, len(words_list), step):
            phrase = " ".join(words_list[i : i + step])
            if len(phrase) >= 12:
                found = page.search_for(phrase)
                fallback_rects.extend(found)
        return fallback_rects

    # Eslesen kelimeleri (block_no, line_no) bazinda gruplayarak puruzsuz satir kutulari olustur
    line_groups: Dict[Tuple[int, int], List[fitz.Rect]] = {}
    for token_pos in sorted(matched_page_indices):
        _, word_rect, block_no, line_no, _ = indexed_page_words[token_pos]
        key = (block_no, line_no)
        line_groups.setdefault(key, []).append(word_rect)

    line_rects: List[fitz.Rect] = []
    for rects in line_groups.values():
        merged = fitz.Rect(rects[0])
        for r in rects[1:]:
            merged.include_rect(r)
        if not merged.is_empty and merged.width > 2.0 and merged.height > 2.0:
            line_rects.append(merged)

    line_rects.sort(key=lambda r: (round(r.y0, 1), round(r.x0, 1)))
    return line_rects


def _resolve_best_page_and_rects(
    doc: fitz.Document,
    reported_page: Optional[int],
    chunk_text: str,
) -> Tuple[int, List[fitz.Rect]]:
    """
    Chunk metninin PDF icindeki gercek sayfasini (1-tabanli) ve satir kutularini bulur.
    """
    total_pages = len(doc)
    if total_pages == 0:
        return 1, []

    # Aday sayfa indeksleri (0-tabanli)
    candidate_indices: List[int] = []
    if reported_page is not None:
        try:
            p_int = int(reported_page)
            for idx in (p_int - 1, p_int, p_int - 2):
                if 0 <= idx < total_pages and idx not in candidate_indices:
                    candidate_indices.append(idx)
        except (ValueError, TypeError):
            pass

    if not candidate_indices:
        candidate_indices = [0]

    best_idx = candidate_indices[0]
    best_rects: List[fitz.Rect] = []

    for page_idx in candidate_indices:
        rects = _find_chunk_line_rects_on_page(doc[page_idx], chunk_text)
        if len(rects) > len(best_rects):
            best_rects = rects
            best_idx = page_idx
        if len(best_rects) >= 3:
            break

    return best_idx + 1, best_rects


def _annotate_chunk_on_page(
    page: fitz.Page,
    line_rects: List[fitz.Rect],
    style: ChunkStyle,
    rank_idx: int,
    is_active_target: bool,
) -> None:
    """
    Sayfa uzerindeki chunk satirlarini reranker skoruna gore dinamik renkle highlight eder
    ve sol kenar vurgu cubugu + odak etiketi cizer.
    """
    if not line_rects:
        return

    # 1. Satir bazli yumusak highlight (Multiply blend mode ile metin netligini korur)
    effective_opacity = (
        style.highlight_opacity
        if is_active_target
        else max(0.12, round(style.highlight_opacity * 0.65, 3))
    )

    for rect in line_rects:
        padded_line = fitz.Rect(
            rect.x0 - 1.0,
            rect.y0 - 0.5,
            rect.x1 + 1.0,
            rect.y1 + 0.5,
        )
        annot = page.add_highlight_annot(padded_line)
        annot.set_colors(stroke=style.highlight_rgb)
        annot.set_opacity(effective_opacity)
        annot.update()

    # 2. Chunk kapsayici sinir kutusu (Bounding Box)
    bbox = fitz.Rect(line_rects[0])
    for r in line_rects[1:]:
        bbox.include_rect(r)

    page_rect = page.rect
    pad = 4.0
    frame_rect = fitz.Rect(
        max(page_rect.x0 + 4.0, bbox.x0 - pad),
        max(page_rect.y0 + 4.0, bbox.y0 - pad),
        min(page_rect.x1 - 4.0, bbox.x1 + pad),
        min(page_rect.y1 - 4.0, bbox.y1 + pad),
    )

    # 3. Sol kenar dinamik vurgu cubugu (Skor yukseldikce daha kalin ve belirgin)
    bar_x = max(page_rect.x0 + 3.0, frame_rect.x0 - 4.5)
    shape = page.new_shape()
    shape.draw_line(
        fitz.Point(bar_x, frame_rect.y0),
        fitz.Point(bar_x, frame_rect.y1),
    )
    shape.finish(
        color=style.accent_rgb,
        width=style.border_width if is_active_target else max(1.2, style.border_width * 0.7),
        stroke_opacity=0.88 if is_active_target else 0.55,
    )

    # 4. Secili (tiklanan) chunk icin zarif cerceve ve skor rozeti
    if is_active_target:
        box_opacity = _clamp(0.32 + 0.38 * style.normalized_intensity, 0.30, 0.72)
        fill_opacity = _clamp(0.03 + 0.05 * style.normalized_intensity, 0.03, 0.08)
        shape.draw_rect(frame_rect)
        shape.finish(
            color=style.accent_rgb,
            fill=style.highlight_rgb,
            width=max(0.9, style.border_width * 0.45),
            stroke_opacity=box_opacity,
            fill_opacity=fill_opacity,
        )

        # Ust-sol bilgi etiketi (Chunk # • %Skor • Seviye)
        label_text = f"Chunk {rank_idx}  |  Rerank: %{style.score_pct:.1f} ({style.tier_label})"
        pill_height = 13.0
        pill_width = min(page_rect.width - 20.0, max(135.0, len(label_text) * 4.3))
        pill_y0 = max(page_rect.y0 + 2.0, frame_rect.y0 - pill_height - 1.5)
        pill_x0 = frame_rect.x0
        pill_rect = fitz.Rect(
            pill_x0,
            pill_y0,
            pill_x0 + pill_width,
            pill_y0 + pill_height,
        )

        shape.draw_rect(pill_rect)
        shape.finish(
            color=style.accent_rgb,
            fill=style.badge_bg_rgb,
            width=0.8,
            stroke_opacity=0.85,
            fill_opacity=0.94,
        )
        shape.commit()

        page.insert_textbox(
            fitz.Rect(pill_rect.x0 + 4.0, pill_rect.y0 + 1.5, pill_rect.x1 - 2.0, pill_rect.y1),
            label_text,
            fontsize=7.2,
            fontname="helv",
            color=style.accent_rgb,
            align=fitz.TEXT_ALIGN_LEFT,
        )
    else:
        shape.commit()


def build_highlighted_chunk_pdf(
    pdf_path: str,
    target_rank_idx: int,
    target_source: Dict[str, Any],
    target_style: ChunkStyle,
    same_pdf_items: Optional[List[Tuple[int, Dict[str, Any], ChunkStyle]]] = None,
) -> Tuple[str, int]:
    """
    Hedef chunk'i (ve varsa ayni PDF icindeki diger esik ustu chunklari)
    reranker skoruna gore dinamik olarak renklendirip gecici bir PDF dosyasina kaydeder.
    Donus: (olusturulan_pdf_yolu, acilacak_sayfa_numarasi_1_tabanli)
    """
    reported_page = target_source.get("page")
    default_page = int(reported_page) if reported_page is not None else 1

    if fitz is None:
        return pdf_path, default_page

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning("Failed to open PDF for highlighting (%s): %s", pdf_path, exc)
        return pdf_path, default_page

    try:
        # Once ayni PDF'teki diger chunklari ikincil vurguyla isaretle
        if same_pdf_items:
            for other_rank_idx, other_src, other_style in same_pdf_items:
                if other_rank_idx == target_rank_idx:
                    continue
                other_content = other_src.get("content") or ""
                other_page_num, other_rects = _resolve_best_page_and_rects(
                    doc, other_src.get("page"), other_content
                )
                if other_rects:
                    _annotate_chunk_on_page(
                        page=doc[other_page_num - 1],
                        line_rects=other_rects,
                        style=other_style,
                        rank_idx=other_rank_idx,
                        is_active_target=False,
                    )

        # Hedef (tiklanan) chunk'i birincil odak vurgusuyla isaretle
        target_content = target_source.get("content") or ""
        resolved_page, target_rects = _resolve_best_page_and_rects(
            doc, reported_page, target_content
        )

        if target_rects:
            _annotate_chunk_on_page(
                page=doc[resolved_page - 1],
                line_rects=target_rects,
                style=target_style,
                rank_idx=target_rank_idx,
                is_active_target=True,
            )

        stem = Path(pdf_path).stem
        out_filename = f"{stem}_chunk_{target_rank_idx}_{uuid.uuid4().hex[:8]}.pdf"
        out_path = HIGHLIGHT_CACHE_DIR / out_filename
        doc.save(str(out_path), garbage=1, deflate=True)
        return str(out_path), resolved_page
    except Exception as exc:
        logger.warning("Error while annotating PDF chunk (%s): %s", pdf_path, exc)
        return pdf_path, default_page
    finally:
        doc.close()


def cleanup_highlighted_pdfs(file_paths: Sequence[str]) -> None:
    """Onceki sorguda olusturulan gecici highlight edilmis PDF dosyalarini temizler."""
    for path_str in file_paths:
        try:
            path_obj = Path(path_str)
            if (
                path_obj.is_file()
                and HIGHLIGHT_CACHE_DIR in path_obj.resolve().parents
            ):
                os.remove(path_obj)
        except Exception:
            pass


def format_chunk_detail_card(
    rank_idx: int,
    source_data: Dict[str, Any],
    style: ChunkStyle,
) -> str:
    """
    Yan panelde (cl.Text) gosterilecek zenginlestirilmis chunk metin ve skor karti.
    """
    raw_source = source_data.get("source", "unknown")
    page_num = source_data.get("page")
    chunk_id = source_data.get("chunk_id", f"doc_{rank_idx}")
    chunk_index = source_data.get("chunk_index", rank_idx - 1)
    rerank_score = float(source_data.get("rerank_score") or 0.0)
    retrieval_score = source_data.get("retrieval_score")
    threshold = float(source_data.get("threshold") or RERANK_DEFAULT_THRESHOLD)
    raw_content = source_data.get("content") or ""
    headers: List[str] = source_data.get("headers") or []

    injected_header, clean_body = _strip_injected_header_prefix(raw_content)
    header_display = (
        " > ".join(headers)
        if headers
        else (injected_header if injected_header else "Genel Bölüm")
    )

    threshold_ratio = rerank_score / max(1e-6, threshold)
    cosine_str = f"`{float(retrieval_score):.4f}`" if retrieval_score is not None else "`N/A`"

    lines = [
        f"### {style.badge} Chunk {rank_idx} — {style.tier_label}",
        f"**Reranker Skoru:** `%{style.score_pct:.2f}` (`{rerank_score:.4f}`)  `{style.progress_bar}`",
        "",
        "| Metrik / Bilgi | Değer |",
        "| :--- | :--- |",
        f"| **Kaynak Dosya** | `{raw_source}` |",
        f"| **Sayfa / Chunk Sırası** | Sayfa `{page_num if page_num is not None else 'N/A'}` • Parça `#{chunk_index}` |",
        f"| **Bölüm Başlığı** | {header_display} |",
        f"| **Reranker Eşiği (Threshold)** | `%{threshold * 100:.1f}` (`{threshold:.4f}`) — **{threshold_ratio:.1f}x** eşik üstü |",
        f"| **Qdrant Vektör Skoru** | {cosine_str} |",
        f"| **Chunk ID** | `{chunk_id}` |",
        "",
        "---",
        "#### 📝 Chunk İçeriği",
        "",
        clean_body if clean_body else "*(Chunk metni bulunamadı)*",
    ]
    return "\n".join(lines)


def build_sources_markdown_section(
    source_rows: List[Dict[str, Any]],
    threshold: float = RERANK_DEFAULT_THRESHOLD,
) -> str:
    """
    Yanit mesajinin altina eklenecek, tiklanabilir chunk etiketlerini ve
    dinamik skor gostergelerini iceren Markdown kaynak bolumu.
    """
    if not source_rows:
        return ""

    threshold_pct = threshold * 100.0
    lines = [
        "",
        "---",
        f"**📚 Kaynak Chunklar** *(Reranker Eşiği: `%{threshold_pct:.1f}` • "
        "🟢 Yüksek `≥%70` · 🟡 İyi `%35-%70` · 🟠 Orta `%12-%35` · 🔵 Sınırda `<%12`)*",
        "",
    ]

    for row in source_rows:
        pdf_tag = row["pdf_label"]
        text_tag = row.get("text_label")
        style: ChunkStyle = row["style"]
        source_name = row["source_name"]

        if text_tag:
            lines.append(
                f"- {pdf_tag} · {text_tag} — `{style.progress_bar}` *{ style.tier_label }* (`{source_name}`)"
            )
        else:
            lines.append(
                f"- {pdf_tag} — `{style.progress_bar}` *{ style.tier_label }* (`{source_name}`)"
            )

    return "\n".join(lines)
