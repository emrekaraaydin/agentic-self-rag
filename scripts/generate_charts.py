#!/usr/bin/env python3
"""Reproducible dark-card benchmark chart generator for Agentic RAG.

Computes all retrieval, safety, latency, and 32B RAGAS slice metrics directly
from the raw JSON/CSV evaluation artifacts (plus the documented 100-query
manual ground-truth audit) using pure Python stdlib + Pillow + NumPy only.

Outputs (2x supersampled 2800x1560 -> LANCZOS 1400x780 PNG):
  - assets/benchmark_comparison.png
  - assets/ragas_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Canvas, Supersampling & Theme Constants (Matches assets/architecture_flow.gif)
# ---------------------------------------------------------------------------
LOGICAL_W, LOGICAL_H = 1400, 780
SCALE = 2
SUP_W, SUP_H = LOGICAL_W * SCALE, LOGICAL_H * SCALE

BG_HEX = "#0d1117"
CARD_BORDER_HEX = "#21262d"
DOT_GRID_HEX = "#1a2230"
PANEL_BG_HEX = "#161b22"
SUBCARD_BG_HEX = "#131922"
PANEL_BORDER_HEX = "#30363d"
SUBCARD_BORDER_HEX = "#26303d"
GRIDLINE_HEX = "#21262d"

TEXT_PRIMARY_HEX = "#f0f6fc"
TEXT_SECONDARY_HEX = "#8b949e"
TEXT_MUTED_HEX = "#6e7681"

NAIVE_BAR_HEX = "#64748b"
NAIVE_VAL_HEX = "#cbd5e1"
V2_BAR_HEX = "#38bdf8"
V2_VAL_HEX = "#7dd3fc"

POS_TEXT_HEX = "#34d399"
POS_BG_HEX = "#0d2a20"
POS_BORDER_HEX = "#166534"
POS_ACCENT_HEX = "#22c55e"

NEG_TEXT_HEX = "#f87171"
NEG_BG_HEX = "#2d1216"
NEG_BORDER_HEX = "#7f1d1d"
NEG_ACCENT_HEX = "#ef4444"

INFO_BG_HEX = "#0f2436"
INFO_BORDER_HEX = "#1e4976"


def hex_to_rgb(hx: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB hex string to an (R, G, B) tuple."""
    hx = hx.lstrip("#")
    return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))


BG_RGB = hex_to_rgb(BG_HEX)
CARD_BORDER_RGB = hex_to_rgb(CARD_BORDER_HEX)
DOT_GRID_RGB = hex_to_rgb(DOT_GRID_HEX)
PANEL_BG_RGB = hex_to_rgb(PANEL_BG_HEX)
SUBCARD_BG_RGB = hex_to_rgb(SUBCARD_BG_HEX)
PANEL_BORDER_RGB = hex_to_rgb(PANEL_BORDER_HEX)
SUBCARD_BORDER_RGB = hex_to_rgb(SUBCARD_BORDER_HEX)
GRIDLINE_RGB = hex_to_rgb(GRIDLINE_HEX)

TEXT_PRIMARY_RGB = hex_to_rgb(TEXT_PRIMARY_HEX)
TEXT_SECONDARY_RGB = hex_to_rgb(TEXT_SECONDARY_HEX)
TEXT_MUTED_RGB = hex_to_rgb(TEXT_MUTED_HEX)

NAIVE_BAR_RGB = hex_to_rgb(NAIVE_BAR_HEX)
NAIVE_VAL_RGB = hex_to_rgb(NAIVE_VAL_HEX)
V2_BAR_RGB = hex_to_rgb(V2_BAR_HEX)
V2_VAL_RGB = hex_to_rgb(V2_VAL_HEX)

POS_TEXT_RGB = hex_to_rgb(POS_TEXT_HEX)
POS_BG_RGB = hex_to_rgb(POS_BG_HEX)
POS_BORDER_RGB = hex_to_rgb(POS_BORDER_HEX)
POS_ACCENT_RGB = hex_to_rgb(POS_ACCENT_HEX)

NEG_TEXT_RGB = hex_to_rgb(NEG_TEXT_HEX)
NEG_BG_RGB = hex_to_rgb(NEG_BG_HEX)
NEG_BORDER_RGB = hex_to_rgb(NEG_BORDER_HEX)
NEG_ACCENT_RGB = hex_to_rgb(NEG_ACCENT_HEX)

INFO_BG_RGB = hex_to_rgb(INFO_BG_HEX)
INFO_BORDER_RGB = hex_to_rgb(INFO_BORDER_HEX)


# ---------------------------------------------------------------------------
# Documented Manual Ground-Truth Human Audit (Cross-Referenced with README.md)
# ---------------------------------------------------------------------------
# Source notes:
#   - Every response across all 100 queries in golden_dataset_100.json was
#     audited manually against ground_truth (see README.md "Full 100-Question
#     Ground-Truth Audit").
#   - Naive RAG (naive_rag_evaluation_results.json):
#       80 in-scope queries -> 49 Full Correct + 5 Partial Correct + 15 Wrong /
#       Hallucinated + 11 Unnecessary Refusal (11 clean, 0 fallback).
#       Hallucination Rate = 15/80 (18.75% -> 18.8%)
#       Precision Among Answered = 54/69 (78.26% -> 78.3%)
#       Total Useful = 54/80 (67.5%)
#   - Agentic v2 ModernBERT-NLI (evaluation_results_nli_100.json):
#       80 in-scope queries -> 46 Full Correct + 8 Partial Correct + 6 Wrong /
#       Hallucinated + 20 Unnecessary Refusal (8 clean, 12 fallback_node).
#       Hallucination Rate = 6/80 (7.50% -> 7.5%, -60% relative vs Naive)
#       Precision Among Answered = 54/60 (90.00% -> 90.0%)
#       Total Useful = 54/80 (67.5%)
#       Head-to-Head vs Naive (C > P > R > W): 16W / 11L / 53T (Net +5)
MANUAL_GROUND_TRUTH_AUDIT: Dict[str, Dict[str, object]] = {
    "naive": {
        "n_in_scope": 80,
        "full_correct": 49,
        "partial_correct": 5,
        "wrong_hallucinated": 15,
        "unnecessary_refusal": 11,
        "in_scope_clean_refusal": 11,
        "in_scope_fallback": 0,
    },
    "agentic_v2": {
        "n_in_scope": 80,
        "full_correct": 46,
        "partial_correct": 8,
        "wrong_hallucinated": 6,
        "unnecessary_refusal": 20,
        "in_scope_clean_refusal": 8,
        "in_scope_fallback": 12,
        "head_to_head": "16W / 11L / 53T (Net +5)",
    },
}


# ---------------------------------------------------------------------------
# Verbatim Clean Refusal Patterns (Replicated Inline to Avoid Heavy Imports)
# ---------------------------------------------------------------------------
# Copied verbatim from src/nodes/grade_hallucination_node.py so this generator
# has zero dependency on torch, transformers, or LangGraph packages.
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

_HEAD_REFUSAL_PHRASES: Tuple[str, ...] = (
    "the knowledge base does not contain",
    "the knowledge base lacks",
    "knowledge base does not contain",
    "a verified answer could not be generated",
    "no relevant documents",
)

RAGAS_METRICS: Tuple[str, ...] = (
    "faithfulness",
    "context_recall",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "answer_correctness",
)


def detect_clean_refusal(response: str) -> bool:
    """Return True when response is a grounded refusal without positive facts."""
    low = (response or "").lower()
    return any(p in low for p in _REFUSAL_PATTERNS) and not any(
        ind in low for ind in _POSITIVE_FACT_INDICATORS
    )


def is_positive_answer(csv_row: Dict[str, str], json_rec: Dict[str, object]) -> bool:
    """Return True when a RAGAS CSV row represents a non-fallback positive answer."""
    fb = json_rec.get("fallback_status")
    if isinstance(fb, dict) and fb.get("is_fallback"):
        return False
    resp = csv_row.get("response") or ""
    if not resp.strip():
        return False
    if detect_clean_refusal(resp):
        return False
    head = resp.strip()[:120].lower()
    return not any(phrase in head for phrase in _HEAD_REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Data Structures & Computation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JsonMetrics:
    name: str
    n_total: int
    n_in_scope: int
    target_doc_hits: int
    cross_doc_hits: int
    cross_doc_total: int
    purity_pct: float
    ood_cutoffs: int
    ood_total: int
    in_scope_fallback: int
    mean_latency_sec: float

    @property
    def target_doc_hit_pct(self) -> float:
        return self.target_doc_hits / self.n_in_scope * 100.0

    @property
    def cross_doc_hit_pct(self) -> float:
        return self.cross_doc_hits / self.cross_doc_total * 100.0 if self.cross_doc_total else 0.0

    @property
    def ood_cutoff_pct(self) -> float:
        return self.ood_cutoffs / self.ood_total * 100.0 if self.ood_total else 0.0


@dataclass(frozen=True)
class AuditMetrics:
    hallucination_num: int
    hallucination_den: int
    hallucination_pct: float
    precision_num: int
    precision_den: int
    precision_pct: float
    useful_num: int
    useful_den: int
    useful_pct: float


@dataclass(frozen=True)
class RagasSlices:
    raw_naive_n: int
    raw_v2_n: int
    v2_non_fb_n: int
    pos_naive_n: int
    pos_v2_n: int
    common_n: int
    v2_fallback_csv_n: int
    raw_naive: Tuple[float, ...]
    raw_v2: Tuple[float, ...]
    v2_non_fb: Tuple[float, ...]
    pos_naive: Tuple[float, ...]
    pos_v2: Tuple[float, ...]
    common_naive: Tuple[float, ...]
    common_v2: Tuple[float, ...]


@dataclass(frozen=True)
class BenchmarkBundle:
    naive_json: JsonMetrics
    v2_json: JsonMetrics
    v3_json: JsonMetrics
    naive_audit: AuditMetrics
    v2_audit: AuditMetrics
    head_to_head: str
    ragas: RagasSlices


def _load_json_records(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, list), f"Expected JSON list in {path}"
    return data


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _compute_json_metrics(records: Sequence[Dict[str, object]], name: str) -> JsonMetrics:
    in_scope = [r for r in records if r.get("category") != "fallback"]
    hits = sum(1 for r in in_scope if bool(r.get("target_doc_hit")))

    xdoc = [r for r in records if r.get("subcategory") == "cross_doc_multi_hop"]
    xdoc_hits = sum(1 for r in xdoc if bool(r.get("target_doc_hit")))

    purities: List[float] = []
    for r in in_scope:
        td = r.get("target_doc")
        if isinstance(td, list):
            targets = set(td)
        elif isinstance(td, str) and td:
            targets = {td}
        else:
            targets = set()
        srcs = r.get("sources") or []
        if isinstance(srcs, list) and len(srcs) > 0:
            matches = sum(1 for s in srcs if isinstance(s, dict) and s.get("source") in targets)
            purities.append(matches / len(srcs))
        else:
            purities.append(0.0)

    ood = [r for r in records if r.get("subcategory") == "out_of_domain"]
    ood_fb = sum(
        1
        for r in ood
        if isinstance(r.get("fallback_status"), dict) and bool(r["fallback_status"].get("is_fallback"))
    )

    in_scope_fb = sum(
        1
        for r in in_scope
        if isinstance(r.get("fallback_status"), dict) and bool(r["fallback_status"].get("is_fallback"))
    )

    mean_lat = statistics.mean(float(r["latency_sec"]) for r in records)

    return JsonMetrics(
        name=name,
        n_total=len(records),
        n_in_scope=len(in_scope),
        target_doc_hits=hits,
        cross_doc_hits=xdoc_hits,
        cross_doc_total=len(xdoc),
        purity_pct=statistics.mean(purities) * 100.0,
        ood_cutoffs=ood_fb,
        ood_total=len(ood),
        in_scope_fallback=in_scope_fb,
        mean_latency_sec=mean_lat,
    )


def _compute_audit_metrics(entry: Dict[str, object]) -> AuditMetrics:
    n_scope = int(entry["n_in_scope"])
    full_c = int(entry["full_correct"])
    part_c = int(entry["partial_correct"])
    wrong = int(entry["wrong_hallucinated"])
    refusal = int(entry["unnecessary_refusal"])
    clean_ref = int(entry["in_scope_clean_refusal"])
    fb = int(entry["in_scope_fallback"])

    assert full_c + part_c + wrong + refusal == n_scope, "Audit breakdown must sum to n_in_scope"
    assert clean_ref + fb == refusal, "Clean refusals + fallbacks must equal unnecessary_refusal"

    answered = n_scope - refusal
    useful = full_c + part_c
    return AuditMetrics(
        hallucination_num=wrong,
        hallucination_den=n_scope,
        hallucination_pct=wrong / n_scope * 100.0,
        precision_num=useful,
        precision_den=answered,
        precision_pct=useful / answered * 100.0,
        useful_num=useful,
        useful_den=n_scope,
        useful_pct=useful / n_scope * 100.0,
    )


def _csv_slice_means(rows: Sequence[Dict[str, str]]) -> Tuple[float, ...]:
    means: List[float] = []
    for col in RAGAS_METRICS:
        vals = [float(r[col]) for r in rows]
        assert all(math.isfinite(v) for v in vals), f"Non-finite value in column {col}"
        means.append(statistics.mean(vals))
    return tuple(means)


def load_and_verify_all(data_dir: Path) -> BenchmarkBundle:
    """Load all 5 raw artifacts, compute metrics, and assert every invariant."""
    naive_records = _load_json_records(data_dir / "naive_rag_evaluation_results.json")
    v2_records = _load_json_records(data_dir / "evaluation_results_nli_100.json")
    v3_records = _load_json_records(data_dir / "evaluation_results_v3_30.json")
    ragas_naive_rows = _load_csv_rows(data_dir / "ragas_results_naive_80.csv")
    ragas_v2_rows = _load_csv_rows(data_dir / "ragas_results_v2_80.csv")

    assert len(naive_records) == 100
    assert len(v2_records) == 100
    assert len(v3_records) == 30
    assert len(ragas_naive_rows) == 80
    assert len(ragas_v2_rows) == 79

    naive_m = _compute_json_metrics(naive_records, "Naive RAG")
    v2_m = _compute_json_metrics(v2_records, "Agentic v2")
    v3_m = _compute_json_metrics(v3_records, "Agentic v3")

    assert naive_m.n_in_scope == 80 and v2_m.n_in_scope == 80 and v3_m.n_in_scope == 25
    assert naive_m.target_doc_hits == 73 and v2_m.target_doc_hits == 77 and v3_m.target_doc_hits == 25
    assert naive_m.cross_doc_hits == 8 and naive_m.cross_doc_total == 15
    assert v2_m.cross_doc_hits == 13 and v2_m.cross_doc_total == 15
    assert naive_m.ood_cutoffs == 0 and naive_m.ood_total == 8
    assert v2_m.ood_cutoffs == 8 and v2_m.ood_total == 8
    assert v3_m.ood_cutoffs == 2 and v3_m.ood_total == 2

    naive_audit = _compute_audit_metrics(MANUAL_GROUND_TRUTH_AUDIT["naive"])
    v2_audit = _compute_audit_metrics(MANUAL_GROUND_TRUTH_AUDIT["agentic_v2"])

    assert MANUAL_GROUND_TRUTH_AUDIT["naive"]["in_scope_fallback"] == naive_m.in_scope_fallback == 0
    assert MANUAL_GROUND_TRUTH_AUDIT["agentic_v2"]["in_scope_fallback"] == v2_m.in_scope_fallback == 12

    naive_by_id = {str(r["id"]): r for r in naive_records}
    v2_by_id = {str(r["id"]): r for r in v2_records}

    v2_csv_ids = {r["id"] for r in ragas_v2_rows}
    assert "gdpr_lm02" in v2_by_id and "gdpr_lm02" not in v2_csv_ids
    assert not (v2_by_id["gdpr_lm02"].get("contexts") or []), "gdpr_lm02 expected empty context in v2"

    v2_fb_rows = [
        r for r in ragas_v2_rows
        if isinstance(v2_by_id[r["id"]].get("fallback_status"), dict)
        and bool(v2_by_id[r["id"]]["fallback_status"].get("is_fallback"))
    ]
    assert len(v2_fb_rows) == 11, f"Expected 11 fallback rows in CSV, got {len(v2_fb_rows)}"
    for fb_row in v2_fb_rows:
        assert float(fb_row["faithfulness"]) == 0.0, f"Expected 0.0 faithfulness on {fb_row['id']}"
        assert float(fb_row["answer_relevancy"]) == 0.0, f"Expected 0.0 relevancy on {fb_row['id']}"

    v2_non_fb_rows = [
        r for r in ragas_v2_rows
        if not (
            isinstance(v2_by_id[r["id"]].get("fallback_status"), dict)
            and bool(v2_by_id[r["id"]]["fallback_status"].get("is_fallback"))
        )
    ]
    pos_naive_rows = [r for r in ragas_naive_rows if is_positive_answer(r, naive_by_id[r["id"]])]
    pos_v2_rows = [r for r in ragas_v2_rows if is_positive_answer(r, v2_by_id[r["id"]])]

    pos_naive_map = {r["id"]: r for r in pos_naive_rows}
    pos_v2_map = {r["id"]: r for r in pos_v2_rows}
    common_ids = [r["id"] for r in ragas_naive_rows if r["id"] in pos_naive_map and r["id"] in pos_v2_map]
    common_naive_rows = [pos_naive_map[qid] for qid in common_ids]
    common_v2_rows = [pos_v2_map[qid] for qid in common_ids]

    assert len(v2_non_fb_rows) == 68
    assert len(pos_naive_rows) == 66
    assert len(pos_v2_rows) == 54
    assert len(common_ids) == 52

    ragas = RagasSlices(
        raw_naive_n=len(ragas_naive_rows),
        raw_v2_n=len(ragas_v2_rows),
        v2_non_fb_n=len(v2_non_fb_rows),
        pos_naive_n=len(pos_naive_rows),
        pos_v2_n=len(pos_v2_rows),
        common_n=len(common_ids),
        v2_fallback_csv_n=len(v2_fb_rows),
        raw_naive=_csv_slice_means(ragas_naive_rows),
        raw_v2=_csv_slice_means(ragas_v2_rows),
        v2_non_fb=_csv_slice_means(v2_non_fb_rows),
        pos_naive=_csv_slice_means(pos_naive_rows),
        pos_v2=_csv_slice_means(pos_v2_rows),
        common_naive=_csv_slice_means(common_naive_rows),
        common_v2=_csv_slice_means(common_v2_rows),
    )

    return BenchmarkBundle(
        naive_json=naive_m,
        v2_json=v2_m,
        v3_json=v3_m,
        naive_audit=naive_audit,
        v2_audit=v2_audit,
        head_to_head=str(MANUAL_GROUND_TRUTH_AUDIT["agentic_v2"]["head_to_head"]),
        ragas=ragas,
    )


# ---------------------------------------------------------------------------
# Font & 2x Supersampled Drawing Helpers
# ---------------------------------------------------------------------------
class FontSet:
    """Loaded 2x-scaled macOS font cache with safe default fallback."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        self._tofu_cache: Dict[int, bytes] = {}

    def _load(self, path: str, index: int, px_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, px_size, index=index)
        except OSError:
            pass
        return ImageFont.load_default()

    def get(self, role: str, logical_size: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        px = max(8, int(round(logical_size * SCALE)))
        key = (role, px)
        if key not in self._cache:
            specs = {
                "sans_bold": ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
                "sans_medium": ("/System/Library/Fonts/HelveticaNeue.ttc", 10),
                "sans_regular": ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
                "mono_regular": ("/System/Library/Fonts/Menlo.ttc", 0),
                "mono_bold": ("/System/Library/Fonts/Menlo.ttc", 1),
            }
            path, idx = specs[role]
            self._cache[key] = self._load(path, idx, px)
        return self._cache[key]

    def verify_glyphs(self, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> None:
        """Fail fast if any non-space glyph resolves to a missing-glyph tofu box."""
        if not isinstance(font, ImageFont.FreeTypeFont):
            return
        fid = id(font)
        if fid not in self._tofu_cache:
            self._tofu_cache[fid] = bytes(font.getmask("\uffff"))
        tofu = self._tofu_cache[fid]
        for ch in text:
            if ch.isspace():
                continue
            mask = font.getmask(ch)
            assert bytes(mask) != tofu, f"Missing glyph {ch!r} in font {font.path}"


def s(v: float) -> float:
    return v * SCALE


def _round_half_up_1(val: float) -> float:
    return float(Decimal(str(val)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Tuple[float, float]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return (bbox[2] - bbox[0]) / SCALE, (bbox[3] - bbox[1]) / SCALE


def draw_text_left_vcenter(
    draw: ImageDraw.ImageDraw,
    fonts: FontSet,
    x_left: float,
    cy: float,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> float:
    fonts.verify_glyphs(font, text)
    bbox = draw.textbbox((0, 0), text, font=font)
    th = bbox[3] - bbox[1]
    x = s(x_left) - bbox[0]
    y = s(cy) - th / 2.0 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)
    return x_left + (bbox[2] - bbox[0]) / SCALE


def draw_text_right_vcenter(
    draw: ImageDraw.ImageDraw,
    fonts: FontSet,
    x_right: float,
    cy: float,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> float:
    fonts.verify_glyphs(font, text)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = s(x_right) - tw - bbox[0]
    y = s(cy) - th / 2.0 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)
    return x_right - tw / SCALE


def draw_text_centered(
    draw: ImageDraw.ImageDraw,
    fonts: FontSet,
    cx: float,
    cy: float,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    fonts.verify_glyphs(font, text)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = s(cx) - tw / 2.0 - bbox[0]
    y = s(cy) - th / 2.0 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def draw_inline_runs(
    draw: ImageDraw.ImageDraw,
    fonts: FontSet,
    x_left: float,
    cy: float,
    runs: Sequence[Tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, Tuple[int, int, int]]],
) -> float:
    """Draw multi-colored inline text segments cleanly sharing a single baseline."""
    cur_x = x_left
    # Align all segments against the ascent of the tallest font in the sequence
    ref_ascent = max(
        (r[1].getmetrics()[0] if isinstance(r[1], ImageFont.FreeTypeFont) else 10) for r in runs
    )
    # Center the line optically using the primary font's cap box
    primary_font = runs[0][1]
    pb = draw.textbbox((0, 0), "Ag09%", font=primary_font)
    line_h = pb[3] - pb[1]
    base_y = s(cy) - line_h / 2.0 - pb[1]

    for text, font, fill in runs:
        fonts.verify_glyphs(font, text)
        ascent = font.getmetrics()[0] if isinstance(font, ImageFont.FreeTypeFont) else ref_ascent
        y_adj = base_y + (ref_ascent - ascent)
        draw.text((s(cur_x), y_adj), text, font=font, fill=fill)
        cur_x += draw.textlength(text, font=font) / SCALE
    return cur_x


def draw_pill(
    draw: ImageDraw.ImageDraw,
    fonts: FontSet,
    cx: float,
    cy: float,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    text_color: Tuple[int, int, int],
    fill_color: Tuple[int, int, int],
    border_color: Tuple[int, int, int],
    pad_x: float = 9.0,
    pad_y: float = 4.5,
    radius: float = 10.5,
) -> Tuple[float, float, float, float]:
    tw, th = measure_text(draw, text, font)
    hw = tw / 2.0 + pad_x
    hh = th / 2.0 + pad_y
    x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
    draw.rounded_rectangle(
        (s(x1), s(y1), s(x2), s(y2)),
        radius=s(min(radius, hh)),
        fill=fill_color,
        outline=border_color,
        width=int(round(s(1.0))),
    )
    draw_text_centered(draw, fonts, cx, cy, text, font, text_color)
    return (x1, y1, x2, y2)


def draw_panel_with_top_accent(
    canvas: Image.Image,
    box: Tuple[float, float, float, float],
    radius: float,
    fill_color: Tuple[int, int, int],
    border_color: Tuple[int, int, int],
    accent_color: Optional[Tuple[int, int, int]] = None,
    accent_h: float = 3.0,
) -> None:
    """Draw a rounded card/panel with an optional flush rounded top accent bar."""
    x1, y1, x2, y2 = box
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (s(x1), s(y1), s(x2), s(y2)),
        radius=s(radius),
        fill=fill_color,
        outline=border_color,
        width=int(round(s(1.1))),
    )
    if accent_color is None:
        return

    px1, py1 = int(round(s(x1))), int(round(s(y1)))
    px2, py2 = int(round(s(x2))), int(round(s(y2)))
    w, h = px2 - px1, py2 - py1
    ah = max(2, int(round(s(accent_h))))

    # Render accent bar masked to the panel's rounded interior so corners hug cleanly
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle((0, 0, w - 1, h - 1), radius=s(radius), fill=255)

    strip = Image.new("RGB", (w, ah), accent_color)
    sub_mask = mask.crop((0, 0, w, ah))
    canvas.paste(strip, (px1, py1), sub_mask)

    # Re-stroke border crisp over the pasted edge
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (s(x1), s(y1), s(x2), s(y2)),
        radius=s(radius),
        outline=border_color,
        width=int(round(s(1.1))),
    )


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Tuple[int, int, int],
    width_logical: float = 1.0,
    dash_len: float = 4.0,
    gap_len: float = 4.0,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    w_px = max(1, int(round(s(width_logical))))
    while pos < dist:
        end_pos = min(dist, pos + dash_len)
        draw.line(
            (
                s(x1 + ux * pos),
                s(y1 + uy * pos),
                s(x1 + ux * end_pos),
                s(y1 + uy * end_pos),
            ),
            fill=color,
            width=w_px,
        )
        pos += dash_len + gap_len


def create_dark_base_canvas(
    fonts: FontSet,
    title: str,
    subtitle: str,
    legend_right_x: float = 1364.0,
) -> Image.Image:
    """Create the shared #0d1117 dark card canvas, dot grid, header, and legend."""
    # Build subtle 24px dot-grid in NumPy for crisp, instant background texture
    arr = np.empty((SUP_H, SUP_W, 3), dtype=np.uint8)
    arr[:, :] = BG_RGB

    grid_step = 24 * SCALE
    dot_r = 2  # ~1px logical radius
    y_start = 96 * SCALE
    y_end = 760 * SCALE
    x_start = 24 * SCALE
    x_end = (LOGICAL_W - 20) * SCALE
    for gy in range(y_start, y_end, grid_step):
        for gx in range(x_start, x_end, grid_step):
            arr[gy - dot_r : gy + dot_r, gx - dot_r : gx + dot_r] = DOT_GRID_RGB

    canvas = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(canvas)

    # Outer #21262d card border matching assets/architecture_flow.gif
    draw.rounded_rectangle(
        (s(2), s(2), s(LOGICAL_W - 2), s(LOGICAL_H - 2)),
        radius=s(10),
        outline=CARD_BORDER_RGB,
        width=int(round(s(1.2))),
    )

    # Header left title + subtitle
    draw_text_left_vcenter(draw, fonts, 36.0, 34.0, title, fonts.get("sans_bold", 21.5), TEXT_PRIMARY_RGB)
    draw_text_left_vcenter(draw, fonts, 36.0, 62.0, subtitle, fonts.get("mono_regular", 11.8), TEXT_SECONDARY_RGB)

    # Header right legend pills
    legend_items = [
        ("Naive RAG (Baseline)", NAIVE_BAR_RGB, NAIVE_VAL_RGB),
        ("Agentic v2 (ModernBERT-NLI)", V2_BAR_RGB, V2_VAL_RGB),
    ]
    font_leg = fonts.get("sans_medium", 11.5)
    cur_right = legend_right_x
    for label, swatch_col, txt_col in reversed(legend_items):
        tw, _ = measure_text(draw, label, font_leg)
        pill_w = 12.0 + 10.0 + 7.0 + tw + 12.0
        px1 = cur_right - pill_w
        px2 = cur_right
        py1, py2 = 32.0, 60.0
        draw.rounded_rectangle(
            (s(px1), s(py1), s(px2), s(py2)),
            radius=s(8),
            fill=SUBCARD_BG_RGB,
            outline=PANEL_BORDER_RGB,
            width=int(round(s(1.0))),
        )
        sw_x1 = px1 + 11.0
        sw_cy = (py1 + py2) / 2.0
        draw.rounded_rectangle(
            (s(sw_x1), s(sw_cy - 5.0), s(sw_x1 + 10.0), s(sw_cy + 5.0)),
            radius=s(2.5),
            fill=swatch_col,
        )
        draw_text_left_vcenter(draw, fonts, sw_x1 + 17.0, sw_cy, label, font_leg, txt_col)
        cur_right = px1 - 10.0

    # Header bottom hairline
    draw.line((s(36), s(88), s(LOGICAL_W - 36), s(88)), fill=CARD_BORDER_RGB, width=int(round(s(1.0))))
    return canvas


# ---------------------------------------------------------------------------
# Chart 1: Controlled Benchmark Comparison (1400 x 780)
# ---------------------------------------------------------------------------
def render_benchmark_chart(bundle: BenchmarkBundle, output_path: Path, fonts: FontSet) -> None:
    """Render assets/benchmark_comparison.png at 2x supersampling -> 1400x780."""
    canvas = create_dark_base_canvas(
        fonts,
        title="Controlled Benchmark: Naive RAG vs. Agentic v2 (ModernBERT-NLI)",
        subtitle="n = 100 regulatory queries across 5 statutes (GDPR, AI Act, NIS2, HIPAA, KVKK) · Shared Qdrant index & prompt",
    )
    draw = ImageDraw.Draw(canvas)

    na = bundle.naive_audit
    v2a = bundle.v2_audit
    nj = bundle.naive_json
    v2j = bundle.v2_json
    v3j = bundle.v3_json

    hal_rel = (v2a.hallucination_pct - na.hallucination_pct) / na.hallucination_pct * 100.0
    hal_pp = _round_half_up_1(v2a.hallucination_pct - na.hallucination_pct)
    prec_pp = _round_half_up_1(v2a.precision_pct - na.precision_pct)
    xdoc_pp = _round_half_up_1(v2j.cross_doc_hit_pct - nj.cross_doc_hit_pct)
    hit_pp = _round_half_up_1(v2j.target_doc_hit_pct - nj.target_doc_hit_pct)
    pur_pp = _round_half_up_1(v2j.purity_pct - nj.purity_pct)
    ood_pp = _round_half_up_1(v2j.ood_cutoff_pct - nj.ood_cutoff_pct)

    # 1. Top 4 Hero KPI Callout Cards (y = 104..218)
    kpis = [
        (
            "Hallucination Rate (n=80)",
            f"{na.hallucination_pct:.1f}%",
            f"{v2a.hallucination_pct:.1f}%",
            f"{na.hallucination_num}/{na.hallucination_den} -> {v2a.hallucination_num}/{v2a.hallucination_den}",
            f"{hal_rel:.0f}% relative",
        ),
        (
            "Answered Precision",
            f"{na.precision_pct:.1f}%",
            f"{v2a.precision_pct:.1f}%",
            f"{na.precision_num}/{na.precision_den} -> {v2a.precision_num}/{v2a.precision_den}",
            f"{prec_pp:+.1f} pp",
        ),
        (
            f"Cross-Doc 2/2 Hit (n={v2j.cross_doc_total})",
            f"{nj.cross_doc_hit_pct:.1f}%",
            f"{v2j.cross_doc_hit_pct:.1f}%",
            f"{nj.cross_doc_hits}/{nj.cross_doc_total} -> {v2j.cross_doc_hits}/{v2j.cross_doc_total}",
            f"{xdoc_pp:+.1f} pp",
        ),
        (
            f"Structural OOD Cutoff (n={v2j.ood_total})",
            f"{nj.ood_cutoff_pct:.1f}%",
            f"{v2j.ood_cutoff_pct:.1f}%",
            f"{nj.ood_cutoffs}/{nj.ood_total} -> {v2j.ood_cutoffs}/{v2j.ood_total}",
            "Zero LLM Gen",
        ),
    ]

    card_w = (1328.0 - 3 * 16.0) / 4.0  # 320.0px
    card_y1, card_y2 = 104.0, 218.0
    for idx, (label, v_naive, v_v2, frac_str, pill_text) in enumerate(kpis):
        x0 = 36.0 + idx * (card_w + 16.0)
        x1 = x0 + card_w
        draw_panel_with_top_accent(
            canvas,
            (x0, card_y1, x1, card_y2),
            radius=10.0,
            fill_color=PANEL_BG_RGB,
            border_color=PANEL_BORDER_RGB,
            accent_color=POS_ACCENT_RGB,
            accent_h=2.5,
        )
        draw = ImageDraw.Draw(canvas)
        draw_text_left_vcenter(draw, fonts, x0 + 18.0, card_y1 + 23.0, label, fonts.get("sans_medium", 11.8), TEXT_SECONDARY_RGB)
        draw_inline_runs(
            draw,
            fonts,
            x0 + 18.0,
            card_y1 + 54.0,
            [
                (v_naive, fonts.get("sans_bold", 21.0), NAIVE_VAL_RGB),
                (" -> ", fonts.get("sans_bold", 19.0), TEXT_MUTED_RGB),
                (v_v2, fonts.get("sans_bold", 21.0), V2_VAL_RGB),
            ],
        )
        draw_text_left_vcenter(draw, fonts, x0 + 18.0, card_y1 + 90.0, frac_str, fonts.get("mono_regular", 11.2), TEXT_SECONDARY_RGB)

        pill_font = fonts.get("mono_bold", 10.6)
        pw, _ = measure_text(draw, pill_text, pill_font)
        pill_cx = x1 - 16.0 - (pw / 2.0 + 8.5)
        draw_pill(
            draw,
            fonts,
            pill_cx,
            card_y1 + 90.0,
            pill_text,
            pill_font,
            POS_TEXT_RGB,
            POS_BG_RGB,
            POS_BORDER_RGB,
            pad_x=8.5,
            pad_y=4.0,
        )

    # 2. Main Horizontal Paired Bar Panel (x = 36..1364, y = 236..684)
    panel_x1, panel_y1, panel_x2, panel_y2 = 36.0, 236.0, 1364.0, 684.0
    draw_panel_with_top_accent(
        canvas,
        (panel_x1, panel_y1, panel_x2, panel_y2),
        radius=10.0,
        fill_color=PANEL_BG_RGB,
        border_color=PANEL_BORDER_RGB,
    )
    draw = ImageDraw.Draw(canvas)

    track_x0 = 376.0
    track_x1 = 1040.0
    track_w = track_x1 - track_x0  # 664px -> 100.0%
    grid_y_top = 256.0
    grid_y_bot = 648.0

    # Vertical percentage gridlines & bottom tick labels
    for pct in (0, 25, 50, 75, 100):
        gx = track_x0 + (pct / 100.0) * track_w
        if pct == 0:
            draw.line((s(gx), s(grid_y_top), s(gx), s(grid_y_bot)), fill=PANEL_BORDER_RGB, width=int(round(s(1.2))))
        else:
            draw_dashed_line(draw, (gx, grid_y_top), (gx, grid_y_bot), GRIDLINE_RGB, width_logical=1.0, dash_len=4.0, gap_len=4.0)
        draw_text_centered(draw, fonts, gx, 665.0, f"{pct}%", fonts.get("mono_regular", 10.5), TEXT_MUTED_RGB)

    # 6 Metric Rows
    rows = [
        (
            "Hallucination / Wrong",
            f"n={na.hallucination_den} · lower is better",
            na.hallucination_pct,
            v2a.hallucination_pct,
            f"{hal_rel:.0f}% ({hal_pp:+.1f} pp)",
        ),
        (
            "Precision Among Answered",
            f"{na.precision_num}/{na.precision_den} vs {v2a.precision_num}/{v2a.precision_den}",
            na.precision_pct,
            v2a.precision_pct,
            f"{prec_pp:+.1f} pp",
        ),
        (
            "Cross-Doc 2/2 Hit Rate",
            f"n={v2j.cross_doc_total} · 2 statutes",
            nj.cross_doc_hit_pct,
            v2j.cross_doc_hit_pct,
            f"{xdoc_pp:+.1f} pp",
        ),
        (
            "Target Doc Hit Rate",
            f"n={v2j.n_in_scope} in-scope",
            nj.target_doc_hit_pct,
            v2j.target_doc_hit_pct,
            f"{hit_pp:+.1f} pp",
        ),
        (
            "Context Source Purity",
            f"n={v2j.n_in_scope} in-scope",
            nj.purity_pct,
            v2j.purity_pct,
            f"{pur_pp:+.1f} pp",
        ),
        (
            "Structural OOD Cutoff",
            f"n={v2j.ood_total} out-of-domain",
            nj.ood_cutoff_pct,
            v2j.ood_cutoff_pct,
            f"{ood_pp:+.1f} pp",
        ),
    ]

    row_start_cy = 289.0
    row_step = 65.0
    bar_h = 15.0
    bar_gap = 4.0

    for i, (m_title, m_sub, val_n, val_v2, badge_txt) in enumerate(rows):
        cy = row_start_cy + i * row_step
        if i > 0:
            sep_y = cy - row_step / 2.0
            draw.line((s(56), s(sep_y), s(1344), s(sep_y)), fill=GRIDLINE_RGB, width=int(round(s(1.0))))

        # Left label + subtitle
        draw_text_left_vcenter(draw, fonts, 58.0, cy - 9.0, m_title, fonts.get("sans_bold", 13.5), TEXT_PRIMARY_RGB)
        draw_text_left_vcenter(draw, fonts, 58.0, cy + 10.5, m_sub, fonts.get("mono_regular", 10.5), TEXT_SECONDARY_RGB)

        # Paired horizontal bars (Naive top, Agentic v2 bottom)
        bar_specs = [
            (cy - bar_gap / 2.0 - bar_h, cy - bar_gap / 2.0, val_n, NAIVE_BAR_RGB, NAIVE_VAL_RGB),
            (cy + bar_gap / 2.0, cy + bar_gap / 2.0 + bar_h, val_v2, V2_BAR_RGB, V2_VAL_RGB),
        ]
        for by1, by2, pct_val, b_col, v_col in bar_specs:
            bar_len = (pct_val / 100.0) * track_w
            bar_cy = (by1 + by2) / 2.0
            if bar_len < 2.0:
                # Draw a crisp 3px baseline tick for 0.0% so the zero value stays visually anchored
                draw.rounded_rectangle(
                    (s(track_x0), s(by1), s(track_x0 + 3.0), s(by2)),
                    radius=s(1.5),
                    fill=b_col,
                )
                label_x = track_x0 + 10.0
            else:
                draw.rounded_rectangle(
                    (s(track_x0), s(by1), s(track_x0 + bar_len), s(by2)),
                    radius=s(min(3.5, bar_len / 2.0)),
                    fill=b_col,
                    corners=(False, True, True, False),
                )
                label_x = track_x0 + bar_len + 9.0
            draw_text_left_vcenter(draw, fonts, label_x, bar_cy, f"{pct_val:.1f}%", fonts.get("mono_bold", 11.3), v_col)

        # Centered emerald delta badge pill in dedicated right column (cx = 1262)
        b_font = fonts.get("mono_bold", 11.0)
        draw_pill(
            draw,
            fonts,
            1262.0,
            cy,
            badge_txt,
            b_font,
            POS_TEXT_RGB,
            POS_BG_RGB,
            POS_BORDER_RGB,
            pad_x=10.5,
            pad_y=5.0,
            radius=9.0,
        )

    # 3. Footer Strip (x = 36..1364, y = 702..754)
    draw_panel_with_top_accent(
        canvas,
        (36.0, 702.0, 1364.0, 754.0),
        radius=10.0,
        fill_color=SUBCARD_BG_RGB,
        border_color=PANEL_BORDER_RGB,
    )
    draw = ImageDraw.Draw(canvas)
    footer_cy = 728.0
    tradeoff_detail = (
        f"Useful Answers matched ({v2a.useful_num}/{v2a.useful_den} = {v2a.useful_pct:.1f}%)  ·  "
        f"Head-to-Head {bundle.head_to_head}  ·  "
        f"Mean Latency {nj.mean_latency_sec:.1f}s vs {v2j.mean_latency_sec:.1f}s"
    )
    draw_inline_runs(
        draw,
        fonts,
        56.0,
        footer_cy,
        [
            ("Trade-off: ", fonts.get("sans_bold", 11.6), TEXT_PRIMARY_RGB),
            (tradeoff_detail, fonts.get("sans_regular", 11.6), TEXT_SECONDARY_RGB),
        ],
    )

    v3_pill = f"v3 Ablation (n={v3j.n_in_scope}): {v3j.target_doc_hit_pct:.1f}% Hit · {v3j.purity_pct:.1f}% Purity"
    v3_font = fonts.get("mono_bold", 10.8)
    v3_w, _ = measure_text(draw, v3_pill, v3_font)
    draw_pill(
        draw,
        fonts,
        1346.0 - (v3_w / 2.0 + 10.5),
        footer_cy,
        v3_pill,
        v3_font,
        V2_VAL_RGB,
        INFO_BG_RGB,
        INFO_BORDER_RGB,
        pad_x=10.5,
        pad_y=5.0,
        radius=8.0,
    )

    downsampled = canvas.resize((LOGICAL_W, LOGICAL_H), resample=Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downsampled.save(output_path, format="PNG", optimize=True)


# ---------------------------------------------------------------------------
# Chart 2: Independent 32B RAGAS Comparison (1400 x 780)
# ---------------------------------------------------------------------------
def _draw_ragas_panel(
    canvas: Image.Image,
    fonts: FontSet,
    panel_box: Tuple[float, float, float, float],
    accent_color: Tuple[int, int, int],
    title: str,
    subtitle_runs: Sequence[Tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, Tuple[int, int, int]]],
    naive_vals: Sequence[float],
    v2_vals: Sequence[float],
    positive_deltas: bool,
    bottom_note: str,
) -> None:
    x0, y0, x1, y1 = panel_box
    draw_panel_with_top_accent(
        canvas,
        panel_box,
        radius=10.0,
        fill_color=PANEL_BG_RGB,
        border_color=PANEL_BORDER_RGB,
        accent_color=accent_color,
        accent_h=3.0,
    )
    draw = ImageDraw.Draw(canvas)

    draw_text_left_vcenter(draw, fonts, x0 + 22.0, y0 + 27.0, title, fonts.get("sans_bold", 15.5), TEXT_PRIMARY_RGB)
    draw_inline_runs(draw, fonts, x0 + 22.0, y0 + 50.0, subtitle_runs)

    # Plot coordinates: aligned delta pill row at y=190 cleanly above y=1.00 gridline
    plot_left = x0 + 62.0
    plot_right = x1 - 22.0
    plot_w = plot_right - plot_left  # 566px
    y_pill_row = y0 + 86.0  # 190.0
    y_100 = y0 + 116.0  # 220.0
    y_000 = y0 + 424.0  # 528.0
    plot_h = y_000 - y_100  # 308.0px

    # Horizontal gridlines (0.00 to 1.00)
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        gy = y_000 - tick * plot_h
        if tick == 0.0:
            draw.line((s(plot_left), s(gy), s(plot_right), s(gy)), fill=PANEL_BORDER_RGB, width=int(round(s(1.2))))
        else:
            draw_dashed_line(draw, (plot_left, gy), (plot_right, gy), GRIDLINE_RGB, width_logical=1.0, dash_len=4.0, gap_len=4.0)
        draw_text_right_vcenter(draw, fonts, plot_left - 10.0, gy, f"{tick:.2f}", fonts.get("mono_regular", 10.2), TEXT_MUTED_RGB)

    # 5 Grouped Metric Pairs
    metric_labels = (
        ("Faithfulness", ""),
        ("Context", "Recall"),
        ("Answer", "Relevancy"),
        ("Context", "Precision"),
        ("Answer", "Correctness"),
    )
    group_pitch = plot_w / len(metric_labels)  # 113.2px
    bar_w = 38.0
    pair_gap = 6.0  # center-to-center bar distance = 44.0px

    badge_fg = POS_TEXT_RGB if positive_deltas else NEG_TEXT_RGB
    badge_bg = POS_BG_RGB if positive_deltas else NEG_BG_RGB
    badge_border = POS_BORDER_RGB if positive_deltas else NEG_BORDER_RGB

    for idx, (lbl1, lbl2) in enumerate(metric_labels):
        gx = plot_left + (idx + 0.5) * group_pitch
        vn = naive_vals[idx]
        vv = v2_vals[idx]
        delta = vv - vn

        bars = [
            (gx - pair_gap / 2.0 - bar_w, gx - pair_gap / 2.0, vn, NAIVE_BAR_RGB, NAIVE_VAL_RGB),
            (gx + pair_gap / 2.0, gx + pair_gap / 2.0 + bar_w, vv, V2_BAR_RGB, V2_VAL_RGB),
        ]
        top_ys: List[float] = []
        for bx1, bx2, val, b_col, t_col in bars:
            bh = val * plot_h
            by_top = y_000 - bh
            top_ys.append(by_top)
            draw.rounded_rectangle(
                (s(bx1), s(by_top), s(bx2), s(y_000)),
                radius=s(min(4.0, bh / 2.0)),
                fill=b_col,
                corners=(True, True, False, False),
            )
            draw_text_centered(draw, fonts, (bx1 + bx2) / 2.0, by_top - 11.0, f"{val:.3f}", fonts.get("mono_bold", 10.3), t_col)

        # Delta badge pill in clean uniform row directly above each bar pair (zero gridline collision)
        draw_pill(
            draw,
            fonts,
            gx,
            y_pill_row,
            f"{delta:+.3f}",
            fonts.get("mono_bold", 10.4),
            badge_fg,
            badge_bg,
            badge_border,
            pad_x=8.0,
            pad_y=3.5,
            radius=8.5,
        )

        # Centered X-axis labels
        if lbl2:
            draw_text_centered(draw, fonts, gx, y_000 + 18.0, lbl1, fonts.get("sans_medium", 11.4), TEXT_PRIMARY_RGB)
            draw_text_centered(draw, fonts, gx, y_000 + 34.0, lbl2, fonts.get("sans_medium", 11.4), TEXT_PRIMARY_RGB)
        else:
            draw_text_centered(draw, fonts, gx, y_000 + 25.0, lbl1, fonts.get("sans_medium", 11.4), TEXT_PRIMARY_RGB)

    # Bottom mini-callout box inside panel
    sub_y1, sub_y2 = y1 - 60.0, y1 - 16.0
    draw.rounded_rectangle(
        (s(x0 + 18.0), s(sub_y1), s(x1 - 18.0), s(sub_y2)),
        radius=s(7.5),
        fill=SUBCARD_BG_RGB,
        outline=SUBCARD_BORDER_RGB,
        width=int(round(s(1.0))),
    )
    draw_text_centered(draw, fonts, (x0 + x1) / 2.0, (sub_y1 + sub_y2) / 2.0, bottom_note, fonts.get("mono_regular", 10.1), TEXT_SECONDARY_RGB)


def render_ragas_chart(bundle: BenchmarkBundle, output_path: Path, fonts: FontSet) -> None:
    """Render assets/ragas_comparison.png at 2x supersampling -> 1400x780."""
    rg = bundle.ragas
    v2a = bundle.v2_audit

    canvas = create_dark_base_canvas(
        fonts,
        title="Independent 32B RAGAS Evaluation (qwen2.5:32b + bge-m3)",
        subtitle=f"Positive answers vs. raw aggregate fallback (0.0000) distortion across {bundle.v2_json.n_in_scope} in-scope queries · Zero NaNs",
    )

    # Left Panel: Positive Answers Only
    panel_a_note = (
        f"Common Positive Subset (n={rg.common_n}): "
        f"Faithfulness {rg.common_naive[0]:.3f} -> {rg.common_v2[0]:.3f} · "
        f"Relevancy {rg.common_naive[2]:.3f} -> {rg.common_v2[2]:.3f}"
    )
    _draw_ragas_panel(
        canvas=canvas,
        fonts=fonts,
        panel_box=(36.0, 104.0, 686.0, 648.0),
        accent_color=POS_ACCENT_RGB,
        title="A. Positive Answers Only (Core Quality)",
        subtitle_runs=[
            (f"Naive (n={rg.pos_naive_n}) vs. Agentic v2 (n={rg.pos_v2_n})  ·  ", fonts.get("sans_regular", 11.4), TEXT_SECONDARY_RGB),
            ("Agentic v2 wins all 5 metrics", fonts.get("sans_medium", 11.4), POS_TEXT_RGB),
        ],
        naive_vals=rg.pos_naive,
        v2_vals=rg.pos_v2,
        positive_deltas=True,
        bottom_note=panel_a_note,
    )

    # Right Panel: Raw All Rows (Fallback 0.0000 Distortion)
    faith_jump = rg.v2_non_fb[0] - rg.raw_v2[0]
    panel_b_note = (
        f"Non-Fallback v2 (n={rg.v2_non_fb_n}): "
        f"Faithfulness jumps {rg.raw_v2[0]:.3f} -> {rg.v2_non_fb[0]:.3f} ({faith_jump:+.3f}) · "
        f"Relevancy {rg.raw_v2[2]:.3f} -> {rg.v2_non_fb[2]:.3f}"
    )
    _draw_ragas_panel(
        canvas=canvas,
        fonts=fonts,
        panel_box=(714.0, 104.0, 1364.0, 648.0),
        accent_color=NEG_ACCENT_RGB,
        title="B. Raw All Rows (Fallback 0.0000 Effect)",
        subtitle_runs=[
            (f"Naive (n={rg.raw_naive_n}) vs. Agentic v2 (n={rg.raw_v2_n}, incl. ", fonts.get("sans_regular", 11.4), TEXT_SECONDARY_RGB),
            (f"{rg.v2_fallback_csv_n} fallbacks scored 0.0000", fonts.get("sans_medium", 11.4), NEG_TEXT_RGB),
            (")", fonts.get("sans_regular", 11.4), TEXT_SECONDARY_RGB),
        ],
        naive_vals=rg.raw_naive,
        v2_vals=rg.raw_v2,
        positive_deltas=False,
        bottom_note=panel_b_note,
    )

    # Bottom Methodological Callout Bar (x = 36..1364, y = 668..754)
    draw_panel_with_top_accent(
        canvas,
        (36.0, 668.0, 1364.0, 754.0),
        radius=10.0,
        fill_color=PANEL_BG_RGB,
        border_color=PANEL_BORDER_RGB,
    )
    draw = ImageDraw.Draw(canvas)

    # Left vertical cyan bar accent inside card
    draw.rounded_rectangle(
        (s(48.0), s(682.0), s(52.0), s(740.0)),
        radius=s(2.0),
        fill=V2_BAR_RGB,
    )

    tag_text = "METHODOLOGICAL FINDING"
    tag_font = fonts.get("mono_bold", 9.8)
    tag_w, _ = measure_text(draw, tag_text, tag_font)
    tag_cx = 64.0 + tag_w / 2.0 + 8.5
    draw_pill(
        draw,
        fonts,
        tag_cx,
        695.0,
        tag_text,
        tag_font,
        V2_VAL_RGB,
        INFO_BG_RGB,
        INFO_BORDER_RGB,
        pad_x=8.5,
        pad_y=3.5,
        radius=6.0,
    )

    line1_x = 64.0 + tag_w + 17.0 + 12.0
    draw_inline_runs(
        draw,
        fonts,
        line1_x,
        695.0,
        [
            ("Deterministic ", fonts.get("sans_regular", 11.8), TEXT_PRIMARY_RGB),
            ("fallback_node", fonts.get("mono_bold", 11.2), V2_VAL_RGB),
            (
                f" abstentions score 0.0000 Faithfulness & 0.0000 Answer Relevancy across all {rg.v2_fallback_csv_n} fallback rows in raw RAGAS",
                fonts.get("sans_regular", 11.8),
                TEXT_PRIMARY_RGB,
            ),
        ],
    )
    draw_inline_runs(
        draw,
        fonts,
        64.0,
        727.0,
        [
            (
                "because meta-refusal wording is absent from source PDFs. Filtering abstentions reveals ",
                fonts.get("sans_regular", 11.6),
                TEXT_SECONDARY_RGB,
            ),
            (f"{rg.pos_v2[0]:.4f} Faithfulness", fonts.get("mono_bold", 11.2), POS_TEXT_RGB),
            (" — closely corroborating the ", fonts.get("sans_regular", 11.6), TEXT_SECONDARY_RGB),
            (f"{v2a.precision_pct:.1f}%", fonts.get("mono_bold", 11.2), V2_VAL_RGB),
            (" manual precision audit.", fonts.get("sans_regular", 11.6), TEXT_SECONDARY_RGB),
        ],
    )

    downsampled = canvas.resize((LOGICAL_W, LOGICAL_H), resample=Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downsampled.save(output_path, format="PNG", optimize=True)


# ---------------------------------------------------------------------------
# Stdout Verification Report
# ---------------------------------------------------------------------------
def print_verification_summary(bundle: BenchmarkBundle) -> None:
    """Print an auditable metric verification table to stdout."""
    nj, v2j, v3j = bundle.naive_json, bundle.v2_json, bundle.v3_json
    na, v2a = bundle.naive_audit, bundle.v2_audit
    rg = bundle.ragas

    print("=" * 84)
    print("VERIFIED BENCHMARK METRICS (Computed Dynamically from Raw JSON / CSV Artifacts)")
    print("=" * 84)
    print(f"{'Metric':<34} | {'Naive RAG':<20} | {'Agentic v2':<22}")
    print("-" * 84)
    print(f"{'Target Doc Hit (in-scope)':<34} | {nj.target_doc_hits}/{nj.n_in_scope} ({nj.target_doc_hit_pct:6.2f}%)   | {v2j.target_doc_hits}/{v2j.n_in_scope} ({v2j.target_doc_hit_pct:6.2f}%)")
    print(f"{'Cross-Doc 2/2 Hit':<34} | {nj.cross_doc_hits}/{nj.cross_doc_total} ({nj.cross_doc_hit_pct:6.2f}%)    | {v2j.cross_doc_hits}/{v2j.cross_doc_total} ({v2j.cross_doc_hit_pct:6.2f}%)")
    print(f"{'Context Source Purity':<34} | {nj.purity_pct:6.2f}%              | {v2j.purity_pct:6.2f}%")
    print(f"{'Structural OOD Cutoff':<34} | {nj.ood_cutoffs}/{nj.ood_total} ({nj.ood_cutoff_pct:5.1f}%)      | {v2j.ood_cutoffs}/{v2j.ood_total} ({v2j.ood_cutoff_pct:5.1f}%)")
    print(f"{'In-Scope Fallback Count':<34} | {nj.in_scope_fallback:<20} | {v2j.in_scope_fallback:<22}")
    print(f"{'Mean Latency (all records)':<34} | {nj.mean_latency_sec:5.2f}s (n={nj.n_total})      | {v2j.mean_latency_sec:5.2f}s (n={v2j.n_total})")
    print(f"{'Hallucination / Wrong (Audit)':<34} | {na.hallucination_num}/{na.hallucination_den} ({na.hallucination_pct:5.2f}% -> {na.hallucination_pct:.1f}%) | {v2a.hallucination_num}/{v2a.hallucination_den} ({v2a.hallucination_pct:5.2f}% -> {v2a.hallucination_pct:.1f}%)")
    print(f"{'Precision Among Answered (Audit)':<34} | {na.precision_num}/{na.precision_den} ({na.precision_pct:5.2f}% -> {na.precision_pct:.1f}%) | {v2a.precision_num}/{v2a.precision_den} ({v2a.precision_pct:5.2f}% -> {v2a.precision_pct:.1f}%)")
    print(f"{'Total Useful (Audit)':<34} | {na.useful_num}/{na.useful_den} ({na.useful_pct:5.1f}%)       | {v2a.useful_num}/{v2a.useful_den} ({v2a.useful_pct:5.1f}%)")
    print(f"{'v3 Ablation (n=25 in-scope)':<34} | {'--':<20} | {v3j.target_doc_hits}/{v3j.n_in_scope} ({v3j.target_doc_hit_pct:.2f}%), Purity {v3j.purity_pct:.2f}%, Lat {v3j.mean_latency_sec:.2f}s")
    print("-" * 84)
    print("32B RAGAS EVALUATION SLICES [Faithfulness, Context Recall, Answer Relevancy, Context Precision, Answer Correctness]")
    print(f"  Raw Naive         (n={rg.raw_naive_n}): " + ", ".join(f"{v:.4f}" for v in rg.raw_naive))
    print(f"  Raw Agentic v2    (n={rg.raw_v2_n}): " + ", ".join(f"{v:.4f}" for v in rg.raw_v2) + f"  (incl. {rg.v2_fallback_csv_n} fallback rows @ 0.0000)")
    print(f"  Non-Fallback v2   (n={rg.v2_non_fb_n}): " + ", ".join(f"{v:.4f}" for v in rg.v2_non_fb))
    print(f"  Positive Naive    (n={rg.pos_naive_n}): " + ", ".join(f"{v:.4f}" for v in rg.pos_naive))
    print(f"  Positive v2       (n={rg.pos_v2_n}): " + ", ".join(f"{v:.4f}" for v in rg.pos_v2))
    print(f"  Common Pos Naive  (n={rg.common_n}): " + ", ".join(f"{v:.4f}" for v in rg.common_naive))
    print(f"  Common Pos v2     (n={rg.common_n}): " + ", ".join(f"{v:.4f}" for v in rg.common_v2))
    print("=" * 84)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate dark-card benchmark PNG charts.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root,
        help="Directory containing the 5 raw evaluation JSON/CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "assets",
        help="Directory where PNG charts will be written.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    t0 = time.perf_counter()
    args = parse_args(argv)
    bundle = load_and_verify_all(args.data_dir)
    print_verification_summary(bundle)

    fonts = FontSet()
    chart1_path = args.output_dir / "benchmark_comparison.png"
    chart2_path = args.output_dir / "ragas_comparison.png"

    render_benchmark_chart(bundle, chart1_path, fonts)
    render_ragas_chart(bundle, chart2_path, fonts)

    elapsed = time.perf_counter() - t0
    for p in (chart1_path, chart2_path):
        with Image.open(p) as im:
            assert im.size == (LOGICAL_W, LOGICAL_H), f"Unexpected dimensions for {p}: {im.size}"
        print(f"Rendered {p} ({LOGICAL_W}x{LOGICAL_H}, {p.stat().st_size:,} bytes)")
    print(f"Done in {elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
