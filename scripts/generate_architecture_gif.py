#!/usr/bin/env python3
"""Reproducible generator for the Agentic RAG LangGraph architecture animation.

Renders 120 frames at 2x supersampling (2400x1240 RGB -> 1200x620 via LANCZOS),
builds a 3x2 QA contact sheet, and encodes an 8.0s 15fps looping GIF (< 2 MB)
matching src/graph.py, src/edges/routing.py, and config/settings.py.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ---------------------------------------------------------------------------
# Paths & Output Parameters
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
GIF_OUTPUT_PATH = REPO_ROOT / "assets" / "architecture_flow.gif"
FRAMES_DIR = Path("/tmp/arch_gif_frames")
PALETTE_PATH = Path("/tmp/arch_gif_palette.png")
CONTACT_SHEET_PATH = Path("/tmp/arch_gif_contact.png")
FFMPEG_BIN = Path("/opt/homebrew/bin/ffmpeg")
FFPROBE_BIN = Path("/opt/homebrew/bin/ffprobe")

LOGICAL_W, LOGICAL_H = 1200, 620
SCALE = 2
SUP_W, SUP_H = LOGICAL_W * SCALE, LOGICAL_H * SCALE
FPS = 15
TOTAL_FRAMES = 120  # 8.0s @ 15fps
CONTACT_FRAME_INDICES = [10, 40, 62, 78, 98, 114]


# ---------------------------------------------------------------------------
# Palette & Theme Colors (#0d1117 GitHub/Chainlit Dark Theme)
# ---------------------------------------------------------------------------
BG_HEX = "#0d1117"
CARD_BORDER_HEX = "#21262d"
DOT_GRID_HEX = "#1a2230"
NODE_BG_HEX = "#161b22"
PILL_BG_HEX = "#131922"
NODE_BORDER_IDLE_HEX = "#30363d"
EDGE_IDLE_HEX = "#2d3542"
REFLECT_EDGE_IDLE_HEX = "#29313d"

TEXT_PRIMARY_HEX = "#f0f6fc"
TEXT_TITLE_IDLE_HEX = "#e6edf3"
TEXT_SUB_HEX = "#8b949e"
TEXT_MUTED_HEX = "#6e7681"

CYAN_HEX = "#38bdf8"
EMERALD_HEX = "#22c55e"
ORANGE_HEX = "#fb923c"
AMBER_HEX = "#f59e0b"
RED_HEX = "#ef4444"


def hex_to_rgb(hx: str) -> Tuple[int, int, int]:
    hx = hx.lstrip("#")
    return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))


def lerp_color(
    c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float
) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(round(c1[0] + (c2[0] - c1[0]) * t)),
        int(round(c1[1] + (c2[1] - c1[1]) * t)),
        int(round(c1[2] + (c2[2] - c1[2]) * t)),
    )


BG_RGB = hex_to_rgb(BG_HEX)
CARD_BORDER_RGB = hex_to_rgb(CARD_BORDER_HEX)
DOT_GRID_RGB = hex_to_rgb(DOT_GRID_HEX)
NODE_BG_RGB = hex_to_rgb(NODE_BG_HEX)
PILL_BG_RGB = hex_to_rgb(PILL_BG_HEX)
NODE_BORDER_IDLE_RGB = hex_to_rgb(NODE_BORDER_IDLE_HEX)
EDGE_IDLE_RGB = hex_to_rgb(EDGE_IDLE_HEX)
REFLECT_EDGE_IDLE_RGB = hex_to_rgb(REFLECT_EDGE_IDLE_HEX)

TEXT_PRIMARY_RGB = hex_to_rgb(TEXT_PRIMARY_HEX)
TEXT_TITLE_IDLE_RGB = hex_to_rgb(TEXT_TITLE_IDLE_HEX)
TEXT_SUB_RGB = hex_to_rgb(TEXT_SUB_HEX)
TEXT_MUTED_RGB = hex_to_rgb(TEXT_MUTED_HEX)

CYAN_RGB = hex_to_rgb(CYAN_HEX)
EMERALD_RGB = hex_to_rgb(EMERALD_HEX)
ORANGE_RGB = hex_to_rgb(ORANGE_HEX)
AMBER_RGB = hex_to_rgb(AMBER_HEX)
RED_RGB = hex_to_rgb(RED_HEX)


# ---------------------------------------------------------------------------
# Font Loading (macOS System Fonts with Safe Fallback)
# ---------------------------------------------------------------------------
class FontSet:
    """Loaded 2x-scaled font dictionary matching macOS system fonts."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

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


# ---------------------------------------------------------------------------
# Graph Geometry Definitions (Logical 1200x620 Coordinates)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeSpec:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    title: str
    sub1: str = ""
    sub2: str = ""
    is_pill: bool = False
    is_muted: bool = False
    radius: int = 10

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class EdgeLabelSpec:
    text: str
    cx: float
    cy: float
    font_role: str = "sans_medium"
    font_size: float = 8.5
    pad_x: float = 5.0
    pad_y: float = 2.5


@dataclass(frozen=True)
class EdgeSpec:
    id: str
    waypoints: Tuple[Tuple[float, float], ...]
    label: Optional[EdgeLabelSpec] = None
    corner_radius: float = 10.0
    is_muted: bool = False


NODES: Dict[str, NodeSpec] = {
    "query": NodeSpec(
        id="query",
        x1=28, y1=277, x2=120, y2=333,
        title="User Query",
        is_pill=True, radius=14,
    ),
    "decompose": NodeSpec(
        id="decompose",
        x1=152, y1=266, x2=286, y2=344,
        title="Decompose",
        sub1="qwen2.5:7b",
        sub2="<= 3 sub-queries",
    ),
    "retrieve": NodeSpec(
        id="retrieve",
        x1=320, y1=266, x2=454, y2=344,
        title="Retrieve",
        sub1="Qdrant Top-50",
        sub2="nomic-embed",
    ),
    "grade_ret": NodeSpec(
        id="grade_ret",
        x1=488, y1=266, x2=628, y2=344,
        title="Grade Docs",
        sub1="FlashRank",
        sub2="top-8 · th>=0.03",
    ),
    "generate": NodeSpec(
        id="generate",
        x1=678, y1=266, x2=812, y2=344,
        title="Generate",
        sub1="qwen2.5:7b",
        sub2="grounded draft",
    ),
    "grade_hal": NodeSpec(
        id="grade_hal",
        x1=852, y1=266, x2=1002, y2=344,
        title="Verify NLI",
        sub1="ModernBERT-NLI",
        sub2="entail>=0.40",
    ),
    "verified": NodeSpec(
        id="verified",
        x1=1054, y1=275, x2=1170, y2=335,
        title="Verified",
        sub1="END",
        is_pill=True, radius=14,
    ),
    "transform": NodeSpec(
        id="transform",
        x1=408, y1=170, x2=546, y2=226,
        title="Transform Query",
        sub1="qwen2.5:3b (SLM)",
        radius=9,
    ),
    "reflect": NodeSpec(
        id="reflect",
        x1=668, y1=92, x2=852, y2=148,
        title="Reflect Missing",
        sub1="qwen2.5:7b (reflect < 1)",
        is_muted=True, radius=9,
    ),
    "fallback": NodeSpec(
        id="fallback",
        x1=600, y1=458, x2=860, y2=532,
        title="Deterministic Fallback",
        sub1="fallback_node · safe abstention",
        sub2="0 docs / retries exhausted",
        radius=10,
    ),
    "safe_end": NodeSpec(
        id="safe_end",
        x1=1054, y1=466, x2=1170, y2=524,
        title="Safe End",
        sub1="Abstention",
        is_pill=True, radius=14,
    ),
}


EDGES: Dict[str, EdgeSpec] = {
    "query->decompose": EdgeSpec(
        id="query->decompose",
        waypoints=((120, 305), (152, 305)),
    ),
    "decompose->retrieve": EdgeSpec(
        id="decompose->retrieve",
        waypoints=((286, 305), (320, 305)),
    ),
    "retrieve->grade_ret": EdgeSpec(
        id="retrieve->grade_ret",
        waypoints=((454, 305), (488, 305)),
    ),
    "grade_ret->generate": EdgeSpec(
        id="grade_ret->generate",
        waypoints=((628, 305), (678, 305)),
        label=EdgeLabelSpec(
            text="docs >= 1",
            cx=653, cy=282,
            font_role="sans_medium", font_size=7.0, pad_x=2.4, pad_y=1.8,
        ),
    ),
    "generate->grade_hal": EdgeSpec(
        id="generate->grade_hal",
        waypoints=((812, 305), (852, 305)),
    ),
    "grade_hal->verified": EdgeSpec(
        id="grade_hal->verified",
        waypoints=((1002, 305), (1054, 305)),
        label=EdgeLabelSpec(
            text="Grounded",
            cx=1028, cy=282,
            font_role="sans_medium", font_size=7.0, pad_x=2.4, pad_y=1.8,
        ),
    ),
    "grade_ret->transform": EdgeSpec(
        id="grade_ret->transform",
        waypoints=((576, 266), (576, 198), (546, 198)),
        label=EdgeLabelSpec(
            text="0 docs (< 1)",
            cx=619, cy=232,
            font_role="sans_medium", font_size=8.8, pad_x=6.0, pad_y=2.8,
        ),
    ),
    "transform->retrieve": EdgeSpec(
        id="transform->retrieve",
        waypoints=((408, 198), (374, 198), (374, 266)),
    ),
    "grade_hal->reflect": EdgeSpec(
        id="grade_hal->reflect",
        waypoints=((927, 266), (927, 120), (852, 120)),
        label=EdgeLabelSpec(
            text="missing context (< 1)",
            cx=924, cy=97,
            font_role="sans_medium", font_size=8.8, pad_x=6.5, pad_y=2.8,
        ),
        is_muted=True,
    ),
    "reflect->retrieve": EdgeSpec(
        id="reflect->retrieve",
        waypoints=((668, 120), (352, 120), (352, 266)),
        is_muted=True,
    ),
    "grade_hal->generate": EdgeSpec(
        id="grade_hal->generate",
        waypoints=((895, 344), (895, 398), (745, 398), (745, 344)),
        label=EdgeLabelSpec(
            text="NLI Flag · retry < 2",
            cx=820, cy=398,
            font_role="sans_medium", font_size=8.8, pad_x=6.5, pad_y=2.8,
        ),
    ),
    "grade_ret->fallback": EdgeSpec(
        id="grade_ret->fallback",
        waypoints=((558, 344), (558, 495), (600, 495)),
        label=EdgeLabelSpec(
            text="0 docs (retry max)",
            cx=502, cy=424,
            font_role="sans_medium", font_size=8.8, pad_x=6.5, pad_y=2.8,
        ),
    ),
    "grade_hal->fallback": EdgeSpec(
        id="grade_hal->fallback",
        waypoints=((955, 344), (955, 438), (825, 438), (825, 458)),
        label=EdgeLabelSpec(
            text="NLI Flag (retry max)",
            cx=1016, cy=394,
            font_role="sans_medium", font_size=8.8, pad_x=6.5, pad_y=2.8,
        ),
    ),
    "fallback->safe_end": EdgeSpec(
        id="fallback->safe_end",
        waypoints=((860, 495), (1054, 495)),
        label=EdgeLabelSpec(
            text="Abstain",
            cx=957, cy=495,
            font_role="sans_medium", font_size=8.8, pad_x=7.0, pad_y=2.8,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Rounded Polyline Geometry & Sampling
# ---------------------------------------------------------------------------
def build_rounded_polyline(
    waypoints: Sequence[Tuple[float, float]],
    radius: float = 10.0,
    arc_steps: int = 12,
    trim_end: float = 4.0,
) -> Tuple[List[Tuple[float, float]], List[float], Tuple[float, float], Tuple[float, float]]:
    """Convert orthogonal waypoints into densely sampled rounded-corner points.

    Returns:
        pts: sampled polyline points (shortened at tip by trim_end for clean arrowhead)
        cum_lens: cumulative arc lengths along pts
        tip_pos: exact endpoint of original last waypoint (for arrowhead tip)
        tip_dir: unit direction vector entering tip_pos
    """
    raw: List[Tuple[float, float]] = [tuple(p) for p in waypoints]
    tip_pos = raw[-1]
    prev_p = raw[-2]
    dx, dy = tip_pos[0] - prev_p[0], tip_pos[1] - prev_p[1]
    seg_len = math.hypot(dx, dy)
    tip_dir = (dx / seg_len, dy / seg_len)

    # Shorten final point slightly so thick line stroke doesn't blunt arrowhead tip
    if trim_end > 0 and seg_len > trim_end + 1.0:
        raw[-1] = (tip_pos[0] - tip_dir[0] * trim_end, tip_pos[1] - tip_dir[1] * trim_end)

    pts: List[Tuple[float, float]] = [raw[0]]
    for i in range(1, len(raw) - 1):
        p_prev = raw[i - 1]
        p_curr = raw[i]
        p_next = raw[i + 1]

        v_in = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v_out = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        l_in = math.hypot(*v_in)
        l_out = math.hypot(*v_out)
        if l_in < 1e-6 or l_out < 1e-6:
            pts.append(p_curr)
            continue

        u_in = (v_in[0] / l_in, v_in[1] / l_in)
        u_out = (v_out[0] / l_out, v_out[1] / l_out)
        d = min(radius, 0.45 * l_in, 0.45 * l_out)

        p_a = (p_curr[0] - u_in[0] * d, p_curr[1] - u_in[1] * d)
        p_b = (p_curr[0] + u_out[0] * d, p_curr[1] + u_out[1] * d)
        pts.append(p_a)

        # Quadratic Bezier corner fillet P_a -> P_curr -> P_b matches orthogonal 90° join cleanly
        for s in range(1, arc_steps + 1):
            t = s / arc_steps
            mt = 1.0 - t
            qx = mt * mt * p_a[0] + 2 * mt * t * p_curr[0] + t * t * p_b[0]
            qy = mt * mt * p_a[1] + 2 * mt * t * p_curr[1] + t * t * p_b[1]
            pts.append((qx, qy))

    pts.append(raw[-1])

    cum_lens = [0.0]
    for i in range(1, len(pts)):
        cum_lens.append(cum_lens[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return pts, cum_lens, tip_pos, tip_dir


EDGE_GEOM: Dict[str, Tuple[List[Tuple[float, float]], List[float], Tuple[float, float], Tuple[float, float]]] = {
    eid: build_rounded_polyline(spec.waypoints, radius=spec.corner_radius, trim_end=4.5)
    for eid, spec in EDGES.items()
}


def sample_edge_at_fraction(eid: str, frac: float) -> Tuple[float, float]:
    """Return (x, y) at fraction frac in [0, 1] along full edge (up to arrowhead)."""
    pts, cum, tip_pos, _ = EDGE_GEOM[eid]
    frac = max(0.0, min(1.0, frac))
    total = cum[-1]
    if total <= 1e-6:
        return tip_pos
    target = frac * total
    for i in range(1, len(cum)):
        if cum[i] >= target:
            seg = cum[i] - cum[i - 1]
            u = 0.0 if seg < 1e-6 else (target - cum[i - 1]) / seg
            return (
                pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * u,
                pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * u,
            )
    return pts[-1]


def slice_edge_to_fraction(eid: str, frac: float) -> List[Tuple[float, float]]:
    """Return partial polyline points from start up to frac in [0, 1]."""
    pts, cum, _, _ = EDGE_GEOM[eid]
    frac = max(0.0, min(1.0, frac))
    if frac <= 0.0:
        return []
    if frac >= 1.0:
        return list(pts)
    target = frac * cum[-1]
    out = [pts[0]]
    for i in range(1, len(cum)):
        if cum[i] <= target:
            out.append(pts[i])
        else:
            seg = cum[i] - cum[i - 1]
            u = 0.0 if seg < 1e-6 else (target - cum[i - 1]) / seg
            out.append((
                pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * u,
                pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * u,
            ))
            break
    return out


# ---------------------------------------------------------------------------
# Drawing Primitives (2x Supersampled Coordinates)
# ---------------------------------------------------------------------------
def s(v: float) -> float:
    return v * SCALE


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    """Draw text centered optically around logical (cx, cy) using bbox metrics."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = s(cx) - tw / 2.0 - bbox[0]
    y = s(cy) - th / 2.0 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def draw_left_text_vcenter(
    draw: ImageDraw.ImageDraw,
    x_left: float,
    cy: float,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> float:
    """Draw left-aligned text vertically centered on logical cy; returns logical right x."""
    bbox = draw.textbbox((0, 0), text, font=font)
    th = bbox[3] - bbox[1]
    x = s(x_left) - bbox[0]
    y = s(cy) - th / 2.0 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)
    return x_left + (bbox[2] - bbox[0]) / SCALE


def draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    tip_pos: Tuple[float, float],
    tip_dir: Tuple[float, float],
    color: Tuple[int, int, int],
    length: float = 8.0,
    half_width: float = 4.2,
) -> None:
    tx, ty = tip_pos
    ux, uy = tip_dir
    px, py = -uy, ux
    bx, by = tx - ux * length, ty - uy * length
    poly = [
        (s(tx), s(ty)),
        (s(bx + px * half_width), s(by + py * half_width)),
        (s(bx - px * half_width), s(by - py * half_width)),
    ]
    draw.polygon(poly, fill=color)


def composite_glow_patch(
    canvas: Image.Image,
    bbox_logical: Tuple[float, float, float, float],
    draw_fn,
    blur_radius_logical: float,
    pad_logical: float = 24.0,
) -> None:
    """Render a soft RGBA glow layer on a localized sub-rectangle and alpha-composite it.

    Keeping glow operations local guarantees bit-identical static background pixels
    across frames for clean, compact GIF delta compression.
    """
    x1 = max(0, int(math.floor(s(bbox_logical[0] - pad_logical))))
    y1 = max(0, int(math.floor(s(bbox_logical[1] - pad_logical))))
    x2 = min(SUP_W, int(math.ceil(s(bbox_logical[2] + pad_logical))))
    y2 = min(SUP_H, int(math.ceil(s(bbox_logical[3] + pad_logical))))
    if x2 <= x1 or y2 <= y1:
        return

    patch = Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(patch)
    draw_fn(pdraw, x1, y1)
    if blur_radius_logical > 0:
        patch = patch.filter(ImageFilter.GaussianBlur(radius=s(blur_radius_logical)))

    sub = canvas.crop((x1, y1, x2, y2)).convert("RGBA")
    blended = Image.alpha_composite(sub, patch).convert("RGB")
    canvas.paste(blended, (x1, y1))


def draw_node_card(
    canvas: Image.Image,
    node: NodeSpec,
    fonts: FontSet,
    active_color: Optional[Tuple[int, int, int]] = None,
    glow_strength: float = 0.0,
) -> None:
    """Draw a node box (with optional localized Gaussian halo when active)."""
    if active_color is not None and glow_strength > 0.0:
        alpha_outer = int(round(115 * glow_strength))
        stroke_glow = max(2, int(round(s(4.0))))

        def _glow(pdraw: ImageDraw.ImageDraw, ox: int, oy: int) -> None:
            pdraw.rounded_rectangle(
                (s(node.x1) - ox, s(node.y1) - oy, s(node.x2) - ox, s(node.y2) - oy),
                radius=s(node.radius),
                outline=(*active_color, alpha_outer),
                width=stroke_glow,
            )

        composite_glow_patch(
            canvas,
            (node.x1, node.y1, node.x2, node.y2),
            _glow,
            blur_radius_logical=8.5,
            pad_logical=22.0,
        )

    draw = ImageDraw.Draw(canvas)
    base_bg = PILL_BG_RGB if node.is_pill else NODE_BG_RGB
    if active_color is not None and glow_strength > 0.0:
        bg_col = lerp_color(base_bg, active_color, 0.11 * glow_strength)
        border_col = lerp_color(NODE_BORDER_IDLE_RGB, active_color, min(1.0, 0.35 + 0.65 * glow_strength))
        border_w = int(round(s(2.0)))
    else:
        bg_col = base_bg
        border_col = NODE_BORDER_IDLE_RGB
        border_w = int(round(s(1.4)))

    draw.rounded_rectangle(
        (s(node.x1), s(node.y1), s(node.x2), s(node.y2)),
        radius=s(node.radius),
        fill=bg_col,
        outline=border_col,
        width=border_w,
    )

    # Text styling
    if active_color is not None and glow_strength > 0.2:
        title_col = lerp_color(TEXT_TITLE_IDLE_RGB, (255, 255, 255), glow_strength)
        sub1_col = lerp_color(TEXT_SUB_RGB, active_color, 0.45 * glow_strength)
        sub2_col = lerp_color(TEXT_MUTED_RGB, TEXT_TITLE_IDLE_RGB, 0.45 * glow_strength)
    elif node.is_muted:
        title_col = lerp_color(TEXT_TITLE_IDLE_RGB, TEXT_SUB_RGB, 0.35)
        sub1_col = TEXT_MUTED_RGB
        sub2_col = TEXT_MUTED_RGB
    else:
        title_col = TEXT_TITLE_IDLE_RGB
        sub1_col = TEXT_SUB_RGB
        sub2_col = TEXT_MUTED_RGB

    if node.id == "query":
        draw_centered_text(draw, node.cx, node.cy, node.title, fonts.get("sans_bold", 13.5), title_col)
    elif node.is_pill:
        draw_centered_text(draw, node.cx, node.cy - 9.5, node.title, fonts.get("sans_bold", 13.5), title_col)
        draw_centered_text(draw, node.cx, node.cy + 10.5, node.sub1, fonts.get("mono_bold", 10.0), sub1_col)
    elif node.sub2:
        draw_centered_text(draw, node.cx, node.cy - 18.0, node.title, fonts.get("sans_bold", 13.5), title_col)
        draw_centered_text(draw, node.cx, node.cy + 2.0, node.sub1, fonts.get("mono_regular", 10.0), sub1_col)
        draw_centered_text(draw, node.cx, node.cy + 20.0, node.sub2, fonts.get("mono_regular", 10.0), sub2_col)
    else:
        draw_centered_text(draw, node.cx, node.cy - 10.0, node.title, fonts.get("sans_bold", 13.0), title_col)
        draw_centered_text(draw, node.cx, node.cy + 11.0, node.sub1, fonts.get("mono_regular", 10.0), sub1_col)


def draw_edge_label_pill(
    draw: ImageDraw.ImageDraw,
    label: EdgeLabelSpec,
    fonts: FontSet,
    active_color: Optional[Tuple[int, int, int]] = None,
    glow_strength: float = 0.0,
) -> None:
    font = fonts.get(label.font_role, label.font_size)
    bbox = draw.textbbox((0, 0), label.text, font=font)
    tw_log = (bbox[2] - bbox[0]) / SCALE
    th_log = (bbox[3] - bbox[1]) / SCALE

    hw = tw_log / 2.0 + label.pad_x
    hh = th_log / 2.0 + label.pad_y
    x1, y1 = label.cx - hw, label.cy - hh
    x2, y2 = label.cx + hw, label.cy + hh

    if active_color is not None and glow_strength > 0.0:
        fill_col = lerp_color(BG_RGB, active_color, 0.14 * glow_strength)
        border_col = lerp_color(NODE_BORDER_IDLE_RGB, active_color, min(1.0, 0.4 + 0.6 * glow_strength))
        txt_col = lerp_color(TEXT_SUB_RGB, lerp_color(active_color, (255, 255, 255), 0.35), glow_strength)
    else:
        fill_col = BG_RGB
        border_col = NODE_BORDER_IDLE_RGB
        txt_col = TEXT_SUB_RGB

    draw.rounded_rectangle(
        (s(x1), s(y1), s(x2), s(y2)),
        radius=s(hh),
        fill=fill_col,
        outline=border_col,
        width=int(round(s(1.0))),
    )
    draw_centered_text(draw, label.cx, label.cy, label.text, font, txt_col)


def draw_edge_stroke(
    canvas: Image.Image,
    eid: str,
    color: Tuple[int, int, int],
    width_logical: float,
    progress: float = 1.0,
    with_arrow: bool = True,
    glow: bool = False,
) -> None:
    pts_full, _, tip_pos, tip_dir = EDGE_GEOM[eid]
    pts = slice_edge_to_fraction(eid, progress)
    if len(pts) < 2:
        return

    scaled_pts = [(s(px), s(py)) for px, py in pts]
    if glow:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        def _edge_glow(pdraw: ImageDraw.ImageDraw, ox: int, oy: int) -> None:
            shifted = [(px - ox, py - oy) for px, py in scaled_pts]
            pdraw.line(shifted, fill=(*color, 115), width=int(round(s(width_logical + 3.5))), joint="curve")

        composite_glow_patch(canvas, bbox, _edge_glow, blur_radius_logical=5.0, pad_logical=16.0)

    draw = ImageDraw.Draw(canvas)
    draw.line(scaled_pts, fill=color, width=int(round(s(width_logical))), joint="curve")
    if with_arrow and progress >= 0.96:
        draw_arrowhead(draw, tip_pos, tip_dir, color)


# ---------------------------------------------------------------------------
# Static Base Canvas (Rendered Once at 2400x1240)
# ---------------------------------------------------------------------------
def build_static_base(fonts: FontSet) -> Image.Image:
    """Create the immutable dark-card canvas, dot grid, header title, idle graph, and footer legend."""
    base = Image.new("RGB", (SUP_W, SUP_H), BG_RGB)
    draw = ImageDraw.Draw(base)

    # 1. Subtle #21262d inner card border inset by 2px logical
    draw.rounded_rectangle(
        (s(2), s(2), s(LOGICAL_W - 2), s(LOGICAL_H - 2)),
        radius=s(10),
        fill=BG_RGB,
        outline=CARD_BORDER_RGB,
        width=int(round(s(1.2))),
    )

    # 2. Faint #1a2230 dot grid (24px spacing) in graph region
    dot_r = s(0.95)
    for gx in range(24, LOGICAL_W - 12, 24):
        for gy in range(84, 552, 24):
            draw.ellipse(
                (s(gx) - dot_r, s(gy) - dot_r, s(gx) + dot_r, s(gy) + dot_r),
                fill=DOT_GRID_RGB,
            )

    # 3. Subtle horizontal header & footer hairlines
    draw.line((s(24), s(72), s(LOGICAL_W - 24), s(72)), fill=CARD_BORDER_RGB, width=int(round(s(1.0))))
    draw.line((s(24), s(556), s(LOGICAL_W - 24), s(556)), fill=CARD_BORDER_RGB, width=int(round(s(1.0))))

    # 4. Header left title + subtitle
    draw_left_text_vcenter(draw, 28, 30, "Self-Reflective Agentic RAG", fonts.get("sans_bold", 21.0), TEXT_PRIMARY_RGB)
    draw_left_text_vcenter(
        draw,
        28,
        54,
        "LangGraph Bounded Cyclic StateGraph  ·  src/graph.py",
        fonts.get("mono_regular", 12.0),
        TEXT_SUB_RGB,
    )

    # 5. Idle edges (muted first, then normal idle edges)
    for eid, spec in EDGES.items():
        col = REFLECT_EDGE_IDLE_RGB if spec.is_muted else EDGE_IDLE_RGB
        draw_edge_stroke(base, eid, col, width_logical=1.8, progress=1.0, with_arrow=True, glow=False)

    draw = ImageDraw.Draw(base)
    for eid, spec in EDGES.items():
        if spec.label is not None:
            draw_edge_label_pill(draw, spec.label, fonts, active_color=None, glow_strength=0.0)

    # 6. Idle nodes
    for node in NODES.values():
        draw_node_card(base, node, fonts, active_color=None, glow_strength=0.0)

    # 7. Footer compact color legend (x = 28..420, cy = 585)
    draw = ImageDraw.Draw(base)
    legend_items = [
        (CYAN_RGB, EMERALD_RGB, "Grounded Path"),
        (ORANGE_RGB, AMBER_RGB, "Cyclic Retry / Reflect"),
        (RED_RGB, RED_RGB, "Guardrail / Fallback"),
    ]
    lx = 28.0
    for c_a, c_b, label in legend_items:
        dot_c = lerp_color(c_a, c_b, 0.5)
        r = 4.5
        draw.ellipse((s(lx), s(585 - r), s(lx + 2 * r), s(585 + r)), fill=dot_c)
        lx += 2 * r + 7.0
        lx = draw_left_text_vcenter(draw, lx, 585, label, fonts.get("sans_medium", 11.0), TEXT_SUB_RGB)
        lx += 20.0

    # 8. Bottom progress track background
    draw.rounded_rectangle(
        (s(16), s(611), s(LOGICAL_W - 16), s(614.5)),
        radius=s(1.8),
        fill=NODE_BG_RGB,
    )

    return base


# ---------------------------------------------------------------------------
# Comet Particle & Trail Rendering
# ---------------------------------------------------------------------------
def collect_comet_points(
    segments: Sequence[Tuple[str, float, float]],
    head_global_t: float,
    trail_count: int = 5,
    trail_step: float = 0.032,
) -> List[Tuple[float, float, float]]:
    """Evaluate head and fading trail positions along a sequence of (edge_id, start_t, end_t).

    Returns list of (x, y, intensity) ordered back-to-front (tail first, head last).
    """
    def eval_at(t: float) -> Optional[Tuple[float, float]]:
        for eid, t0, t1 in segments:
            if t0 <= t <= t1:
                u = (t - t0) / max(1e-6, t1 - t0)
                return sample_edge_at_fraction(eid, u)
        return None

    pts: List[Tuple[float, float, float]] = []
    min_t = segments[0][1]
    for i in range(trail_count, 0, -1):
        t_pos = head_global_t - i * trail_step
        if t_pos >= min_t:
            xy = eval_at(t_pos)
            if xy is not None:
                strength = 0.62 * (1.0 - (i - 1) / trail_count)
                pts.append((xy[0], xy[1], strength))
    head_xy = eval_at(head_global_t)
    if head_xy is not None:
        pts.append((head_xy[0], head_xy[1], 1.0))
    return pts


def draw_comet(
    canvas: Image.Image,
    comet_pts: Sequence[Tuple[float, float, float]],
    color: Tuple[int, int, int],
) -> None:
    """Render Gaussian halo + crisp bright core + 5 fading trail dots."""
    if not comet_pts:
        return
    xs = [p[0] for p in comet_pts]
    ys = [p[1] for p in comet_pts]
    bbox = (min(xs), min(ys), max(xs), max(ys))

    def _glow(pdraw: ImageDraw.ImageDraw, ox: int, oy: int) -> None:
        for px, py, strength in comet_pts[:-1]:
            r = 2.2 + 2.4 * strength
            a = int(round(145 * strength))
            pdraw.ellipse(
                (s(px - r) - ox, s(py - r) - oy, s(px + r) - ox, s(py + r) - oy),
                fill=(*color, a),
            )
        hx, hy, _ = comet_pts[-1]
        hr = 10.5
        pdraw.ellipse(
            (s(hx - hr) - ox, s(hy - hr) - oy, s(hx + hr) - ox, s(hy + hr) - oy),
            fill=(*color, 195),
        )

    composite_glow_patch(canvas, bbox, _glow, blur_radius_logical=4.5, pad_logical=18.0)

    draw = ImageDraw.Draw(canvas)
    for px, py, strength in comet_pts[:-1]:
        r = 1.6 + 1.8 * strength
        c = lerp_color(color, (255, 255, 255), 0.35 * strength)
        draw.ellipse((s(px - r), s(py - r), s(px + r), s(py + r)), fill=c)

    hx, hy, _ = comet_pts[-1]
    r_mid = 4.2
    r_core = 2.1
    draw.ellipse((s(hx - r_mid), s(hy - r_mid), s(hx + r_mid), s(hy + r_mid)), fill=color)
    draw.ellipse((s(hx - r_core), s(hy - r_core), s(hx + r_core), s(hy + r_core)), fill=(255, 255, 255))


# ---------------------------------------------------------------------------
# Animation Choreography (Frames 0..119)
# ---------------------------------------------------------------------------
@dataclass
class FrameState:
    header_text: str
    header_color: Tuple[int, int, int]
    telemetry_text: str
    telemetry_color: Tuple[int, int, int]
    progress_color: Tuple[int, int, int]
    lit_nodes: Dict[str, Tuple[Tuple[int, int, int], float]]
    lit_edges: Dict[str, Tuple[Tuple[int, int, int], float]]  # eid -> (color, progress)
    lit_labels: Dict[str, Tuple[Tuple[int, int, int], float]]  # eid -> (color, glow)
    comet: Optional[Tuple[List[Tuple[float, float, float]], Tuple[int, int, int]]] = None


def compute_segment_1(f: int) -> FrameState:
    """Segment 1 (Frames 0-47, 3.2s): Grounded Flow (cyan -> emerald)."""
    # 6 hops over frames 0..35, hold frames 36..47
    hops = [
        ("query->decompose", "query", "decompose", 0.0, 5.5, CYAN_RGB),
        ("decompose->retrieve", "decompose", "retrieve", 5.5, 11.0, CYAN_RGB),
        ("retrieve->grade_ret", "retrieve", "grade_ret", 11.0, 16.5, CYAN_RGB),
        ("grade_ret->generate", "grade_ret", "generate", 16.5, 23.0, CYAN_RGB),
        ("generate->grade_hal", "generate", "grade_hal", 23.0, 29.0, CYAN_RGB),
        ("grade_hal->verified", "grade_hal", "verified", 29.0, 35.5, EMERALD_RGB),
    ]

    lit_nodes: Dict[str, Tuple[Tuple[int, int, int], float]] = {"query": (CYAN_RGB, 0.85)}
    lit_edges: Dict[str, Tuple[Tuple[int, int, int], float]] = {}
    lit_labels: Dict[str, Tuple[Tuple[int, int, int], float]] = {}
    comet = None

    # Normalized timeline for edge traversal
    tf = float(f)
    for eid, src, dst, f0, f1, ecol in hops:
        if tf >= f1:
            lit_edges[eid] = (ecol, 1.0)
            lit_nodes[src] = (CYAN_RGB if src != "verified" else EMERALD_RGB, 0.85)
            dst_col = EMERALD_RGB if dst == "verified" else CYAN_RGB
            lit_nodes[dst] = (dst_col, 0.90)
            if EDGES[eid].label is not None:
                lit_labels[eid] = (ecol, 0.90)
        elif f0 <= tf < f1:
            u = (tf - f0) / (f1 - f0)
            lit_edges[eid] = (ecol, u)
            lit_nodes[src] = (CYAN_RGB, 0.95)
            if u > 0.55 and EDGES[eid].label is not None:
                lit_labels[eid] = (ecol, (u - 0.55) / 0.45)
            if u > 0.72:
                dst_col = EMERALD_RGB if dst == "verified" else CYAN_RGB
                lit_nodes[dst] = (dst_col, (u - 0.72) / 0.28)
            segs = [(h[0], h[3], h[4]) for h in hops]
            c_pts = collect_comet_points(segs, tf, trail_count=5, trail_step=0.65)
            c_col = lerp_color(CYAN_RGB, EMERALD_RGB, u) if eid == "grade_hal->verified" else CYAN_RGB
            comet = (c_pts, c_col)

    if f >= 36:
        pulse = 0.72 + 0.28 * math.sin((f - 36) * 0.55)
        lit_nodes["verified"] = (EMERALD_RGB, pulse)
        header_col = EMERALD_RGB
    elif f >= 29:
        header_col = lerp_color(CYAN_RGB, EMERALD_RGB, (f - 29) / 7.0)
    else:
        header_col = CYAN_RGB

    if f < 16:
        tel = "State: sub_queries<=3 · top_k=50 · rerank_top_n=8"
        tel_col = CYAN_RGB
    elif f < 25:
        tel = "State: is_relevant=True · docs=8 (th>=0.03) -> generate"
        tel_col = CYAN_RGB
    elif f < 35:
        tel = "State: verifying NLI · entail>=0.40 · contra<=0.25"
        tel_col = CYAN_RGB
    else:
        tel = "State: is_relevant=True · entail=0.84 · contra=0.02 -> Verified"
        tel_col = EMERALD_RGB

    return FrameState(
        header_text="PATH 1 · GROUNDED FLOW (IN-SCOPE)",
        header_color=header_col,
        telemetry_text=tel,
        telemetry_color=tel_col,
        progress_color=header_col,
        lit_nodes=lit_nodes,
        lit_edges=lit_edges,
        lit_labels=lit_labels,
        comet=comet,
    )


def compute_segment_2(f: int) -> FrameState:
    """Segment 2 (Frames 48-83, 2.4s): OOD / 0-Docs Guardrail (orange -> red)."""
    lit_nodes: Dict[str, Tuple[Tuple[int, int, int], float]] = {}
    lit_edges: Dict[str, Tuple[Tuple[int, int, int], float]] = {}
    lit_labels: Dict[str, Tuple[Tuple[int, int, int], float]] = {}
    comet = None

    # Phase A (48..55): quick sweep query -> decompose -> retrieve -> grade_ret
    if f <= 55:
        sweep_segs = [
            ("query->decompose", "query", "decompose", 48.0, 50.2),
            ("decompose->retrieve", "decompose", "retrieve", 50.2, 52.6),
            ("retrieve->grade_ret", "retrieve", "grade_ret", 52.6, 55.2),
        ]
        tf = float(f)
        lit_nodes["query"] = (CYAN_RGB, 0.85)
        for eid, src, dst, t0, t1 in sweep_segs:
            col = lerp_color(CYAN_RGB, ORANGE_RGB, (tf - 48.0) / 7.5)
            if tf >= t1:
                lit_edges[eid] = (col, 1.0)
                lit_nodes[src] = (col, 0.8)
                lit_nodes[dst] = (col, 0.9)
            elif t0 <= tf < t1:
                u = (tf - t0) / (t1 - t0)
                lit_edges[eid] = (col, u)
                lit_nodes[src] = (col, 0.85)
                if u > 0.6:
                    lit_nodes[dst] = (col, (u - 0.6) / 0.4)
                c_pts = collect_comet_points([(s[0], s[3], s[4]) for s in sweep_segs], tf, trail_count=5, trail_step=0.38)
                comet = (c_pts, col)
        tel = "State: sub_queries<=3 · top_k=50 · reranking candidates"
        return FrameState(
            header_text="PATH 2 · OOD GUARDRAIL (0 DOCS -> FALLBACK)",
            header_color=ORANGE_RGB,
            telemetry_text=tel,
            telemetry_color=ORANGE_RGB,
            progress_color=ORANGE_RGB,
            lit_nodes=lit_nodes,
            lit_edges=lit_edges,
            lit_labels=lit_labels,
            comet=comet,
        )

    # Keep initial sweep softly lit in orange
    for eid in ("query->decompose", "decompose->retrieve", "retrieve->grade_ret"):
        lit_edges[eid] = (ORANGE_RGB, 1.0)
    for nid in ("query", "decompose", "retrieve"):
        lit_nodes[nid] = (ORANGE_RGB, 0.55)

    # Phase B (56..58): grade_ret flashes red (0 docs)
    if f <= 58:
        flash = 0.75 + 0.25 * math.sin((f - 55) * 1.4)
        lit_nodes["grade_ret"] = (RED_RGB, flash)
        lit_labels["grade_ret->transform"] = (ORANGE_RGB, 0.7)
        return FrameState(
            header_text="PATH 2 · OOD GUARDRAIL (0 DOCS -> FALLBACK)",
            header_color=ORANGE_RGB,
            telemetry_text="State: is_relevant=False (0 docs) · rewrite_retry=0/1 -> transform",
            telemetry_color=ORANGE_RGB,
            progress_color=ORANGE_RGB,
            lit_nodes=lit_nodes,
            lit_edges=lit_edges,
            lit_labels=lit_labels,
            comet=None,
        )

    # Phase C (59..65): orange comet loops grade_ret -> transform -> retrieve -> grade_ret
    if f <= 65:
        loop_segs = [
            ("grade_ret->transform", 59.0, 61.2),
            ("transform->retrieve", 61.2, 63.4),
            ("retrieve->grade_ret", 63.4, 65.5),
        ]
        tf = float(f)
        lit_nodes["grade_ret"] = (ORANGE_RGB, 0.85)
        lit_labels["grade_ret->transform"] = (ORANGE_RGB, 0.95)
        for eid, t0, t1 in loop_segs:
            if tf >= t1:
                lit_edges[eid] = (ORANGE_RGB, 1.0)
            elif t0 <= tf < t1:
                lit_edges[eid] = (ORANGE_RGB, (tf - t0) / (t1 - t0))
        if tf >= 60.5:
            lit_nodes["transform"] = (ORANGE_RGB, 0.95)
        if tf >= 63.0:
            lit_nodes["retrieve"] = (ORANGE_RGB, 0.90)
        c_pts = collect_comet_points(loop_segs, tf, trail_count=5, trail_step=0.34)
        comet = (c_pts, ORANGE_RGB)
        return FrameState(
            header_text="PATH 2 · OOD GUARDRAIL (0 DOCS -> FALLBACK)",
            header_color=ORANGE_RGB,
            telemetry_text="State: is_relevant=False (0 docs) · rewrite_retry=1/1 -> retrieve",
            telemetry_color=ORANGE_RGB,
            progress_color=ORANGE_RGB,
            lit_nodes=lit_nodes,
            lit_edges=lit_edges,
            lit_labels=lit_labels,
            comet=comet,
        )

    # Keep transform loop lit orange as history
    lit_edges["grade_ret->transform"] = (ORANGE_RGB, 1.0)
    lit_edges["transform->retrieve"] = (ORANGE_RGB, 1.0)
    lit_nodes["transform"] = (ORANGE_RGB, 0.65)
    lit_labels["grade_ret->transform"] = (ORANGE_RGB, 0.70)

    # Phase D (66..68): grade_ret flashes red again (rewrite_retry=1/1 exhausted)
    if f <= 68:
        lit_nodes["grade_ret"] = (RED_RGB, 1.0)
        lit_labels["grade_ret->fallback"] = (RED_RGB, 0.85)
        return FrameState(
            header_text="PATH 2 · OOD GUARDRAIL (0 DOCS -> FALLBACK)",
            header_color=RED_RGB,
            telemetry_text="State: is_relevant=False (0 docs) · rewrite_retry=1/1 (max)",
            telemetry_color=RED_RGB,
            progress_color=RED_RGB,
            lit_nodes=lit_nodes,
            lit_edges=lit_edges,
            lit_labels=lit_labels,
            comet=None,
        )

    # Phase E (69..75): red comet drops grade_ret -> fallback -> safe_end; hold 76..83
    lit_nodes["grade_ret"] = (RED_RGB, 0.90)
    lit_labels["grade_ret->fallback"] = (RED_RGB, 0.95)
    drop_segs = [
        ("grade_ret->fallback", 69.0, 72.5),
        ("fallback->safe_end", 72.5, 75.8),
    ]
    tf = float(f)
    if f <= 75:
        for eid, t0, t1 in drop_segs:
            if tf >= t1:
                lit_edges[eid] = (RED_RGB, 1.0)
            elif t0 <= tf < t1:
                lit_edges[eid] = (RED_RGB, (tf - t0) / (t1 - t0))
        if tf >= 71.8:
            lit_nodes["fallback"] = (RED_RGB, 0.95)
        if tf >= 73.5:
            lit_labels["fallback->safe_end"] = (RED_RGB, 0.95)
        if tf >= 75.0:
            lit_nodes["safe_end"] = (RED_RGB, 0.90)
        c_pts = collect_comet_points(drop_segs, tf, trail_count=5, trail_step=0.38)
        comet = (c_pts, RED_RGB)
    else:
        lit_edges["grade_ret->fallback"] = (RED_RGB, 1.0)
        lit_edges["fallback->safe_end"] = (RED_RGB, 1.0)
        lit_labels["fallback->safe_end"] = (RED_RGB, 0.95)
        lit_nodes["fallback"] = (RED_RGB, 0.90)
        pulse = 0.72 + 0.28 * math.sin((f - 76) * 0.7)
        lit_nodes["safe_end"] = (RED_RGB, pulse)

    return FrameState(
        header_text="PATH 2 · OOD GUARDRAIL (0 DOCS -> FALLBACK)",
        header_color=RED_RGB,
        telemetry_text="State: is_relevant=False (0 docs) · rewrite_retry=1/1 -> fallback_node",
        telemetry_color=RED_RGB,
        progress_color=RED_RGB,
        lit_nodes=lit_nodes,
        lit_edges=lit_edges,
        lit_labels=lit_labels,
        comet=comet,
    )


def compute_segment_3(f: int) -> FrameState:
    """Segment 3 (Frames 84-119, 2.4s): NLI Hallucination Defense (amber -> red)."""
    lit_nodes: Dict[str, Tuple[Tuple[int, int, int], float]] = {}
    lit_edges: Dict[str, Tuple[Tuple[int, int, int], float]] = {}
    lit_labels: Dict[str, Tuple[Tuple[int, int, int], float]] = {}
    comet = None

    # Keep upstream context softly grounded so focus stays on generate <-> grade_hal -> fallback
    for nid in ("query", "decompose", "retrieve", "grade_ret"):
        lit_nodes[nid] = (CYAN_RGB, 0.35)
    for eid in ("query->decompose", "decompose->retrieve", "retrieve->grade_ret", "grade_ret->generate"):
        lit_edges[eid] = (lerp_color(EDGE_IDLE_RGB, CYAN_RGB, 0.45), 1.0)
    lit_labels["grade_ret->generate"] = (CYAN_RGB, 0.35)

    # Phase A (84..87): quick pulse generate -> grade_hal
    if f <= 87:
        tf = float(f)
        lit_nodes["generate"] = (AMBER_RGB, 0.95)
        u = min(1.0, (tf - 84.0) / 3.2)
        lit_edges["generate->grade_hal"] = (AMBER_RGB, u)
        if u > 0.65:
            lit_nodes["grade_hal"] = (AMBER_RGB, (u - 0.65) / 0.35)
        c_pts = collect_comet_points([("generate->grade_hal", 84.0, 87.2)], tf, trail_count=5, trail_step=0.35)
        comet = (c_pts, AMBER_RGB)
        return FrameState(
            header_text="PATH 3 · NLI GUARDRAIL (RETRY -> FALLBACK)",
            header_color=AMBER_RGB,
            telemetry_text="State: draft generated · verifying ModernBERT-NLI",
            telemetry_color=AMBER_RGB,
            progress_color=AMBER_RGB,
            lit_nodes=lit_nodes,
            lit_edges=lit_edges,
            lit_labels=lit_labels,
            comet=comet,
        )

    lit_edges["generate->grade_hal"] = (AMBER_RGB, 1.0)
    lit_nodes["generate"] = (AMBER_RGB, 0.75)

    # Phase B (88..90): grade_hal flashes red (NLI Flag)
    if f <= 90:
        flash = 0.8 + 0.2 * math.sin((f - 88) * 1.5)
        lit_nodes["grade_hal"] = (RED_RGB, flash)
        lit_labels["grade_hal->generate"] = (AMBER_RGB, 0.75)
        return FrameState(
            header_text="PATH 3 · NLI GUARDRAIL (RETRY -> FALLBACK)",
            header_color=AMBER_RGB,
            telemetry_text="State: has_hallucination=True · contra=0.61 · hal_retry=0/2",
            telemetry_color=RED_RGB,
            progress_color=AMBER_RGB,
            lit_nodes=lit_nodes,
            lit_edges=lit_edges,
            lit_labels=lit_labels,
            comet=None,
        )

    # Phase C (91..101): amber comet loops grade_hal -> generate -> grade_hal (hal_retry < 2)
    if f <= 101:
        loop_segs = [
            ("grade_hal->generate", 91.0, 97.5),
            ("generate->grade_hal", 97.5, 101.2),
        ]
        tf = float(f)
        lit_nodes["grade_hal"] = (AMBER_RGB, 0.85)
        lit_labels["grade_hal->generate"] = (AMBER_RGB, 0.95)
        if tf < 97.5:
            lit_edges["grade_hal->generate"] = (AMBER_RGB, (tf - 91.0) / 6.5)
        else:
            lit_edges["grade_hal->generate"] = (AMBER_RGB, 1.0)
            lit_nodes["generate"] = (AMBER_RGB, 1.0)
            lit_edges["generate->grade_hal"] = (AMBER_RGB, (tf - 97.5) / 3.7)
        c_pts = collect_comet_points(loop_segs, tf, trail_count=5, trail_step=0.42)
        comet = (c_pts, AMBER_RGB)
        return FrameState(
            header_text="PATH 3 · NLI GUARDRAIL (RETRY -> FALLBACK)",
            header_color=AMBER_RGB,
            telemetry_text="State: has_hallucination=True · hal_retry=1/2 -> generate",
            telemetry_color=AMBER_RGB,
            progress_color=AMBER_RGB,
            lit_nodes=lit_nodes,
            lit_edges=lit_edges,
            lit_labels=lit_labels,
            comet=comet,
        )

    lit_edges["grade_hal->generate"] = (AMBER_RGB, 1.0)
    lit_labels["grade_hal->generate"] = (AMBER_RGB, 0.70)

    # Phase D (102..104): grade_hal flashes red again (hal_retry=2/2)
    if f <= 104:
        lit_nodes["grade_hal"] = (RED_RGB, 1.0)
        lit_labels["grade_hal->fallback"] = (RED_RGB, 0.85)
        return FrameState(
            header_text="PATH 3 · NLI GUARDRAIL (RETRY -> FALLBACK)",
            header_color=RED_RGB,
            telemetry_text="State: has_hallucination=True · contra=0.58 · hal_retry=2/2",
            telemetry_color=RED_RGB,
            progress_color=RED_RGB,
            lit_nodes=lit_nodes,
            lit_edges=lit_edges,
            lit_labels=lit_labels,
            comet=None,
        )

    # Phase E (105..110): red comet drops grade_hal -> fallback -> safe_end; hold 111..119
    lit_nodes["grade_hal"] = (RED_RGB, 0.90)
    lit_labels["grade_hal->fallback"] = (RED_RGB, 0.95)
    drop_segs = [
        ("grade_hal->fallback", 105.0, 108.0),
        ("fallback->safe_end", 108.0, 110.8),
    ]
    tf = float(f)
    if f <= 110:
        for eid, t0, t1 in drop_segs:
            if tf >= t1:
                lit_edges[eid] = (RED_RGB, 1.0)
            elif t0 <= tf < t1:
                lit_edges[eid] = (RED_RGB, (tf - t0) / (t1 - t0))
        if tf >= 107.2:
            lit_nodes["fallback"] = (RED_RGB, 0.95)
        if tf >= 108.8:
            lit_labels["fallback->safe_end"] = (RED_RGB, 0.95)
        if tf >= 110.2:
            lit_nodes["safe_end"] = (RED_RGB, 0.90)
        c_pts = collect_comet_points(drop_segs, tf, trail_count=5, trail_step=0.36)
        comet = (c_pts, RED_RGB)
    else:
        lit_edges["grade_hal->fallback"] = (RED_RGB, 1.0)
        lit_edges["fallback->safe_end"] = (RED_RGB, 1.0)
        lit_labels["fallback->safe_end"] = (RED_RGB, 0.95)
        lit_nodes["fallback"] = (RED_RGB, 0.90)
        pulse = 0.72 + 0.28 * math.sin((f - 111) * 0.65)
        lit_nodes["safe_end"] = (RED_RGB, pulse)

    return FrameState(
        header_text="PATH 3 · NLI GUARDRAIL (RETRY -> FALLBACK)",
        header_color=RED_RGB,
        telemetry_text="State: has_hallucination=True · hal_retry=2/2 -> fallback_node",
        telemetry_color=RED_RGB,
        progress_color=RED_RGB,
        lit_nodes=lit_nodes,
        lit_edges=lit_edges,
        lit_labels=lit_labels,
        comet=comet,
    )


def get_frame_state(frame_idx: int) -> FrameState:
    if frame_idx < 48:
        return compute_segment_1(frame_idx)
    if frame_idx < 84:
        return compute_segment_2(frame_idx)
    return compute_segment_3(frame_idx)


# ---------------------------------------------------------------------------
# Frame Compositing (2400x1240 -> 1200x620 LANCZOS)
# ---------------------------------------------------------------------------
def draw_header_pill(canvas: Image.Image, state: FrameState, fonts: FontSet, frame_idx: int) -> None:
    x1, y1, x2, y2 = 760, 20, 1170, 62
    cy = (y1 + y2) / 2.0
    col = state.header_color
    pill_bg = lerp_color(NODE_BG_RGB, col, 0.08)
    pill_border = lerp_color(NODE_BORDER_IDLE_RGB, col, 0.65)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (s(x1), s(y1), s(x2), s(y2)),
        radius=s(21),
        fill=pill_bg,
        outline=pill_border,
        width=int(round(s(1.5))),
    )

    # Measure text + dot so group is centered horizontally inside x1..x2
    font = fonts.get("mono_bold", 11.5)
    tb = draw.textbbox((0, 0), state.header_text, font=font)
    tw = (tb[2] - tb[0]) / SCALE
    dot_d = 9.0
    gap = 10.0
    total_w = dot_d + gap + tw
    start_x = (x1 + x2) / 2.0 - total_w / 2.0

    # Glowing status dot
    dot_cx = start_x + dot_d / 2.0
    pulse = 0.8 + 0.2 * math.sin(frame_idx * 0.45)

    def _dot_glow(pdraw: ImageDraw.ImageDraw, ox: int, oy: int) -> None:
        r_g = 6.5
        pdraw.ellipse(
            (s(dot_cx - r_g) - ox, s(cy - r_g) - oy, s(dot_cx + r_g) - ox, s(cy + r_g) - oy),
            fill=(*col, int(round(190 * pulse))),
        )

    composite_glow_patch(
        canvas,
        (dot_cx - 7, cy - 7, dot_cx + 7, cy + 7),
        _dot_glow,
        blur_radius_logical=3.0,
        pad_logical=8.0,
    )

    draw = ImageDraw.Draw(canvas)
    r_dot = 4.2
    draw.ellipse((s(dot_cx - r_dot), s(cy - r_dot), s(dot_cx + r_dot), s(cy + r_dot)), fill=col)
    draw_left_text_vcenter(draw, start_x + dot_d + gap, cy, state.header_text, font, lerp_color(col, (255, 255, 255), 0.35))


def draw_footer_telemetry(canvas: Image.Image, state: FrameState, fonts: FontSet, frame_idx: int) -> None:
    x1, y1, x2, y2 = 656, 567, 1170, 603
    cy = (y1 + y2) / 2.0
    col = state.telemetry_color
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (s(x1), s(y1), s(x2), s(y2)),
        radius=s(8),
        fill=NODE_BG_RGB,
        outline=lerp_color(NODE_BORDER_IDLE_RGB, col, 0.45),
        width=int(round(s(1.2))),
    )
    font = fonts.get("mono_regular", 10.5)
    txt_col = lerp_color(col, TEXT_PRIMARY_RGB, 0.45)
    draw_centered_text(draw, (x1 + x2) / 2.0, cy, state.telemetry_text, font, txt_col)

    # Thin animated loop progress bar along bottom edge (y = 611..614.5)
    bar_x1 = 16.0
    bar_x2 = LOGICAL_W - 16.0
    frac = (frame_idx + 1) / float(TOTAL_FRAMES)
    fill_x2 = bar_x1 + (bar_x2 - bar_x1) * frac
    draw.rounded_rectangle(
        (s(bar_x1), s(611), s(fill_x2), s(614.5)),
        radius=s(1.8),
        fill=state.progress_color,
    )


def render_frame(base: Image.Image, fonts: FontSet, frame_idx: int) -> Image.Image:
    canvas = base.copy()
    state = get_frame_state(frame_idx)

    # 1. Active edges (drawn before nodes so arrowheads sit cleanly at borders)
    for eid, (col, progress) in state.lit_edges.items():
        draw_edge_stroke(canvas, eid, col, width_logical=2.6, progress=progress, with_arrow=True, glow=True)

    # 2. Comet particle along active path (behind node cards if entering/exiting)
    if state.comet is not None:
        c_pts, c_col = state.comet
        draw_comet(canvas, c_pts, c_col)

    # 3. Active nodes (sharp foreground with soft halo)
    for nid, (col, strength) in state.lit_nodes.items():
        draw_node_card(canvas, NODES[nid], fonts, active_color=col, glow_strength=strength)

    # 4. Re-draw edge label pills cleanly above node halos (idle + active)
    draw = ImageDraw.Draw(canvas)
    for eid, spec in EDGES.items():
        if spec.label is None:
            continue
        if eid in state.lit_labels:
            col, glow = state.lit_labels[eid]
            draw_edge_label_pill(draw, spec.label, fonts, active_color=col, glow_strength=glow)
        elif eid in ("grade_ret->generate", "grade_hal->verified"):
            # Keep compact spine pills crisp if adjacent nodes emit glow halos
            draw_edge_label_pill(draw, spec.label, fonts, active_color=None, glow_strength=0.0)

    # 5. Dynamic Header Pill & Footer Telemetry
    draw_header_pill(canvas, state, fonts, frame_idx)
    draw_footer_telemetry(canvas, state, fonts, frame_idx)

    # 6. Downsample 2x -> 1x with LANCZOS for sharp anti-aliasing
    return canvas.resize((LOGICAL_W, LOGICAL_H), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# Contact Sheet & FFmpeg GIF Encoding
# ---------------------------------------------------------------------------
def build_contact_sheet(frames_dir: Path, output_path: Path, frame_indices: Sequence[int], fonts: FontSet) -> None:
    """Create a 3x2 contact sheet of selected frames for visual QA inspection."""
    cols, rows = 3, 2
    tile_w, tile_h = 800, 413
    header_h = 26
    pad = 16
    cell_h = tile_h + header_h
    sheet_w = cols * tile_w + (cols + 1) * pad
    sheet_h = rows * cell_h + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (10, 12, 16))
    draw = ImageDraw.Draw(sheet)
    label_font = fonts.get("mono_bold", 7.0)  # scaled x2 internally -> 14px crisp label

    for idx, fnum in enumerate(frame_indices[:6]):
        r, c = divmod(idx, cols)
        x = pad + c * (tile_w + pad)
        y = pad + r * (cell_h + pad)
        draw.rectangle((x, y, x + tile_w - 1, y + header_h - 1), fill=(22, 27, 34))
        badge_text = f"frame_{fnum:04d}  ({fnum / FPS:.2f}s)"
        draw.text((x + 10, y + 5), badge_text, font=label_font, fill=(56, 189, 248))

        frame_img = Image.open(frames_dir / f"frame_{fnum:04d}.png").convert("RGB")
        tile = frame_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y + header_h))
        draw.rectangle((x, y, x + tile_w - 1, y + cell_h - 1), outline=(48, 54, 61), width=1)

    sheet.save(output_path, format="PNG")


def encode_gif(frames_dir: Path, palette_path: Path, gif_path: Path, max_colors: int = 128) -> None:
    """Encode frames into a crisp RGB-palette GIF via 2-pass FFmpeg."""
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    input_pattern = str(frames_dir / "frame_%04d.png")

    cmd_palette = [
        str(FFMPEG_BIN),
        "-y",
        "-framerate", str(FPS),
        "-i", input_pattern,
        "-vf", f"format=rgb24,palettegen=max_colors={max_colors}:stats_mode=diff",
        str(palette_path),
    ]
    subprocess.run(cmd_palette, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cmd_gif = [
        str(FFMPEG_BIN),
        "-y",
        "-framerate", str(FPS),
        "-i", input_pattern,
        "-i", str(palette_path),
        "-lavfi", "format=rgb24[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
        "-loop", "0",
        str(gif_path),
    ]
    subprocess.run(cmd_gif, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main() -> None:
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    fonts = FontSet()
    base = build_static_base(fonts)

    for idx in range(TOTAL_FRAMES):
        frame = render_frame(base, fonts, idx)
        frame.save(FRAMES_DIR / f"frame_{idx:04d}.png", format="PNG")

    build_contact_sheet(FRAMES_DIR, CONTACT_SHEET_PATH, CONTACT_FRAME_INDICES, fonts)
    encode_gif(FRAMES_DIR, PALETTE_PATH, GIF_OUTPUT_PATH, max_colors=128)

    size_bytes = GIF_OUTPUT_PATH.stat().st_size
    if size_bytes > 1_900_000:
        encode_gif(FRAMES_DIR, PALETTE_PATH, GIF_OUTPUT_PATH, max_colors=96)
        size_bytes = GIF_OUTPUT_PATH.stat().st_size

    print(f"Generated GIF: {GIF_OUTPUT_PATH} ({size_bytes:,} bytes)")
    print(f"Contact sheet: {CONTACT_SHEET_PATH}")


if __name__ == "__main__":
    main()
