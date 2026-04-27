"""Sentence Map Renderer (Phase 4).

Converts a nested Chunk tree into an SVG visualization.

Two rendering modes:
  - full_blocks (default): Each clause is one colored box containing ALL its
    tokens as a single sequential text line. Syntactic role is conveyed purely
    through text color. No nested sub-boxes.
  - clauses_only: Vertical tree-style layout with depth-based coloring and
    dashed connector lines for subordinate clauses.
"""
from xml.sax.saxutils import escape

from latin_reader.pipeline.chunker import Chunk
from latin_reader.latincy_postprocessor.sentence import Token

# ─── Color Palettes ──────────────────────────────────────────────────────────

CLAUSE_BOX_COLORS = {
    "MainClause":         "#E8F1FF",
    "SubordinateClause":  "#FFF4E0",
    "RelativeClause":     "#F3E8FF",
    "AblativeAbsolute":   "#FFE8E8",
    "IndirectStatement":  "#E8FFE8",
    "Clause":             "#E8F1FF",
    "Sentence":           "transparent",
}

CHUNK_TEXT_COLORS = {
    "NP_nsubj":           "#FDA4AF",  # Rose (Subjects)
    "NP_obj":             "#7DD3FC",  # Sky Blue (Direct Objects)
    "NP_iobj":            "#C4B5FD",  # Violet (Indirect Objects)
    "NP_obl":             "#6EE7B7",  # Emerald (Obliques/Ablatives)
    "NP_nmod":            "#CBD5E1",  # Slate (Genitive Modifiers)
    "NP_other":           "#93C5FD",  # Default light blue
    "NP":                 "#93C5FD",
    "PP":                 "#FCD34D",  # Amber
    "VP":                 "#A7F3D0",  # Emerald
    "ParticipialPhrase":  "#C4B5FD",  # Violet
    "Coordination":       "#FCA5A5",  # Red
    "AdjP":               "#93C5FD",
    "AdvP":               "#93C5FD",
    "Token":              "#D1D5DB",  # Gray fallback
}

DEPTH_COLORS = [
    ("#3B82F6", "#2563EB"),   # Depth 0: Blue
    ("#10B981", "#059669"),   # Depth 1: Emerald
    ("#8B5CF6", "#7C3AED"),   # Depth 2: Violet
    ("#F59E0B", "#D97706"),   # Depth 3: Amber
    ("#EF4444", "#DC2626"),   # Depth 4: Red
]

# ─── Layout Constants ─────────────────────────────────────────────────────────

CHAR_WIDTH_PX = 8.5
PAD = 12
GAP = 6
LINE_H = 20

CLAUSE_TYPES = frozenset([
    "Sentence", "Clause", "MainClause", "SubordinateClause",
    "RelativeClause", "IndirectStatement", "AblativeAbsolute",
])


# ─── Token Type Map ──────────────────────────────────────────────────────────

def _build_token_type_map(chunk: Chunk, out: dict, inherited=None):
    """Map each token_id → most-specific chunk type for text coloring."""
    if chunk.type in CLAUSE_TYPES:
        current = None
    else:
        t = chunk.type
        if t == "NP":
            role = (chunk.role or "").split(":")[0]
            if role in ("nsubj", "obj", "iobj", "obl", "nmod"):
                t = f"NP_{role}"
            else:
                t = "NP_other"
        current = inherited or t

    for tid in chunk.token_ids:
        out[tid] = current or chunk.type

    for child in chunk.children:
        child_inherit = current if current == "PP" else None
        _build_token_type_map(child, out, child_inherit)


# ─── Shared Token Rendering ──────────────────────────────────────────────────

def _render_token_spans(tids, token_map, type_map):
    """Build a list of <tspan> strings for the given token IDs (in order)."""
    spans = []
    for i, tid in enumerate(tids):
        tok = token_map.get(tid)
        if tok is None:
            continue
        form = escape(tok.form)
        lemma = escape(tok.lemma)
        short_def = f"[Placeholder definition for {lemma}]"

        # Spacing
        prefix = ""
        dx = "0"
        if i > 0:
            is_punct = tok.upos == "PUNCT" or tok.form in (",", ".", ";", ":", "?", "!", '"', "'")
            if not is_punct:
                prev_type = type_map.get(tids[i - 1], "Token")
                curr_type = type_map.get(tid, "Token")
                if curr_type != prev_type and prev_type != "Token":
                    dx = "8"
                prefix = " "

        # Color
        t_type = type_map.get(tid, "Token")
        fill = CHUNK_TEXT_COLORS.get(t_type, CHUNK_TEXT_COLORS["Token"])
        if tok.upos == "PUNCT":
            fill = CHUNK_TEXT_COLORS["Token"]

        spans.append(
            f'<tspan class="hoverable-token" data-lemma="{lemma}" '
            f'data-pos="{tok.upos}" fill="{fill}" dx="{dx}">'
            f'{prefix}{form}'
            f'<title>{lemma} ({tok.upos}) - {short_def}</title>'
            f'</tspan>'
        )
    return "".join(spans)


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL BLOCKS MODE
# ═══════════════════════════════════════════════════════════════════════════════

INDENT_PX = 28

# Subordinate clause types that increment visual depth
_SUBORDINATE_TYPES = frozenset([
    "SubordinateClause", "RelativeClause",
    "AblativeAbsolute", "IndirectStatement",
])


def _map_token_depths(chunk: Chunk, depth: int, out: dict):
    """Recursively map every token_id → (clause_chunk, depth).

    - Sentence/Clause/MainClause: pass through at same depth
    - SubordinateClause/RelativeClause/etc: increment depth
    - Non-clause chunks (NP, PP, VP...): pass through at same depth
    """
    # Determine the depth this chunk introduces
    if chunk.type in _SUBORDINATE_TYPES:
        my_depth = depth + 1
    elif chunk.type in ("Sentence",):
        my_depth = depth
    elif chunk.type in CLAUSE_TYPES:
        # Clause / MainClause — coordinated, same depth
        my_depth = depth
    else:
        # Phrasal chunk (NP, PP, etc.) — no depth change
        my_depth = depth

    # Only clause-level chunks "own" the clause identity for their tokens
    is_clause = chunk.type in CLAUSE_TYPES
    clause_for_tokens = chunk if is_clause else None

    # Record direct tokens
    for tid in chunk.token_ids:
        if clause_for_tokens:
            out[tid] = (clause_for_tokens, my_depth)
        # If not a clause chunk, the token inherits from a parent clause
        # (already set when the parent clause iterated its token_ids)

    # Recurse into children
    for child in chunk.children:
        _map_token_depths(child, my_depth, out)
        # After recursing, if child didn't set clause info for its tokens
        # (because it's a phrasal chunk), those tokens need clause info
        # from the current clause ancestor
        if clause_for_tokens:
            for tid in child.flatten_token_ids():
                if tid not in out:
                    out[tid] = (clause_for_tokens, my_depth)


def _render_full_blocks(chunk: Chunk, tokens: list[Token]) -> str:
    """Render with vertically stacked lines, indented by clause depth.

    Tokens are ALWAYS in reading order.  Each contiguous run of tokens
    at the same clause depth gets its own line.  Subordinate clauses
    are indented proportionally to nesting depth.
    """
    token_map = {t.id: t for t in tokens}
    type_map: dict = {}
    _build_token_type_map(chunk, type_map)

    all_tids = sorted(chunk.flatten_token_ids())
    if not all_tids:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="100"></svg>'

    # ── Step 1: Map every token → (clause, depth) ─────────────────────────
    tid_info: dict = {}  # tid → (clause_chunk, depth)
    _map_token_depths(chunk, 0, tid_info)  # Sentence=0, Clause=0, Sub=1

    # Ensure every token has an entry (fallback to depth 0)
    for tid in all_tids:
        if tid not in tid_info:
            tid_info[tid] = (chunk, 0)

    # ── Step 2: Build contiguous segments (same clause + same depth) ──────
    segments: list[dict] = []
    for tid in all_tids:
        clause, depth = tid_info[tid]
        if (segments
                and segments[-1]["clause"] is clause
                and segments[-1]["depth"] == depth):
            segments[-1]["tids"].append(tid)
        else:
            segments.append({
                "clause": clause,
                "tids": [tid],
                "depth": depth,
            })

    # ── Step 3: Measure text widths ───────────────────────────────────────
    for seg in segments:
        chars = 0
        for i, tid in enumerate(seg["tids"]):
            tok = token_map.get(tid)
            if tok:
                chars += len(tok.form)
                if i > 0:
                    is_punct = tok.upos == "PUNCT" or tok.form in (",", ".", ";", ":", "?", "!", '"', "'")
                    if not is_punct:
                        chars += 1
        seg["text_w"] = chars * CHAR_WIDTH_PX

    # ── Step 4: Vertical layout with depth-based indentation ──────────────
    x_base = 10
    y = 10
    box_h = PAD * 2 + LINE_H
    max_w = 0.0

    for seg in segments:
        indent = max(0, seg["depth"]) * INDENT_PX
        seg["x"] = x_base + indent
        seg["y"] = y
        seg["w"] = seg["text_w"] + PAD * 2
        seg["h"] = box_h
        max_w = max(max_w, seg["x"] + seg["w"])
        y += box_h + GAP

    total_w = max(max_w + 10, 600)
    total_h = y + 10

    # ── Step 5: Draw ──────────────────────────────────────────────────────
    elems: list[str] = []

    for seg in segments:
        depth = max(0, seg["depth"])
        depth_idx = min(depth, len(DEPTH_COLORS) - 1)
        fill, stroke = DEPTH_COLORS[depth_idx]

        # Background
        elems.append(
            f'<rect x="{seg["x"]}" y="{seg["y"]}" '
            f'width="{seg["w"]}" height="{seg["h"]}" '
            f'rx="8" fill="{fill}" fill-opacity="0.20" '
            f'stroke="{stroke}" stroke-opacity="0.5" stroke-width="1.5" />'
        )
        # Text
        tspan_str = _render_token_spans(seg["tids"], token_map, type_map)
        tx = seg["x"] + PAD
        ty = seg["y"] + PAD + 14
        elems.append(
            f'<text x="{tx}" y="{ty}" font-family="sans-serif" '
            f'font-size="16">{tspan_str}</text>'
        )

    svg = "\n".join(elems)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">\n{svg}\n</svg>'
    )



# ═══════════════════════════════════════════════════════════════════════════════
#  CLAUSES-ONLY MODE
# ═══════════════════════════════════════════════════════════════════════════════

class _ClauseNode:
    """Lightweight node for the clauses_only vertical tree."""

    def __init__(self, chunk: Chunk, tokens: list[Token],
                 token_map: dict, type_map: dict, depth: int = 0):
        self.chunk = chunk
        self.token_map = token_map
        self.type_map = type_map
        self.depth = depth
        self.children: list[_ClauseNode] = []

        # Flatten: gather all tids owned by this clause minus sub-clause tids
        all_tids = set(chunk.flatten_token_ids())
        sub_tids: set = set()

        for c in chunk.children:
            node = _ClauseNode(c, tokens, token_map, type_map, depth + 1)
            if c.type in CLAUSE_TYPES and c.type != "Sentence":
                self.children.append(node)
                sub_tids.update(c.flatten_token_ids())
            else:
                # Pull up any nested clauses from phrasal nodes
                self._pull_clauses(node, sub_tids)

        self.direct_tids = sorted(all_tids - sub_tids)

        # Interleave own text segments and child nodes by token position
        items: list = []
        for tid in self.direct_tids:
            items.append(("text", tid, None))
        for child in self.children:
            first_tid = min(child.chunk.flatten_token_ids()) if child.chunk.flatten_token_ids() else float("inf")
            items.append(("child", first_tid, child))
        items.sort(key=lambda x: x[1])

        # Merge consecutive text items into segments
        self.segments: list = []
        for kind, tid, child in items:
            if kind == "text":
                if not self.segments or self.segments[-1][0] != "text":
                    self.segments.append(("text", []))
                self.segments[-1][1].append(tid)
            else:
                self.segments.append(("child", child))

        # Geometry (filled by compute / layout)
        self.w = 0.0
        self.h = 0.0
        self.x = 0.0
        self.y = 0.0

    def _pull_clauses(self, phrasal_node: "_ClauseNode", sub_tids: set):
        for pc in phrasal_node.children:
            if pc.chunk.type in CLAUSE_TYPES and pc.chunk.type != "Sentence":
                self.children.append(pc)
                sub_tids.update(pc.chunk.flatten_token_ids())
            else:
                self._pull_clauses(pc, sub_tids)

    # ── Size ──────────────────────────────────────────────────────────────────

    def compute(self):
        w = 0.0
        h = 0.0
        for kind, payload in self.segments:
            if kind == "text":
                tids = payload
                chars = sum(len(self.token_map[t].form) for t in tids if t in self.token_map)
                tw = (chars + len(tids)) * CHAR_WIDTH_PX + PAD * 2
                th = LINE_H + PAD * 2
                w = max(w, tw)
                h += th + GAP
            else:
                child = payload
                child.compute()
                is_sub = child.chunk.type not in ("MainClause", "Clause", "Sentence")
                indent = 16 if is_sub else 0
                w = max(w, child.w + indent)
                h += child.h + GAP
        if h > 0:
            h -= GAP
        self.w = w
        self.h = h

    # ── Layout ────────────────────────────────────────────────────────────────

    def layout(self, x: float, y: float):
        self.x = x
        self.y = y
        cy = y
        for kind, payload in self.segments:
            if kind == "text":
                tids = payload
                chars = sum(len(self.token_map[t].form) for t in tids if t in self.token_map)
                tw = (chars + len(tids)) * CHAR_WIDTH_PX + PAD * 2
                th = LINE_H + PAD * 2
                # Store geometry on the segment for draw phase
                # (we mutate the tuple's list in-place by replacing the segment)
                idx = self.segments.index(("text", tids))
                self.segments[idx] = ("text_box", {
                    "tids": tids, "x": x, "y": cy, "w": tw, "h": th,
                })
                cy += th + GAP
            else:
                child = payload
                is_sub = child.chunk.type not in ("MainClause", "Clause", "Sentence")
                indent = 16 if is_sub else 0
                child.layout(x + indent, cy)
                cy += child.h + GAP

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, elems: list):
        depth_idx = min(self.depth, len(DEPTH_COLORS) - 1)
        fill, stroke = DEPTH_COLORS[depth_idx]
        label_fill = "rgba(255,255,255,0.6)"

        for kind, payload in self.segments:
            if kind == "text_box":
                bx, by, bw, bh = payload["x"], payload["y"], payload["w"], payload["h"]
                tids = payload["tids"]
                # Box
                if self.chunk.type != "Sentence":
                    elems.append(
                        f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" '
                        f'rx="8" fill="{fill}" fill-opacity="0.25" '
                        f'stroke="{stroke}" stroke-opacity="0.6" stroke-width="1.5" />'
                    )
                # Text
                if tids:
                    tspan = _render_token_spans(tids, self.token_map, self.type_map)
                    tx = bx + PAD
                    ty = by + PAD + 14
                    elems.append(
                        f'<text x="{tx}" y="{ty}" font-family="sans-serif" '
                        f'font-size="16">{tspan}</text>'
                    )
            elif kind == "child":
                child = payload
                child.draw(elems)
                # Dashed connector for subordinate clauses
                if child.chunk.type in ("SubordinateClause", "RelativeClause", "AblativeAbsolute"):
                    lx = child.x - 6
                    elems.insert(0,
                        f'<path d="M {child.x} {child.y + 16} L {lx} {child.y + 16} '
                        f'L {lx} {child.y - 4}" '
                        f'stroke="{label_fill}" stroke-width="1.5" fill="none" '
                        f'stroke-dasharray="3" />'
                    )


def _render_clauses_only(chunk: Chunk, tokens: list[Token]) -> str:
    token_map = {t.id: t for t in tokens}
    type_map: dict = {}
    _build_token_type_map(chunk, type_map)

    # Build the Sentence-level node
    root = _ClauseNode(chunk, tokens, token_map, type_map, depth=0)
    root.compute()
    root.layout(10, 10)

    w = max(root.w + 20, 600)
    h = max(root.h + 20, 100)

    elems: list = []
    root.draw(elems)
    svg = "\n".join(elems)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n{svg}\n</svg>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def render_sentence_map(chunk: Chunk, tokens: list[Token],
                        mode: str = "full_blocks") -> str:
    """Entry point for SVG rendering."""
    try:
        if mode == "full_blocks":
            return _render_full_blocks(chunk, tokens)
        else:
            return _render_clauses_only(chunk, tokens)
    except Exception as e:
        return _render_fallback(chunk, tokens)


def _render_fallback(chunk: Chunk, tokens: list[Token]) -> str:
    """HTML UL/LI fallback if rendering crashes."""
    tok_map = {t.id: t for t in tokens}

    def _html(c: Chunk) -> str:
        html = f'<li style="margin:4px 0; padding:4px 8px; border-left:4px solid #888;">'
        html += f'<strong style="color:white;">{escape(c.label)}</strong> '
        if not c.children:
            words = [escape(tok_map[tid].form) for tid in c.token_ids if tid in tok_map]
            html += f'<span style="color:#d1d5db;">{" ".join(words)}</span>'
        else:
            html += '<ul style="list-style-type:none; padding-left:20px;">'
            for child in c.children:
                html += _html(child)
            html += '</ul>'
        html += '</li>'
        return html

    return (
        '<ul style="list-style-type:none; padding:0; font-family:sans-serif;">'
        + _html(chunk)
        + '</ul>'
    )
