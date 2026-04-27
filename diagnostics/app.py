"""
app.py — LatinCy vs. UDante Evaluation App (Streamlit)

Run with:
    bash run_app.sh
or:
    streamlit run app.py
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# CLI argument parsing
# Streamlit forwards everything after -- in `streamlit run app.py -- <args>`
# as sys.argv, so these options can be passed from run_app.sh.
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--udante-path", default=None,
                   help="Path to a .conllu file to pre-load")
    p.add_argument("--model", default="la_core_web_trf",
                   help="LatinCy model name")
    p.add_argument("--max-sents", type=int, default=0,
                   help="Max sentences (0 = all)")
    # parse_known_args so Streamlit's own injected args don't cause errors
    args, _ = p.parse_known_args()
    return args

_cli = _parse_args()

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LatinCy vs. UDante",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — injected via st.markdown for page-level styles only
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] label { color: #c9d1d9 !important; }

.kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 1.25rem 1.75rem; flex: 1; min-width: 180px;
    text-align: center; transition: transform .15s, box-shadow .15s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,.4); }
.kpi-label { font-size:.75rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#8b949e; margin-bottom:.4rem; }
.kpi-value {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1;
}
.kpi-sub { font-size:.7rem; color:#6e7681; margin-top:.3rem; }

h1, h2, h3 { color: #e6edf3 !important; }
p, li { color: #c9d1d9; }

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; color: white !important; width: 100%;
}
.stButton > button:hover { opacity: .9; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Shared CSS string for embedded HTML components (tables)
# ---------------------------------------------------------------------------

_TABLE_CSS = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; font-size: 14px; }
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Phenomena table ── */
.ep-table { width: 100%; border-collapse: collapse; }
.ep-table th {
    text-align: left; font-size: .7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .06em;
    color: #8b949e; padding: .5rem .75rem; border-bottom: 1px solid #30363d;
}
.ep-table td { padding: .55rem .75rem; border-bottom: 1px solid #21262d; color: #c9d1d9; font-size: .85rem; }
.ep-table tr:last-child td { border-bottom: none; }
.bar-wrap { display: flex; align-items: center; gap: .5rem; }
.bar-bg { background: #21262d; border-radius: 4px; height: 8px; flex: 1; overflow: hidden; }
.bar-fill { height: 8px; border-radius: 4px; }
.bar-pct { font-size: .72rem; color: #8b949e; white-space: nowrap; min-width: 40px; }
.na { color: #6e7681; font-style: italic; }

/* ── Token table ── */
.tok-table { width: 100%; border-collapse: collapse; font-size: .78rem; font-family: 'JetBrains Mono', monospace; }
.tok-table th {
    background: #161b22; color: #8b949e; text-transform: uppercase;
    letter-spacing: .06em; font-size: .62rem; padding: .4rem .55rem;
    border-bottom: 1px solid #30363d; text-align: left; position: sticky; top: 0;
}
.tok-table td { padding: .3rem .55rem; border-bottom: 1px solid #21262d; }
.tok-table tr:hover td { background: #161b22; }
.match  { color: #3fb950; }
.miss   { color: #f85149; font-weight: 500; }
.muted  { color: #6e7681; }
.pill {
    display: inline-block; padding: .08rem .4rem; border-radius: 999px;
    font-size: .62rem; font-weight: 600; font-family: 'Inter', sans-serif;
}
.pill-green { background: #12261e; color: #3fb950; border: 1px solid #238636; }
.pill-red   { background: #2c1317; color: #f85149; border: 1px solid #6e1a1a; }
.pill-gold  { background: #2c2007; color: #d29922; border: 1px solid #6e4600; }
.pill-blue  { background: #0c2340; color: #58a6ff; border: 1px solid #1f6feb; }
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    "la_core_web_trf",
    "la_core_web_lg",
    "la_core_web_md",
    "la_core_web_sm",
]


def fmt_pct(value: Optional[float], precision: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{precision}f}%"


def _bar_html(rate: Optional[float], color: str) -> str:
    if rate is None:
        return '<span class="na">n/a</span>'
    pct = rate * 100
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
        f'<span class="bar-pct">{pct:.1f}%</span>'
        f'</div>'
    )


def _pill(match: Optional[bool]) -> str:
    if match is None:
        return '<span class="pill pill-gold">—</span>'
    if match:
        return '<span class="pill pill-green">✓</span>'
    return '<span class="pill pill-red">✗</span>'


def _cell_class(match: Optional[bool]) -> str:
    if match is True:
        return "match"
    if match is False:
        return "miss"
    return "muted"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📜 LatinCy vs. UDante")
    st.caption("Evaluation based on Gamba & Zeman (2023)")
    st.divider()

    # If --udante-path was supplied, show it; file_uploader is still available
    # for overriding via the UI.
    _cli_path = Path(_cli.udante_path) if _cli.udante_path else None
    if _cli_path and not _cli_path.exists():
        st.warning(f"--udante-path not found: {_cli_path}")
        _cli_path = None

    uploaded_file = st.file_uploader(
        "Upload UDante .conllu file",
        type=["conllu", "txt"],
        help="Upload a test or dev set from the UDante treebank.",
    )

    # Resolve which file source to use: UI upload takes priority over CLI path
    _active_file = uploaded_file  # may be overridden below

    if _cli_path and uploaded_file is None:
        st.info(f"Pre-loaded: `{_cli_path.name}`")

    # Model selector — default from --model
    _model_default_idx = AVAILABLE_MODELS.index(_cli.model) \
        if _cli.model in AVAILABLE_MODELS else 0
    model_name = st.selectbox(
        "LatinCy model",
        options=AVAILABLE_MODELS,
        index=_model_default_idx,
        help="The model must be installed in the active virtual environment.",
    )

    # Max sentences — default from --max-sents
    max_sents = st.number_input(
        "Max sentences (0 = all)",
        min_value=0, max_value=10000, value=_cli.max_sents, step=50,
        help="Limit evaluation to the first N sentences for speed.",
    )

    run_btn = st.button("▶  Run Evaluation", use_container_width=True)
    st.divider()
    st.caption(
        "Metrics: POS Accuracy · UAS · LAS\n\n"
        "Targeted: Discourse Particles · Copula Inversion "
        "· Passive Expletives · Compound Numerals"
    )


# ---------------------------------------------------------------------------
# Main area header
# ---------------------------------------------------------------------------

st.markdown("# LatinCy vs. UDante Evaluation")
st.markdown(
    "Systematic evaluation of LatinCy's dependency parsing and POS tagging "
    "against the UDante gold-standard treebank, with targeted checks for the "
    "discrepancies identified in Gamba & Zeman (2023).",
)


# ---------------------------------------------------------------------------
# Run evaluation on button press
# ---------------------------------------------------------------------------

if run_btn:
    # Resolve file source: UI upload > CLI path
    if uploaded_file is not None:
        _active_file = uploaded_file
        _active_file_name = uploaded_file.name
    elif _cli_path is not None:
        _active_file = open(_cli_path, "rb")
        _active_file_name = _cli_path.name
    else:
        _active_file = None
        _active_file_name = ""

    if _active_file is None:
        st.error("Please upload a .conllu file (or pass --udante-path on the command line).")
        st.stop()

    with st.spinner("Loading CoNLL-U file…"):
        try:
            from data_loader import load_conllu
            gold_sentences = load_conllu(_active_file)
        except Exception as e:
            st.error(f"Failed to parse CoNLL-U file: {e}")
            st.stop()

    if max_sents and max_sents > 0:
        gold_sentences = gold_sentences[:max_sents]

    st.info(
        f"Loaded **{len(gold_sentences)} sentences** "
        f"({sum(len(s.tokens) for s in gold_sentences)} tokens) "
        f"from **{_active_file_name}**"
    )

    with st.spinner(f"Loading LatinCy model `{model_name}`… (first load may take a minute)"):
        try:
            from parser import load_model, parse_sentence
            nlp = load_model(model_name)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

    progress_bar = st.progress(0, text="Parsing sentences…")
    pred_sentences = []
    n = len(gold_sentences)
    for i, gs in enumerate(gold_sentences):
        pred_sentences.append(parse_sentence(gs, nlp))
        if i % max(1, n // 100) == 0:
            progress_bar.progress((i + 1) / n, text=f"Parsing sentence {i+1}/{n}…")
    progress_bar.empty()

    from evaluator import evaluate_corpus
    result = evaluate_corpus(gold_sentences, pred_sentences)

    st.session_state["result"] = result
    st.session_state["model_name"] = model_name
    st.success(f"Evaluation complete — {result.total_aligned:,} tokens aligned.")


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

result = st.session_state.get("result", None)

if result is None:
    st.info("Upload a UDante .conllu file, select a model, and click **▶ Run Evaluation** to begin.")
    st.stop()


# ── KPI cards (simple divs — safe for st.markdown) ─────────────────────────

model_used = st.session_state.get("model_name", "—")
st.caption(f"Model: `{model_used}` · {result.total_aligned:,} aligned tokens across {len(result.sentences)} sentences")

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">POS Accuracy</div>
    <div class="kpi-value">{fmt_pct(result.pos_accuracy)}</div>
    <div class="kpi-sub">{result.total_pos_correct:,} / {result.total_aligned:,} tokens</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">UAS</div>
    <div class="kpi-value">{fmt_pct(result.uas)}</div>
    <div class="kpi-sub">Unlabeled Attachment Score</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">LAS</div>
    <div class="kpi-value">{fmt_pct(result.las)}</div>
    <div class="kpi-sub">Labeled Attachment Score</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Sentences</div>
    <div class="kpi-value">{len(result.sentences):,}</div>
    <div class="kpi-sub">{result.total_tokens:,} total tokens</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Targeted Phenomena table — rendered via components.html ────────────────

st.markdown("### Targeted Phenomenon Error Rates")
st.caption("Based on Gamba & Zeman (2023)")

phenomena = [
    ("Discourse Particles",  "PART/discourse → ADV/advmod",   result.discourse_gold,         result.discourse_errors,         result.ep_discourse,         "#f85149"),
    ("Copula Inversion",     "cop (sum) → ROOT",               result.copula_gold,             result.copula_errors,            result.ep_copula,            "#d29922"),
    ("Passive Expletives",   "expl:pass (not in UDante)",      result.passive_expletive_pred,  result.passive_expletive_errors, result.ep_passive_expletive, "#bc8cff"),
    ("Compound Numerals",    "flat → nummod",                  result.numeral_gold,            result.numeral_errors,           result.ep_numeral,           "#58a6ff"),
]

rows_html = ""
for name, desc, gold_n, err_n, ep, color in phenomena:
    rows_html += f"""
    <tr>
      <td><strong>{name}</strong><br><span style="color:#6e7681;font-size:.72rem">{desc}</span></td>
      <td>{gold_n:,}</td>
      <td>{err_n:,}</td>
      <td>{_bar_html(ep, color)}</td>
    </tr>"""

ep_html = f"""{_TABLE_CSS}
<table class="ep-table">
  <thead><tr>
    <th>Phenomenon</th>
    <th>Gold occurrences</th>
    <th>Errors</th>
    <th style="min-width:200px">Error rate E<sub>p</sub></th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""

components.html(ep_html, height=260, scrolling=False)


# ── Sentence Inspector ─────────────────────────────────────────────────────

st.markdown("### Sentence Inspector")

sent_labels = [
    f"[{sr.sent_id}]  {sr.text[:80]}{'…' if len(sr.text) > 80 else ''}"
    for sr in result.sentences
]
selected_idx = st.selectbox(
    "Select sentence",
    options=range(len(sent_labels)),
    format_func=lambda i: sent_labels[i],
    label_visibility="collapsed",
)

sr = result.sentences[selected_idx]

col_l, col_r = st.columns([2, 1])
with col_l:
    st.markdown(f"**{sr.text}**")
with col_r:
    n_err = sum(1 for tr in sr.token_results if tr.pos_match is False or tr.head_match is False)
    n_al = sr.n_aligned
    pos_pct = fmt_pct(sr.n_pos_correct / n_al if n_al else None)
    las_pct = fmt_pct(sr.n_las_correct / n_al if n_al else None)
    st.caption(
        f"**{n_err}** errors · **{n_al}** aligned tokens  \n"
        f"POS acc: **{pos_pct}** · LAS: **{las_pct}**"
    )

# Build token table rows
tok_rows = ""
for tr in sr.token_results:
    g, p = tr.gold, tr.pred
    if g is None and p is None:
        continue

    idx         = g.idx       if g else (p.idx       if p else "—")
    form        = g.form      if g else (p.form      if p else "—")
    gold_upos   = g.upos      if g else "—"
    pred_upos   = p.upos      if p else "—"
    gold_head   = str(g.head) if g else "—"
    pred_head   = str(p.head) if p else "—"
    gold_deprel = g.deprel    if g else "—"
    pred_deprel = p.deprel    if p else "—"

    pos_ok  = tr.pos_match
    head_ok = tr.head_match
    las_ok  = tr.las_match
    pc = _cell_class(pos_ok)
    hc = _cell_class(head_ok)
    lc = _cell_class(las_ok)

    badges = ""
    if tr.is_discourse_particle_error:
        badges += ' <span class="pill pill-red">DISC</span>'
    if tr.is_copula_inversion_error:
        badges += ' <span class="pill pill-gold">COP</span>'
    if tr.is_passive_expletive_error:
        badges += ' <span class="pill pill-red">EXPL</span>'
    if tr.is_numeral_compound_error:
        badges += ' <span class="pill pill-blue">NUM</span>'

    tok_rows += f"""
    <tr>
      <td class="muted">{idx}</td>
      <td><strong>{form}</strong></td>
      <td class="{pc}">{gold_upos}</td>
      <td class="{pc}">{pred_upos}</td>
      <td>{_pill(pos_ok)}</td>
      <td class="{hc}">{gold_head}</td>
      <td class="{hc}">{pred_head}</td>
      <td class="{lc}">{gold_deprel}</td>
      <td class="{lc}">{pred_deprel}</td>
      <td>{_pill(las_ok)}</td>
      <td>{badges}</td>
    </tr>"""

tok_html = f"""{_TABLE_CSS}
<div style="overflow-x:auto">
<table class="tok-table">
  <thead><tr>
    <th>#</th><th>Form</th>
    <th>UDante UPOS</th><th>LatinCy UPOS</th><th>POS</th>
    <th>UDante HEAD</th><th>LatinCy HEAD</th>
    <th>UDante DEPREL</th><th>LatinCy DEPREL</th><th>LAS</th>
    <th>Flags</th>
  </tr></thead>
  <tbody>{tok_rows}</tbody>
</table>
</div>"""

# Height: roughly 32px per token row + 60px header padding
row_count = sum(1 for tr in sr.token_results if not (tr.gold is None and tr.pred is None))
tok_height = min(60 + row_count * 30, 700)
components.html(tok_html, height=tok_height, scrolling=True)


# ── Dependency Error Analysis ───────────────────────────────────────────────

st.markdown("---")
st.markdown("### Dependency Label Error Analysis")
st.caption(
    "All errors across the full corpus, ranked by total error count. "
    "For each UDante DEPREL, shows how often LatinCy disagreed and what it predicted instead."
)

dep_errors = result.deprel_errors(top_n=40)

if not dep_errors:
    st.info("No deprel errors found — perfect prediction!")
else:
    _DEP_CSS = """
    <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; font-size: 13px; }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');

    table { width: 100%; border-collapse: collapse; }
    th {
        text-align: left; font-size: .65rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: .07em;
        color: #8b949e; padding: .45rem .65rem;
        border-bottom: 1px solid #30363d;
        background: #161b22; position: sticky; top: 0;
    }
    td { padding: .4rem .65rem; border-bottom: 1px solid #21262d; vertical-align: middle; }
    tr:hover td { background: #161b22; }
    tr:last-child td { border-bottom: none; }

    .deprel { font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: #e6edf3; font-weight: 500; }
    .num    { font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: #8b949e; }
    .err    { font-family: 'JetBrains Mono', monospace; font-size: .78rem; color: #f85149; font-weight: 600; }

    .bar-wrap { display: flex; align-items: center; gap: .4rem; }
    .bar-bg { background: #21262d; border-radius: 4px; height: 7px; width: 90px; overflow: hidden; flex-shrink: 0; }
    .bar-fill { height: 7px; border-radius: 4px; background: #f85149; }
    .bar-pct { font-size: .7rem; color: #8b949e; white-space: nowrap; }

    .confusion-list { display: flex; flex-wrap: wrap; gap: .3rem; }
    .conf-item {
        background: #161b22; border: 1px solid #30363d; border-radius: 6px;
        padding: .15rem .45rem; font-size: .7rem;
        font-family: 'JetBrains Mono', monospace;
        color: #bc8cff;
        white-space: nowrap;
    }
    .conf-count { color: #6e7681; font-size: .65rem; margin-left: .2rem; }
    </style>"""

    dep_rows = ""
    for row in dep_errors:
        pct = row["error_rate"] * 100
        bar_w = min(pct, 100)
        confusions_html = ""
        for pred_dep, cnt in row["top_confusions"]:
            confusions_html += (
                f'<span class="conf-item">{pred_dep}'
                f'<span class="conf-count">\u00d7{cnt}</span></span>'
            )

        dep_rows += (
            f"<tr>"
            f'<td class="deprel">{row["udante_deprel"]}</td>'
            f'<td class="num">{row["total"]:,}</td>'
            f'<td class="err">{row["errors"]:,}</td>'
            f"<td>"
            f'<div class="bar-wrap">'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{bar_w:.1f}%"></div></div>'
            f'<span class="bar-pct">{pct:.1f}%</span>'
            f"</div></td>"
            f'<td><div class="confusion-list">{confusions_html}</div></td>'
            f"</tr>"
        )

    dep_html = (
        _DEP_CSS
        + '<div style="overflow-x:auto"><table>'
        + "<thead><tr>"
        + "<th>UDante DEPREL</th>"
        + "<th>Total occurrences</th>"
        + "<th>Errors</th>"
        + '<th style="min-width:160px">Error rate</th>'
        + "<th>Top LatinCy predictions (when wrong)</th>"
        + "</tr></thead>"
        + f"<tbody>{dep_rows}</tbody>"
        + "</table></div>"
    )

    dep_height = min(80 + len(dep_errors) * 36, 800)
    components.html(dep_html, height=dep_height, scrolling=True)


# ---------------------------------------------------------------------------
# CLI: python app.py diagnose --udante-path <path> --output-dir <dir>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser_cli = argparse.ArgumentParser(
        description="LatinCy vs. UDante Evaluation — CLI diagnostic mode"
    )
    sub = parser_cli.add_subparsers(dest="command")

    diag = sub.add_parser("diagnose", help="Run full diagnostic report")
    diag.add_argument("--udante-path", required=True,
                      help="Path to UDante .conllu file(s). Single file or glob.")
    diag.add_argument("--output-dir", required=True,
                      help="Directory for report output files.")
    diag.add_argument("--model", default="la_core_web_trf",
                      help="LatinCy model name (default: la_core_web_trf)")
    diag.add_argument("--max-sents", type=int, default=0,
                      help="Max sentences to evaluate (0 = all)")
    diag.add_argument("--seed", type=int, default=42,
                      help="Random seed for sampling (default: 42)")

    args = parser_cli.parse_args()

    if args.command != "diagnose":
        parser_cli.print_help()
        sys.exit(1)

    # Import heavy modules only in CLI mode
    from data_loader import load_conllu
    from parser import load_model, parse_sentence
    from evaluator import evaluate_corpus
    from reporter import run_full_report

    # Load data
    print(f"Loading UDante data from: {args.udante_path}")
    import glob
    paths = glob.glob(args.udante_path)
    if not paths:
        # Try as literal path
        paths = [args.udante_path]

    all_gold = []
    for p in sorted(paths):
        print(f"  Reading {p}...")
        all_gold.extend(load_conllu(p))

    if not all_gold:
        print("ERROR: No sentences loaded.", file=sys.stderr)
        sys.exit(1)

    if args.max_sents and args.max_sents > 0:
        all_gold = all_gold[:args.max_sents]

    print(f"Loaded {len(all_gold)} sentences "
          f"({sum(len(s.tokens) for s in all_gold)} tokens)")

    # Load model
    print(f"Loading LatinCy model: {args.model}")
    try:
        nlp = load_model(args.model)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse
    print("Parsing sentences...")
    pred_sentences = []
    n = len(all_gold)
    failed = 0
    for i, gs in enumerate(all_gold):
        try:
            pred_sentences.append(parse_sentence(gs, nlp))
        except Exception as e:
            failed += 1
            pred_sentences.append([])  # empty prediction
            if i < 5:
                print(f"  WARNING: sentence {gs.sent_id} failed: {e}")
        if (i + 1) % max(1, n // 20) == 0:
            print(f"  {i+1}/{n} ({(i+1)/n*100:.0f}%)")

    fail_rate = failed / n if n else 0
    if fail_rate > 0.05:
        print(f"ERROR: {failed}/{n} sentences ({fail_rate*100:.1f}%) failed "
              f"(threshold: 5%)", file=sys.stderr)
        sys.exit(2)

    # Evaluate
    print("Evaluating...")
    result = evaluate_corpus(all_gold, pred_sentences)

    # Generate report
    print(f"Generating report in: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    summary_md = run_full_report(result, args.output_dir, seed=args.seed)

    print("\n" + "=" * 60)
    print(summary_md)
    print("=" * 60)
    print(f"\nAll report files written to: {args.output_dir}")
    sys.exit(0)

