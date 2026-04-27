"""
parser.py — LatinCy model wrapper for evaluation.

Provides load_model() and parse_sentence() with a simple positional
alignment strategy: run spaCy on the reconstructed sentence text, then
match output tokens 1-to-1 with gold tokens by position. Mismatches in
tokenisation count are flagged but evaluation continues on the overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import spacy
from spacy.tokens import Doc
from spacy.language import Language

from data_loader import GoldSentence, GoldToken

# Register a passthrough stub for uv_normalizer. The real implementation lives
# in the latincy-preprocess package which requires Python >=3.10. On Python 3.9
# we register a no-op so the model loads cleanly.
# Use try/except rather than a presence check — Streamlit hot-reloads can cause
# the registry state to be inconsistent between module re-imports.
try:
    @Language.factory("uv_normalizer", default_config={"method": "rules", "overwrite": False})
    def _create_uv_normalizer(nlp, name, method, overwrite):
        return lambda doc: doc
except Exception:
    pass  # Already registered — safe to ignore


# Module-level model cache
_model_cache: Dict[str, spacy.Language] = {}

# Forms of Latin *sum* (esse) for copula checks
SUM_FORMS = frozenset({
    "sum", "es", "est", "sumus", "estis", "sunt",
    "eram", "eras", "erat", "eramus", "eratis", "erant",
    "ero", "eris", "erit", "erimus", "eritis", "erunt",
    "fui", "fuisti", "fuit", "fuimus", "fuistis", "fuerunt", "fuere",
    "fueram", "fueras", "fuerat", "fueramus", "fueratis", "fuerant",
    "fuero", "fueris", "fuerit", "fuerimus", "fueritis", "fuerint",
    "esse", "fuisse", "fore",
})


@dataclass
class PredToken:
    """A single predicted token from LatinCy."""
    idx: int       # 1-based position
    form: str
    upos: str
    head: int      # 1-based head index (0 = root)
    deprel: str
    lemma: str = "_"
    xpos: str = "_"
    feats: Optional[Dict[str, str]] = None


def load_model(model_name: str) -> spacy.Language:
    """Load (and cache) a spaCy / LatinCy model by name."""
    if model_name not in _model_cache:
        try:
            # Exclude lookup_lemmatizer: requires la_latincy_lookups which needs
            # Python >=3.10. The trainable_lemmatizer still provides good lemmas.
            nlp = spacy.load(model_name, exclude=["lookup_lemmatizer"])
        except OSError as e:
            raise RuntimeError(
                f"Could not load model '{model_name}'. "
                "Make sure it is installed in the active virtual environment.\n"
                f"Original error: {e}"
            ) from e
        # Disable components we don't need for speed (keep parser + tagger)
        _model_cache[model_name] = nlp
    return _model_cache[model_name]


def _spacy_doc_to_pred_tokens(doc: Doc) -> List[PredToken]:
    """Convert a spaCy Doc into a list of PredToken objects (1-based idx)."""
    pred: List[PredToken] = []
    for i, tok in enumerate(doc, start=1):
        # spaCy head is absolute token index (0-based); convert to 1-based.
        # If a token is its own head, it is the root (head = 0 in CoNLL-U).
        if tok.head.i == tok.i:
            head_idx = 0
        else:
            head_idx = tok.head.i + 1  # 0-based → 1-based

        # Extract morphological features from spaCy
        feats_dict = None
        if tok.morph and str(tok.morph):
            feats_dict = {}
            for feat_str in str(tok.morph).split("|"):
                if "=" in feat_str:
                    k, v = feat_str.split("=", 1)
                    feats_dict[k] = v

        pred.append(PredToken(
            idx=i,
            form=tok.text,
            upos=tok.pos_,
            head=head_idx,
            deprel=tok.dep_.lower(),
            lemma=tok.lemma_,
            xpos=tok.tag_ if tok.tag_ else "_",
            feats=feats_dict,
        ))
    return pred


def parse_sentence(gold_sent: GoldSentence, nlp: spacy.Language) -> List[PredToken]:
    """
    Parse a gold sentence and return aligned PredToken list.

    Uses the reconstructed raw text.  If tokenisation count differs
    from gold, returns only the overlapping prefix (shorter of the two).
    """
    raw = gold_sent.raw_text()
    doc = nlp(raw)
    return _spacy_doc_to_pred_tokens(doc)


def align(gold_tokens: List[GoldToken], pred_tokens: List[PredToken]):
    """
    Zip gold and predicted tokens by position.

    Returns a list of (gold, pred | None) tuples.  If lengths differ,
    the shorter list determines the range; excess tokens from the longer
    list are returned paired with None.
    """
    max_len = max(len(gold_tokens), len(pred_tokens))
    pairs = []
    for i in range(max_len):
        g = gold_tokens[i] if i < len(gold_tokens) else None
        p = pred_tokens[i] if i < len(pred_tokens) else None
        pairs.append((g, p))
    return pairs


def _compute_char_spans(tokens, get_form, get_space_after):
    """Compute (start, end) character spans for a list of tokens."""
    spans = []
    offset = 0
    for tok in tokens:
        form = get_form(tok)
        start = offset
        end = offset + len(form)
        spans.append((start, end))
        offset = end
        if get_space_after(tok):
            offset += 1
    return spans


def align_by_char_offset(
    gold_tokens: List[GoldToken],
    pred_tokens: List[PredToken],
    gold_text: str,
) -> Tuple[
    List[Tuple[Optional[GoldToken], Optional[PredToken]]],
    List[dict],
]:
    """
    Align gold and predicted tokens using character-offset matching (§6.2).

    Returns:
        pairs: list of (gold, pred) tuples. Exactly-aligned tokens are paired;
               unaligned tokens are paired with None.
        divergences: list of dicts describing split/merge/mismatch cases.
    """
    # Compute gold spans from form + space_after
    gold_spans = _compute_char_spans(
        gold_tokens,
        get_form=lambda t: t.form,
        get_space_after=lambda t: t.space_after,
    )

    # For pred tokens, we reconstruct spans from the raw text by finding
    # each pred token form in order
    pred_spans = []
    offset = 0
    for pt in pred_tokens:
        # Find the pred token text starting from current offset
        pos = gold_text.find(pt.form, offset)
        if pos == -1:
            # Fallback: try case-insensitive or just advance
            pos = offset
        start = pos
        end = pos + len(pt.form)
        pred_spans.append((start, end))
        offset = end

    # Build alignment via exact span matching
    pairs: List[Tuple[Optional[GoldToken], Optional[PredToken]]] = []
    divergences: List[dict] = []
    g_used = set()
    p_used = set()

    # First pass: exact matches
    p_span_map = {}
    for pi, (ps, pe) in enumerate(pred_spans):
        p_span_map.setdefault((ps, pe), []).append(pi)

    for gi, (gs, ge) in enumerate(gold_spans):
        key = (gs, ge)
        if key in p_span_map and p_span_map[key]:
            pi = p_span_map[key].pop(0)
            pairs.append((gold_tokens[gi], pred_tokens[pi]))
            g_used.add(gi)
            p_used.add(pi)

    # Second pass: unmatched tokens
    for gi, gtok in enumerate(gold_tokens):
        if gi not in g_used:
            gs, ge = gold_spans[gi]
            # Check for split: one gold → many pred
            covering_preds = [
                pi for pi, (ps, pe) in enumerate(pred_spans)
                if pi not in p_used and ps >= gs and pe <= ge
            ]
            if covering_preds:
                divergences.append({
                    "type": "split",
                    "gold_form": gtok.form,
                    "gold_span": (gs, ge),
                    "pred_forms": [pred_tokens[pi].form for pi in covering_preds],
                })
                for pi in covering_preds:
                    p_used.add(pi)
            else:
                divergences.append({
                    "type": "mismatched",
                    "gold_form": gtok.form,
                    "gold_span": (gs, ge),
                })
            # Pair gold with None
            pairs.append((gtok, None))

    for pi, ptok in enumerate(pred_tokens):
        if pi not in p_used:
            ps, pe = pred_spans[pi]
            # Check for merge: many gold → one pred
            covering_golds = [
                gi for gi, (gs, ge) in enumerate(gold_spans)
                if gi not in g_used and gs >= ps and ge <= pe
            ]
            if covering_golds:
                divergences.append({
                    "type": "merged",
                    "pred_form": ptok.form,
                    "pred_span": (ps, pe),
                    "gold_forms": [gold_tokens[gi].form for gi in covering_golds],
                })
            else:
                divergences.append({
                    "type": "mismatched",
                    "pred_form": ptok.form,
                    "pred_span": (ps, pe),
                })
            pairs.append((None, ptok))

    # Sort pairs by the position of whichever side exists
    def sort_key(pair):
        g, p = pair
        if g is not None:
            return g.idx
        if p is not None:
            return p.idx + 0.5  # interleave after same-position gold
        return 9999
    pairs.sort(key=sort_key)

    return pairs, divergences
