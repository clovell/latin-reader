"""
data_loader.py — CoNLL-U ingestion for LatinCy vs. UDante evaluation.

Parses standard .conllu files (UDante test/dev sets) and returns
structured sentence objects ready for evaluation.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import conllu


@dataclass
class GoldToken:
    """A single token from the gold-standard CoNLL-U file."""
    idx: int          # 1-based position in sentence
    form: str         # Surface form
    lemma: str
    upos: str         # Universal POS tag
    head: int         # 0-based head index (0 = root)
    deprel: str       # Dependency relation
    xpos: str = "_"   # Language-specific POS tag
    feats: Optional[Dict[str, str]] = None  # Morphological features
    misc: Optional[Dict[str, str]] = None   # Misc column data
    space_after: bool = True  # Whether a space follows this token


@dataclass
class GoldSentence:
    """A sentence from the gold-standard CoNLL-U file."""
    sent_id: str
    text: str          # Full sentence text (from # text = comment, or reconstructed)
    tokens: List[GoldToken] = field(default_factory=list)

    def raw_text(self) -> str:
        """Reconstruct the raw sentence string respecting SpaceAfter."""
        parts = []
        for tok in self.tokens:
            parts.append(tok.form)
            if tok.space_after:
                parts.append(" ")
        return "".join(parts).strip()


def _space_after(token_data) -> bool:
    """Extract SpaceAfter=No from misc field; default True."""
    misc = token_data.get("misc")
    if misc and isinstance(misc, dict):
        return misc.get("SpaceAfter", "Yes") != "No"
    return True


def load_conllu(source) -> List[GoldSentence]:
    """
    Load a CoNLL-U file and return a list of GoldSentence objects.

    Args:
        source: file path (str), or a file-like object, or raw string content.

    Returns:
        List of GoldSentence objects.
    """
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        sentences_raw = conllu.parse(raw)
    elif isinstance(source, str):
        try:
            with open(source, encoding="utf-8") as f:
                sentences_raw = conllu.parse(f.read())
        except (OSError, FileNotFoundError):
            # Treat as raw CoNLL-U string content
            sentences_raw = conllu.parse(source)
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    sentences: List[GoldSentence] = []
    for sent in sentences_raw:
        # Extract metadata
        sent_id = sent.metadata.get("sent_id", f"sent_{len(sentences) + 1}")
        text_comment = sent.metadata.get("text", "")

        tokens: List[GoldToken] = []
        for tok in sent:
            # Skip multi-word tokens (e.g. "1-2") and empty nodes ("1.1")
            tok_id = tok["id"]
            if not isinstance(tok_id, int):
                continue

            deprel = tok.get("deprel") or "_"
            # Normalise root: CoNLL-U uses HEAD=0 + deprel=root
            head_raw = tok.get("head") or 0

            # Parse morphological features dict
            raw_feats = tok.get("feats")
            feats_dict = dict(raw_feats) if raw_feats and isinstance(raw_feats, dict) else None

            # Parse misc dict
            raw_misc = tok.get("misc")
            misc_dict = dict(raw_misc) if raw_misc and isinstance(raw_misc, dict) else None

            tokens.append(GoldToken(
                idx=tok_id,
                form=tok.get("form") or "_",
                lemma=tok.get("lemma") or "_",
                upos=tok.get("upos") or "_",
                head=head_raw,
                deprel=deprel.lower(),
                xpos=tok.get("xpos") or "_",
                feats=feats_dict,
                misc=misc_dict,
                space_after=_space_after(tok),
            ))

        gs = GoldSentence(
            sent_id=sent_id,
            text=text_comment,
            tokens=tokens,
        )
        # Use reconstructed text if comment was missing
        if not gs.text:
            gs.text = gs.raw_text()
        sentences.append(gs)

    return sentences
