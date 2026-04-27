"""Minimal data model for CoNLL-U sentences."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Token:
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: Dict[str, str] = field(default_factory=dict)
    head: int = 0
    deprel: str = "_"
    deps: str = "_"
    misc: str = "_"

    @property
    def base_deprel(self) -> str:
        return self.deprel.split(":", 1)[0]

    @property
    def subtype(self) -> Optional[str]:
        parts = self.deprel.split(":", 1)
        return parts[1] if len(parts) == 2 else None

    def agrees_with(self, other: "Token", features=("Case", "Number", "Gender")) -> bool:
        """True if all listed features are present on both tokens and match."""
        for f in features:
            a, b = self.feats.get(f), other.feats.get(f)
            if a is None or b is None or a != b:
                return False
        return True


@dataclass
class Sentence:
    tokens: List[Token] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)

    def by_id(self, tok_id: int) -> Optional[Token]:
        if tok_id <= 0 or tok_id > len(self.tokens):
            return None
        tok = self.tokens[tok_id - 1]
        assert tok.id == tok_id, f"ID mismatch: expected {tok_id}, got {tok.id}"
        return tok

    def children_of(self, tok_id: int) -> List[Token]:
        return [t for t in self.tokens if t.head == tok_id]

    def find_case_marker(self, tok_id: int) -> Optional[Token]:
        """Return the ADP child with deprel=case (if any)."""
        for c in self.children_of(tok_id):
            if c.deprel == "case" and c.upos == "ADP":
                return c
        return None

    def sentence_text(self) -> str:
        for c in self.comments:
            if c.startswith("# text ="):
                return c.split("=", 1)[1].strip()
        return " ".join(t.form for t in self.tokens)

    def sent_id(self) -> Optional[str]:
        for c in self.comments:
            if c.startswith("# sent_id ="):
                return c.split("=", 1)[1].strip()
        return None