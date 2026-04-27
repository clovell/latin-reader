from __future__ import annotations
from dataclasses import dataclass
from typing import List
from ..sentence import Sentence, Token


@dataclass
class Change:
    sent_id: str
    token_id: int
    token_form: str
    field: str           # "deprel" | "head"
    old_value: str
    new_value: str
    rule_name: str


class Rule:
    name: str = "base"

    def apply(self, sentence: Sentence) -> List[Change]:
        raise NotImplementedError