"""T1.3: Promote obl -> obl:agent for agent phrases of passive/deponent verbs.

Triggers:
  - Token is NOUN/PROPN/PRON with deprel == "obl"
  - Case is Ablative
  - Either:
      (a) Has an ADP child with deprel=case whose lemma is in AGENT_PREPOSITIONS
          AND head is a passive verb (Voice=Pass) or a deponent lemma; OR
      (b) No preposition, but head is a past participle (VerbForm=Part,
          Voice=Pass) and token is animate-compatible — we are conservative
          here and require preposition unless the governor is clearly passive
          participial. Default: require preposition.
"""
from __future__ import annotations
from typing import List
from .base import Rule, Change
from ..sentence import Sentence, Token
from ..lexicons import AGENT_PREPOSITIONS, DEPONENT_LEMMAS


def _is_passive_or_deponent(verb: Token) -> bool:
    if verb is None:
        return False
    if verb.feats.get("Voice") == "Pass":
        return True
    if verb.lemma in DEPONENT_LEMMAS:
        return True
    # Past participles in passive meaning: VerbForm=Part and Tense=Past or
    # Aspect=Perf, without Voice=Act.
    if verb.feats.get("VerbForm") == "Part":
        if verb.feats.get("Voice") != "Act":
            return True
    return False


class OblAgentRule(Rule):
    name = "T1.3_obl_agent"

    def apply(self, sentence: Sentence) -> List[Change]:
        changes: List[Change] = []
        sent_id = sentence.sent_id() or "?"

        for tok in sentence.tokens:
            if tok.deprel != "obl":
                continue
            if tok.upos not in {"NOUN", "PROPN", "PRON"}:
                continue
            if tok.feats.get("Case") != "Abl":
                continue

            head = sentence.by_id(tok.head)
            if not _is_passive_or_deponent(head):
                continue

            case_marker = sentence.find_case_marker(tok.id)
            has_agent_prep = (case_marker is not None and
                              case_marker.lemma.lower() in AGENT_PREPOSITIONS)

            # Conservative default: require the a/ab preposition.
            if not has_agent_prep:
                continue

            changes.append(Change(
                sent_id=sent_id,
                token_id=tok.id,
                token_form=tok.form,
                field="deprel",
                old_value=tok.deprel,
                new_value="obl:agent",
                rule_name=self.name,
            ))
            tok.deprel = "obl:agent"

        return changes