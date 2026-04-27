"""T1.2: Promote advcl -> advcl:pred for predicative participles.

A predicative participle is a participial clause whose head participle
agrees in Case/Number/Gender with a nominal argument of the matrix verb.
In UDante these are consistently labeled advcl:pred.
"""
from __future__ import annotations
from typing import List
from .base import Rule, Change
from ..sentence import Sentence


AGREEMENT_TARGET_DEPRELS = {
    "nsubj", "nsubj:pass", "nsubj:outer",
    "obj",
    "obl", "obl:arg", "obl:agent", "obl:lmod", "obl:tmod",
}


class AdvclPredRule(Rule):
    name = "T1.2_advcl_pred"

    def apply(self, sentence: Sentence) -> List[Change]:
        changes: List[Change] = []
        sent_id = sentence.sent_id() or "?"

        for tok in sentence.tokens:
            if tok.deprel != "advcl":
                continue
            if tok.upos != "VERB":
                continue
            if tok.feats.get("VerbForm") != "Part":
                continue

            t_case = tok.feats.get("Case")
            t_num  = tok.feats.get("Number")
            t_gen  = tok.feats.get("Gender")
            if not (t_case and t_num and t_gen):
                continue

            head = sentence.by_id(tok.head)
            if head is None:
                continue

            # Search siblings (other dependents of head) for an agreeing arg.
            matched = False
            for sib in sentence.children_of(head.id):
                if sib.id == tok.id:
                    continue
                if sib.deprel not in AGREEMENT_TARGET_DEPRELS:
                    continue
                if (sib.feats.get("Case")   == t_case and
                    sib.feats.get("Number") == t_num  and
                    sib.feats.get("Gender") == t_gen):
                    matched = True
                    break

            # Also allow: the participle itself may have a nominal dependent
            # that it agrees with (e.g., "corpus comitatum virtutibus" where
            # 'comitatum' agrees with 'corpus' as subject of main verb).
            # Handled above via sibling search; no extra branch needed.

            if matched:
                changes.append(Change(
                    sent_id=sent_id,
                    token_id=tok.id,
                    token_form=tok.form,
                    field="deprel",
                    old_value=tok.deprel,
                    new_value="advcl:pred",
                    rule_name=self.name,
                ))
                tok.deprel = "advcl:pred"

        return changes