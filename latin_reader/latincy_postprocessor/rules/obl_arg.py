"""T1.1 (v2): Promote obl -> obl:arg using layered, data-grounded rules.

Rule A (structural, high confidence):
    If head is ADJ or (NOUN in predicative construction) and dependent is
    NOUN/PRON/DET in Dat, promote. Also handle known case-governing
    adjectives with Abl/Gen.

Rule B (lexical verb + dative, high confidence):
    If head verb lemma is in DATIVE_ARG_VERBS, dependent is Dat, no
    preposition, promote.

Rule C (framed verb + prep + case, medium confidence):
    If head verb lemma is in the frame lexicon for the observed
    (preposition, case) combination, promote.

Rule D (bare ablative, medium confidence):
    If head lemma is in BARE_ABL_ARG_LEMMAS, dependent is Abl, no
    preposition, promote.

Deliberate non-firings:
    - Locative/directional adjuncts with unknown verbs: keep 'obl'.
    - Any case where the dependent has no Case feature: keep 'obl'.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
from .base import Rule, Change
from ..sentence import Sentence, Token
from ..lexicons import (
    DATIVE_ARG_VERBS,
    CASE_GOVERNING_ADJECTIVES,
    CASE_ADJ_GEN,
    OBL_ARG_VERB_FRAMES,
    BARE_ABL_ARG_LEMMAS,
)


NOMINAL_UPOS = {"NOUN", "PROPN", "PRON", "DET", "ADJ"}


def _frame_key(prep_lemma: Optional[str], case: Optional[str]) -> Optional[str]:
    """Return the OBL_ARG_VERB_FRAMES key for (prep, case), or None."""
    if prep_lemma is None:
        return None
    prep = prep_lemma.lower()
    if prep in {"a", "ab", "abs"} and case == "Abl":
        return "ab_abl"
    if prep == "ad" and case == "Acc":
        return "ad_acc"
    if prep == "in" and case == "Acc":
        return "in_acc"
    if prep == "in" and case == "Abl":
        return "in_abl"
    if prep == "de" and case == "Abl":
        return "de_abl"
    if prep == "pro" and case == "Abl":
        return "pro_abl"
    if prep == "cum" and case == "Abl":
        return "cum_abl"
    return None


def _why_obl_arg(head: Token,
                 prep_lemma: Optional[str],
                 token_case: Optional[str]) -> Optional[str]:
    """Return a short rule tag if this token should become obl:arg, else None."""
    if head is None or token_case is None:
        return None

    head_lemma = head.lemma
    head_upos = head.upos

    # --- Rule A: ADJ head governing a case argument ---------------------
    if head_upos == "ADJ":
        if token_case == "Dat":
            # Any dative under an ADJ is taken as argument.
            # (Risk: rare dative of reference, but UDante treats these as obl:arg.)
            return "A_adj_dat"
        if token_case == "Abl" and head_lemma in CASE_GOVERNING_ADJECTIVES:
            return "A_adj_abl_lex"
        if token_case == "Gen" and (head_lemma in CASE_ADJ_GEN
                                    or head_lemma in CASE_GOVERNING_ADJECTIVES):
            return "A_adj_gen_lex"
        return None

    # --- Rule B: verb + dative, lexical ---------------------------------
    if head_upos in {"VERB", "AUX"} and token_case == "Dat" and prep_lemma is None:
        if head_lemma in DATIVE_ARG_VERBS:
            return "B_verb_dat_lex"
        return None

    # --- Rule C: verb + preposition + case, lexical ---------------------
    if head_upos in {"VERB", "AUX"} and prep_lemma is not None:
        key = _frame_key(prep_lemma, token_case)
        if key is None:
            return None
        verbs_in_frame = OBL_ARG_VERB_FRAMES.get(key, set())
        if head_lemma in verbs_in_frame:
            return f"C_frame_{key}"
        return None

    # --- Rule D: bare ablative argument ---------------------------------
    if (head_upos in {"VERB", "AUX", "ADJ"}
            and token_case == "Abl"
            and prep_lemma is None
            and head_lemma in BARE_ABL_ARG_LEMMAS):
        return "D_bare_abl_lex"

    return None


class OblArgRule(Rule):
    name = "T1.1_obl_arg"

    def apply(self, sentence: Sentence) -> List[Change]:
        changes: List[Change] = []
        sent_id = sentence.sent_id() or "?"

        for tok in sentence.tokens:
            if tok.deprel != "obl":
                continue
            if tok.upos not in NOMINAL_UPOS:
                continue

            head = sentence.by_id(tok.head)
            if head is None:
                continue

            case_marker = sentence.find_case_marker(tok.id)
            prep_lemma = case_marker.lemma.lower() if case_marker else None
            token_case = tok.feats.get("Case")

            reason = _why_obl_arg(head, prep_lemma, token_case)
            if reason is None:
                continue

            changes.append(Change(
                sent_id=sent_id,
                token_id=tok.id,
                token_form=tok.form,
                field="deprel",
                old_value=tok.deprel,
                new_value="obl:arg",
                rule_name=f"{self.name}:{reason}",
            ))
            tok.deprel = "obl:arg"

        return changes