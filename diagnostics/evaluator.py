"""
evaluator.py — Token-level evaluation metrics for LatinCy vs. UDante.

Implements:
  - POS / XPOS / UFeats / Lemma Accuracy
  - UAS (Unlabeled Attachment Score)
  - LAS (Labeled Attachment Score) with and without subtypes
  - E_p error rates for four targeted Gamba & Zeman (2023) phenomena
  - Confusion matrices for deprel, UPOS, head attachment
  - Per-subcorpus breakdowns and tokenization statistics
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from data_loader import GoldSentence, GoldToken
from parser import PredToken, SUM_FORMS, align


# ---------------------------------------------------------------------------
# Per-token result
# ---------------------------------------------------------------------------

# Features used for partial UFeats F1
UFEATS_KEYS = ("Case", "Number", "Gender", "Tense", "Mood", "Voice",
               "Person", "VerbForm", "Degree")


def _strip_subtype(deprel: str) -> str:
    """advmod:lmod -> advmod"""
    return deprel.split(":")[0]


def _normalize_lemma(lemma: str, normalize_jv: bool = False) -> str:
    """Case-insensitive lemma comparison, optionally j->i and v->u."""
    s = lemma.lower()
    if normalize_jv:
        s = s.replace("j", "i").replace("v", "u")
    return s


@dataclass
class TokenResult:
    """Comparison result for a single (gold, pred) token pair."""
    gold: Optional[GoldToken]
    pred: Optional[PredToken]

    # Core match flags (None when either side is missing)
    pos_match: Optional[bool] = None
    head_match: Optional[bool] = None
    las_match: Optional[bool] = None        # head AND base deprel correct
    las_full_match: Optional[bool] = None   # head AND full deprel correct
    deprel_base_match: Optional[bool] = None
    deprel_full_match: Optional[bool] = None
    xpos_match: Optional[bool] = None
    ufeats_match: Optional[bool] = None
    lemma_match: Optional[bool] = None

    # Targeted phenomenon flags (True = this token triggered the phenomenon)
    is_discourse_particle_error: bool = False   # gold PART/discourse, pred ADV/advmod
    is_copula_inversion_error: bool = False     # gold cop, pred ROOT for *sum*
    is_passive_expletive_error: bool = False    # pred expl:pass
    is_numeral_compound_error: bool = False     # gold flat, pred nummod


# ---------------------------------------------------------------------------
# Per-sentence result
# ---------------------------------------------------------------------------

@dataclass
class SentenceResult:
    sent_id: str
    text: str
    source_file: str = ""  # sub-corpus name extracted from sent_id
    token_results: List[TokenResult] = field(default_factory=list)

    # Aggregate counts for this sentence
    n_tokens: int = 0
    n_gold_tokens: int = 0
    n_pred_tokens: int = 0
    n_pos_correct: int = 0
    n_head_correct: int = 0
    n_las_correct: int = 0
    n_las_full_correct: int = 0
    n_xpos_correct: int = 0
    n_ufeats_correct: int = 0
    n_lemma_correct: int = 0
    n_aligned: int = 0  # tokens where both gold and pred exist
    has_tokenization_mismatch: bool = False

    # Targeted phenomenon counts
    n_discourse_gold: int = 0     # gold occurrences of the phenomenon
    n_discourse_errors: int = 0
    n_copula_gold: int = 0
    n_copula_errors: int = 0
    n_passive_expletive_pred: int = 0   # counts pred occurrences
    n_passive_expletive_errors: int = 0
    n_numeral_gold: int = 0
    n_numeral_errors: int = 0


# ---------------------------------------------------------------------------
# Global result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    sentences: List[SentenceResult] = field(default_factory=list)

    # Aggregate totals
    total_tokens: int = 0
    total_gold_tokens: int = 0
    total_pred_tokens: int = 0
    total_aligned: int = 0
    total_pos_correct: int = 0
    total_head_correct: int = 0
    total_las_correct: int = 0
    total_las_full_correct: int = 0
    total_xpos_correct: int = 0
    total_ufeats_correct: int = 0
    total_lemma_correct: int = 0
    total_excluded: int = 0  # tokens excluded due to alignment failure
    sents_with_tok_mismatch: int = 0

    # Targeted phenomena totals
    discourse_gold: int = 0
    discourse_errors: int = 0
    copula_gold: int = 0
    copula_errors: int = 0
    passive_expletive_pred: int = 0
    passive_expletive_errors: int = 0
    numeral_gold: int = 0
    numeral_errors: int = 0

    # Confusion matrices
    deprel_confusion: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    deprel_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    deprel_confusion_base: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    deprel_total_base: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    upos_confusion: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    upos_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Head attachment error matrix: (pred_head_upos, gold_head_upos) -> count
    head_confusion: Counter = field(default_factory=Counter)

    # Per-subcorpus results: {subcorpus_name: EvalResult}
    per_subcorpus: Dict[str, dict] = field(default_factory=dict)

    # UFeats partial: accumulated per-feature TP/FP/FN
    ufeats_per_feature: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        k: {"tp": 0, "fp": 0, "fn": 0} for k in UFEATS_KEYS
    })

    # Tokenization divergences for §6
    tokenization_divergences: List[dict] = field(default_factory=list)

    def deprel_errors(self, top_n: int = 30):
        """Return a list of dicts summarising deprel errors, sorted by error count desc."""
        rows = []
        for deprel, total in self.deprel_total.items():
            confusion = self.deprel_confusion.get(deprel, Counter())
            n_errors = sum(confusion.values())
            if n_errors == 0:
                continue
            rows.append({
                "udante_deprel": deprel,
                "total": total,
                "errors": n_errors,
                "error_rate": n_errors / total if total else 0.0,
                "top_confusions": confusion.most_common(3),
            })
        rows.sort(key=lambda r: r["errors"], reverse=True)
        return rows[:top_n]

    # -----------------------------------------------------------------------
    # Derived metrics
    # -----------------------------------------------------------------------

    def _safe_div(self, num: int, den: int) -> Optional[float]:
        return num / den if den else None

    @property
    def pos_accuracy(self) -> Optional[float]:
        return self._safe_div(self.total_pos_correct, self.total_aligned)

    @property
    def uas(self) -> Optional[float]:
        return self._safe_div(self.total_head_correct, self.total_aligned)

    @property
    def las(self) -> Optional[float]:
        return self._safe_div(self.total_las_correct, self.total_aligned)

    @property
    def las_with_subtypes(self) -> Optional[float]:
        return self._safe_div(self.total_las_full_correct, self.total_aligned)

    @property
    def xpos_accuracy(self) -> Optional[float]:
        return self._safe_div(self.total_xpos_correct, self.total_aligned)

    @property
    def ufeats_accuracy(self) -> Optional[float]:
        return self._safe_div(self.total_ufeats_correct, self.total_aligned)

    @property
    def lemma_accuracy(self) -> Optional[float]:
        return self._safe_div(self.total_lemma_correct, self.total_aligned)

    @property
    def ufeats_partial_f1(self) -> Optional[float]:
        """Average per-feature F1 across UFEATS_KEYS."""
        f1s = []
        for k in UFEATS_KEYS:
            d = self.ufeats_per_feature[k]
            tp, fp, fn = d["tp"], d["fp"], d["fn"]
            if tp + fp + fn == 0:
                continue
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            f1s.append(f1)
        return sum(f1s) / len(f1s) if f1s else None

    def ep(self, gold_count: int, error_count: int) -> Optional[float]:
        if gold_count == 0:
            return None
        return error_count / gold_count

    @property
    def ep_discourse(self) -> Optional[float]:
        return self.ep(self.discourse_gold, self.discourse_errors)

    @property
    def ep_copula(self) -> Optional[float]:
        return self.ep(self.copula_gold, self.copula_errors)

    @property
    def ep_passive_expletive(self) -> Optional[float]:
        """E_p for passive expletives: proportion of pred expl:pass that are errors
        (all pred expl:pass are counted as errors — UDante doesn't use this label)."""
        if self.passive_expletive_pred == 0:
            return None
        return self.passive_expletive_errors / self.passive_expletive_pred

    @property
    def ep_numeral(self) -> Optional[float]:
        return self.ep(self.numeral_gold, self.numeral_errors)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def _check_discourse_particle(gold: GoldToken, pred: PredToken) -> Tuple[int, int]:
    """Return (is_gold_phenomenon, is_error).

    Discourse particle: gold UPOS=PART or gold DEPREL=discourse,
    but LatinCy predicts UPOS=ADV or DEPREL=advmod.
    """
    gold_is_discourse = (gold.upos == "PART" or gold.deprel == "discourse")
    if not gold_is_discourse:
        return 0, 0
    # Error: LatinCy uses ADV or advmod where gold has PART/discourse
    pred_is_adv = (pred.upos == "ADV" or pred.deprel == "advmod")
    return 1, int(pred_is_adv)


def _check_copula_inversion(
    gold: GoldToken,
    pred: PredToken,
    gold_tokens: List[GoldToken],
) -> Tuple[int, int]:
    """Return (is_gold_phenomenon, is_error).

    Copula inversion: gold assigns COP to a form of *sum* (with an
    oblique head), but LatinCy makes it ROOT.
    """
    is_sum = gold.form.lower() in SUM_FORMS
    if not is_sum:
        return 0, 0
    gold_is_cop = (gold.deprel == "cop")
    if not gold_is_cop:
        return 0, 0
    # Error: LatinCy makes *sum* the root instead
    pred_is_root = (pred.deprel in ("root", "ROOT"))
    return 1, int(pred_is_root)


def _check_passive_expletive(gold: GoldToken, pred: PredToken) -> Tuple[int, int]:
    """Return (is_pred_phenomenon, is_error).

    Passive expletive: LatinCy outputs expl:pass.
    UDante does not use this label, so every instance is an error.
    """
    pred_is_expl_pass = pred.deprel in ("expl:pass",)
    if not pred_is_expl_pass:
        return 0, 0
    # Gold won't have expl:pass, so it's always a mismatch
    return 1, 1


def _check_numeral_compound(gold: GoldToken, pred: PredToken) -> Tuple[int, int]:
    """Return (is_gold_phenomenon, is_error).

    Compound numerals: gold uses flat, but LatinCy predicts nummod.
    """
    gold_is_flat = (gold.deprel == "flat")
    if not gold_is_flat:
        return 0, 0
    pred_is_nummod = (pred.deprel == "nummod")
    return 1, int(pred_is_nummod)


def _extract_source_file(sent_id: str) -> str:
    """Extract sub-corpus name from sent_id (e.g. 'Epistole-01-001' -> 'Epistole')."""
    if not sent_id:
        return "unknown"
    # Try common UDante patterns: first segment before hyphen or underscore
    for sep in ("-", "_"):
        if sep in sent_id:
            return sent_id.split(sep)[0]
    return sent_id


def evaluate_sentence(
    gold_sent: GoldSentence,
    pred_tokens: List[PredToken],
) -> SentenceResult:
    """Compare a gold sentence against predicted tokens and return a SentenceResult."""
    source = _extract_source_file(gold_sent.sent_id)
    sr = SentenceResult(sent_id=gold_sent.sent_id, text=gold_sent.text, source_file=source)
    sr.n_gold_tokens = len(gold_sent.tokens)
    sr.n_pred_tokens = len(pred_tokens)
    sr.has_tokenization_mismatch = (sr.n_gold_tokens != sr.n_pred_tokens)

    pairs = align(gold_sent.tokens, pred_tokens)

    for gold, pred in pairs:
        sr.n_tokens += 1
        tr = TokenResult(gold=gold, pred=pred)

        if gold is None or pred is None:
            # Tokenisation mismatch — skip metric counting for this pair
            sr.token_results.append(tr)
            continue

        sr.n_aligned += 1

        # Core metrics
        tr.pos_match = (gold.upos == pred.upos)
        tr.head_match = (gold.head == pred.head)
        g_base = _strip_subtype(gold.deprel)
        p_base = _strip_subtype(pred.deprel)
        tr.deprel_base_match = (g_base == p_base)
        tr.deprel_full_match = (gold.deprel == pred.deprel)
        tr.las_match = (tr.head_match and tr.deprel_base_match)
        tr.las_full_match = (tr.head_match and tr.deprel_full_match)

        # XPOS
        tr.xpos_match = (gold.xpos == pred.xpos)

        # UFeats exact match
        gold_feats = gold.feats or {}
        pred_feats = pred.feats or {}
        tr.ufeats_match = (gold_feats == pred_feats)

        # Lemma (case-insensitive)
        tr.lemma_match = (_normalize_lemma(gold.lemma) == _normalize_lemma(pred.lemma))

        if tr.pos_match:
            sr.n_pos_correct += 1
        if tr.head_match:
            sr.n_head_correct += 1
        if tr.las_match:
            sr.n_las_correct += 1
        if tr.las_full_match:
            sr.n_las_full_correct += 1
        if tr.xpos_match:
            sr.n_xpos_correct += 1
        if tr.ufeats_match:
            sr.n_ufeats_correct += 1
        if tr.lemma_match:
            sr.n_lemma_correct += 1

        # Targeted phenomena
        disc_gold, disc_err = _check_discourse_particle(gold, pred)
        sr.n_discourse_gold += disc_gold
        sr.n_discourse_errors += disc_err
        tr.is_discourse_particle_error = bool(disc_err)

        cop_gold, cop_err = _check_copula_inversion(gold, pred, gold_sent.tokens)
        sr.n_copula_gold += cop_gold
        sr.n_copula_errors += cop_err
        tr.is_copula_inversion_error = bool(cop_err)

        expl_pred, expl_err = _check_passive_expletive(gold, pred)
        sr.n_passive_expletive_pred += expl_pred
        sr.n_passive_expletive_errors += expl_err
        tr.is_passive_expletive_error = bool(expl_err)

        num_gold, num_err = _check_numeral_compound(gold, pred)
        sr.n_numeral_gold += num_gold
        sr.n_numeral_errors += num_err
        tr.is_numeral_compound_error = bool(num_err)

        sr.token_results.append(tr)

    return sr


def evaluate_corpus(
    gold_sentences: List[GoldSentence],
    pred_sentences: List[List[PredToken]],
) -> EvalResult:
    """Evaluate all sentences and return a global EvalResult."""
    result = EvalResult()

    for gold_sent, pred_tokens in zip(gold_sentences, pred_sentences):
        sr = evaluate_sentence(gold_sent, pred_tokens)
        result.sentences.append(sr)

        result.total_tokens += sr.n_tokens
        result.total_gold_tokens += sr.n_gold_tokens
        result.total_pred_tokens += sr.n_pred_tokens
        result.total_aligned += sr.n_aligned
        result.total_pos_correct += sr.n_pos_correct
        result.total_head_correct += sr.n_head_correct
        result.total_las_correct += sr.n_las_correct
        result.total_las_full_correct += sr.n_las_full_correct
        result.total_xpos_correct += sr.n_xpos_correct
        result.total_ufeats_correct += sr.n_ufeats_correct
        result.total_lemma_correct += sr.n_lemma_correct
        result.total_excluded += (sr.n_tokens - sr.n_aligned)

        if sr.has_tokenization_mismatch:
            result.sents_with_tok_mismatch += 1

        result.discourse_gold += sr.n_discourse_gold
        result.discourse_errors += sr.n_discourse_errors
        result.copula_gold += sr.n_copula_gold
        result.copula_errors += sr.n_copula_errors
        result.passive_expletive_pred += sr.n_passive_expletive_pred
        result.passive_expletive_errors += sr.n_passive_expletive_errors
        result.numeral_gold += sr.n_numeral_gold
        result.numeral_errors += sr.n_numeral_errors

        # Build a map from gold token idx -> gold token for head UPOS lookup
        gold_by_idx = {t.idx: t for t in gold_sent.tokens}
        pred_by_idx = {t.idx: t for t in pred_tokens}

        # Accumulate per-token confusion data
        for tr in sr.token_results:
            if tr.gold is None or tr.pred is None:
                continue
            g_dep = tr.gold.deprel
            p_dep = tr.pred.deprel
            g_base = _strip_subtype(g_dep)
            p_base = _strip_subtype(p_dep)

            # Full deprel confusion
            result.deprel_total[g_dep] += 1
            if g_dep != p_dep:
                result.deprel_confusion[g_dep][p_dep] += 1

            # Base deprel confusion
            result.deprel_total_base[g_base] += 1
            if g_base != p_base:
                result.deprel_confusion_base[g_base][p_base] += 1

            # UPOS confusion
            result.upos_total[tr.gold.upos] += 1
            if tr.gold.upos != tr.pred.upos:
                result.upos_confusion[tr.gold.upos][tr.pred.upos] += 1

            # Head attachment error matrix
            if tr.head_match is False:
                gold_head_upos = "ROOT"
                pred_head_upos = "ROOT"
                if tr.gold.head > 0 and tr.gold.head in gold_by_idx:
                    gold_head_upos = gold_by_idx[tr.gold.head].upos
                if tr.pred.head > 0 and tr.pred.head in pred_by_idx:
                    pred_head_upos = pred_by_idx[tr.pred.head].upos
                result.head_confusion[(pred_head_upos, gold_head_upos)] += 1

            # UFeats per-feature F1
            gold_feats = tr.gold.feats or {}
            pred_feats = tr.pred.feats or {}
            for k in UFEATS_KEYS:
                g_val = gold_feats.get(k)
                p_val = pred_feats.get(k)
                if g_val and p_val and g_val == p_val:
                    result.ufeats_per_feature[k]["tp"] += 1
                elif p_val and (not g_val or g_val != p_val):
                    result.ufeats_per_feature[k]["fp"] += 1
                elif g_val and (not p_val or g_val != p_val):
                    # Only count FN if not already counted as FP
                    if not (p_val and g_val != p_val):
                        result.ufeats_per_feature[k]["fn"] += 1

        # Per-subcorpus accumulation
        sc = sr.source_file
        if sc not in result.per_subcorpus:
            result.per_subcorpus[sc] = {
                "sentences": 0, "aligned": 0, "gold_tokens": 0,
                "pos_correct": 0, "head_correct": 0,
                "las_correct": 0, "las_full_correct": 0,
                "xpos_correct": 0, "ufeats_correct": 0, "lemma_correct": 0,
            }
        d = result.per_subcorpus[sc]
        d["sentences"] += 1
        d["aligned"] += sr.n_aligned
        d["gold_tokens"] += sr.n_gold_tokens
        d["pos_correct"] += sr.n_pos_correct
        d["head_correct"] += sr.n_head_correct
        d["las_correct"] += sr.n_las_correct
        d["las_full_correct"] += sr.n_las_full_correct
        d["xpos_correct"] += sr.n_xpos_correct
        d["ufeats_correct"] += sr.n_ufeats_correct
        d["lemma_correct"] += sr.n_lemma_correct

    return result

