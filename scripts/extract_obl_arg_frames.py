#!/usr/bin/env python3
"""
Extract obl:arg frame data from a UDante (or any UD) CoNLL-U file.

For every gold token with deprel == "obl:arg", record the lexical frame:
  - head verb lemma, UPOS, Voice
  - token form, lemma, UPOS, Case
  - preposition lemma (from child with deprel=case), if any
  - a short context snippet for manual inspection

Two output files are produced:
  1. A detailed per-occurrence CSV (one row per obl:arg token).
  2. An aggregated frame-frequency CSV, grouped by
     (head_lemma, preposition_lemma, token_case), ranked by count.

Usage:
    python extract_obl_arg_frames.py \\
        --input path/to/udante-dev.conllu \\
        --out-detail obl_arg_detail.csv \\
        --out-frames obl_arg_frames.csv \\
        [--deprel obl:arg]         # override target deprel if desired
        [--context-tokens 6]       # +/- N tokens around the target
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Minimal CoNLL-U reader (self-contained so this script has no dependencies)
# ---------------------------------------------------------------------------

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


@dataclass
class Sentence:
    tokens: List[Token] = field(default_factory=list)
    sent_id: Optional[str] = None
    text: Optional[str] = None

    def by_id(self, tid: int) -> Optional[Token]:
        if 1 <= tid <= len(self.tokens):
            t = self.tokens[tid - 1]
            if t.id == tid:
                return t
        # fall back to linear search if ids are non-contiguous
        for t in self.tokens:
            if t.id == tid:
                return t
        return None

    def children_of(self, tid: int) -> List[Token]:
        return [t for t in self.tokens if t.head == tid]


def _parse_feats(s: str) -> Dict[str, str]:
    if s == "_" or not s:
        return {}
    out = {}
    for pair in s.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
    return out


def read_conllu(path: str) -> Iterator[Sentence]:
    sent = Sentence()
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                if sent.tokens or sent.sent_id or sent.text:
                    yield sent
                    sent = Sentence()
                continue
            if line.startswith("#"):
                if line.startswith("# sent_id ="):
                    sent.sent_id = line.split("=", 1)[1].strip()
                elif line.startswith("# text ="):
                    sent.text = line.split("=", 1)[1].strip()
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                continue
            tid = fields[0]
            if "-" in tid or "." in tid:   # skip MWT and empty nodes
                continue
            sent.tokens.append(Token(
                id=int(tid),
                form=fields[1],
                lemma=fields[2],
                upos=fields[3],
                xpos=fields[4],
                feats=_parse_feats(fields[5]),
                head=int(fields[6]) if fields[6] != "_" else 0,
                deprel=fields[7],
            ))
    if sent.tokens:
        yield sent


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    sent_id: str
    token_id: int
    token_form: str
    token_lemma: str
    token_upos: str
    token_case: str
    token_number: str
    token_gender: str
    preposition_lemma: str         # "" if none
    preposition_form: str
    head_id: int
    head_form: str
    head_lemma: str
    head_upos: str
    head_voice: str
    head_verbform: str
    head_mood: str
    context: str                   # +/- N tokens around the target

    def as_row(self) -> List[str]:
        return [
            self.sent_id, str(self.token_id), self.token_form, self.token_lemma,
            self.token_upos, self.token_case, self.token_number, self.token_gender,
            self.preposition_lemma, self.preposition_form,
            str(self.head_id), self.head_form, self.head_lemma, self.head_upos,
            self.head_voice, self.head_verbform, self.head_mood,
            self.context,
        ]


DETAIL_HEADER = [
    "sent_id", "token_id", "token_form", "token_lemma",
    "token_upos", "token_case", "token_number", "token_gender",
    "preposition_lemma", "preposition_form",
    "head_id", "head_form", "head_lemma", "head_upos",
    "head_voice", "head_verbform", "head_mood",
    "context",
]


def _find_case_child(sent: Sentence, tok: Token) -> Optional[Token]:
    for c in sent.children_of(tok.id):
        if c.deprel == "case" and c.upos == "ADP":
            return c
    # Fallback: any ADP child, even if labeled differently
    for c in sent.children_of(tok.id):
        if c.upos == "ADP":
            return c
    return None


def _context_snippet(sent: Sentence, token: Token, radius: int) -> str:
    lo = max(1, token.id - radius)
    hi = min(len(sent.tokens), token.id + radius)
    pieces = []
    for t in sent.tokens:
        if lo <= t.id <= hi:
            mark = "**" if t.id == token.id else ""
            pieces.append(f"{mark}{t.form}{mark}")
    return " ".join(pieces)


def extract_frames(sent: Sentence, target_deprel: str,
                   context_radius: int) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    for tok in sent.tokens:
        if tok.deprel != target_deprel:
            continue
        head = sent.by_id(tok.head) if tok.head else None
        prep = _find_case_child(sent, tok)
        records.append(FrameRecord(
            sent_id=sent.sent_id or "",
            token_id=tok.id,
            token_form=tok.form,
            token_lemma=tok.lemma,
            token_upos=tok.upos,
            token_case=tok.feats.get("Case", ""),
            token_number=tok.feats.get("Number", ""),
            token_gender=tok.feats.get("Gender", ""),
            preposition_lemma=prep.lemma if prep else "",
            preposition_form=prep.form if prep else "",
            head_id=head.id if head else 0,
            head_form=head.form if head else "",
            head_lemma=head.lemma if head else "",
            head_upos=head.upos if head else "",
            head_voice=head.feats.get("Voice", "") if head else "",
            head_verbform=head.feats.get("VerbForm", "") if head else "",
            head_mood=head.feats.get("Mood", "") if head else "",
            context=_context_snippet(sent, tok, context_radius),
        ))
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

FrameKey = Tuple[str, str, str, str]   # (head_lemma, head_upos, prep_lemma, token_case)

FRAMES_HEADER = [
    "rank", "count", "cumulative_pct",
    "head_lemma", "head_upos",
    "preposition_lemma", "token_case",
    "example_context",
    "n_passive_or_deponent_heads",
    "example_head_forms",
    "example_token_forms",
]


def aggregate(records: List[FrameRecord]) -> List[dict]:
    groups: Dict[FrameKey, List[FrameRecord]] = defaultdict(list)
    for r in records:
        key = (r.head_lemma, r.head_upos, r.preposition_lemma, r.token_case)
        groups[key].append(r)

    total = len(records)
    rows = []
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    cumulative = 0
    for rank, (key, recs) in enumerate(ranked, start=1):
        cumulative += len(recs)
        head_lemma, head_upos, prep_lemma, token_case = key
        n_passive = sum(1 for r in recs
                        if r.head_voice == "Pass"
                        or r.head_verbform == "Part")
        head_forms = Counter(r.head_form for r in recs).most_common(3)
        token_forms = Counter(r.token_form for r in recs).most_common(3)
        rows.append({
            "rank": rank,
            "count": len(recs),
            "cumulative_pct": f"{100.0 * cumulative / total:.1f}" if total else "0.0",
            "head_lemma": head_lemma,
            "head_upos": head_upos,
            "preposition_lemma": prep_lemma,
            "token_case": token_case,
            "example_context": recs[0].context,
            "n_passive_or_deponent_heads": n_passive,
            "example_head_forms": "; ".join(f"{f}({n})" for f, n in head_forms),
            "example_token_forms": "; ".join(f"{f}({n})" for f, n in token_forms),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract obl:arg frames from a UD CoNLL-U file.")
    ap.add_argument("--input", required=True, help="Input CoNLL-U file")
    ap.add_argument("--out-detail", required=True,
                    help="Per-occurrence CSV output")
    ap.add_argument("--out-frames", required=True,
                    help="Aggregated frames CSV output")
    ap.add_argument("--deprel", default="obl:arg",
                    help="Target deprel (default: obl:arg). "
                         "Use e.g. 'obl:agent' or 'advcl:pred' to adapt.")
    ap.add_argument("--context-tokens", type=int, default=6,
                    help="Context radius in tokens (default: 6)")
    args = ap.parse_args()

    all_records: List[FrameRecord] = []
    n_sents = 0
    for sent in read_conllu(args.input):
        n_sents += 1
        all_records.extend(
            extract_frames(sent, args.deprel, args.context_tokens))

    # Write detail CSV
    with open(args.out_detail, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(DETAIL_HEADER)
        for r in all_records:
            w.writerow(r.as_row())

    # Write aggregated frames CSV
    agg = aggregate(all_records)
    with open(args.out_frames, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAMES_HEADER)
        w.writeheader()
        for row in agg:
            w.writerow(row)

    # Console summary
    print(f"Sentences processed: {n_sents}")
    print(f"Tokens with deprel '{args.deprel}': {len(all_records)}")
    print(f"Distinct frames (head_lemma × prep × case): {len(agg)}")
    print(f"Detail CSV written to: {args.out_detail}")
    print(f"Aggregated frames CSV written to: {args.out_frames}")

    if agg:
        print("\nTop 15 frames:")
        print(f"{'rank':>4} {'count':>5}  {'cum%':>6}  frame")
        for row in agg[:15]:
            frame = (f"{row['head_lemma']}({row['head_upos']}) "
                     f"+ {row['preposition_lemma'] or '∅'} "
                     f"+ {row['token_case'] or '∅'}")
            print(f"{row['rank']:>4} {row['count']:>5}  "
                  f"{row['cumulative_pct']:>5}%  {frame}")


if __name__ == "__main__":
    main()