"""Minimal CoNLL-U reader and writer. Skips multi-word and empty-node rows."""
from __future__ import annotations
import io
from typing import Iterator, List
from .sentence import Sentence, Token


def _parse_feats(s: str) -> dict:
    if s == "_" or not s:
        return {}
    out = {}
    for pair in s.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k] = v
    return out


def _serialize_feats(feats: dict) -> str:
    if not feats:
        return "_"
    return "|".join(f"{k}={feats[k]}" for k in sorted(feats))


def read_conllu(path: str) -> Iterator[Sentence]:
    sent = Sentence()
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                if sent.tokens or sent.comments:
                    yield sent
                    sent = Sentence()
                continue
            if line.startswith("#"):
                sent.comments.append(line)
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                continue
            tid = fields[0]
            # skip multi-word (e.g. "1-2") and empty-node (e.g. "1.1") rows
            if "-" in tid or "." in tid:
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
                deps=fields[8],
                misc=fields[9],
            ))
    if sent.tokens or sent.comments:
        yield sent


def write_conllu(sentences: List[Sentence], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            for c in s.comments:
                f.write(c + "\n")
            for t in s.tokens:
                f.write("\t".join([
                    str(t.id), t.form, t.lemma, t.upos, t.xpos,
                    _serialize_feats(t.feats),
                    str(t.head), t.deprel, t.deps, t.misc,
                ]) + "\n")
            f.write("\n")


def read_conllu_string(text: str) -> Iterator[Sentence]:
    """Parse CoNLL-U from an in-memory string. Same logic as read_conllu()."""
    sent = Sentence()
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line:
            if sent.tokens or sent.comments:
                yield sent
                sent = Sentence()
            continue
        if line.startswith("#"):
            sent.comments.append(line)
            continue
        fields = line.split("\t")
        if len(fields) != 10:
            continue
        tid = fields[0]
        if "-" in tid or "." in tid:
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
            deps=fields[8],
            misc=fields[9],
        ))
    if sent.tokens or sent.comments:
        yield sent


def write_conllu_string(sentences: List[Sentence]) -> str:
    """Serialize sentences to a CoNLL-U string. Same format as write_conllu()."""
    buf = io.StringIO()
    for s in sentences:
        for c in s.comments:
            buf.write(c + "\n")
        for t in s.tokens:
            buf.write("\t".join([
                str(t.id), t.form, t.lemma, t.upos, t.xpos,
                _serialize_feats(t.feats),
                str(t.head), t.deprel, t.deps, t.misc,
            ]) + "\n")
        buf.write("\n")
    return buf.getvalue()