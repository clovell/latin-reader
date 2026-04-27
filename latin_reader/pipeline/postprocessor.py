"""Thin wrapper around latincy_postprocessor.

Converts between spaCy Doc ↔ CoNLL-U string ↔ postprocessor Sentence
objects, applies the rule pipeline, and returns harmonized output.
"""
from __future__ import annotations

from latin_reader.latincy_postprocessor.pipeline import run_pipeline, RunReport
from latin_reader.latincy_postprocessor.conllu_io import read_conllu_string, write_conllu_string


def harmonize_conllu(conllu_text: str) -> tuple[str, list, RunReport]:
    """Apply post-processor rules to CoNLL-U text.

    Args:
        conllu_text: Raw CoNLL-U formatted string from the parser.

    Returns:
        Tuple of (harmonized_conllu, change_log, report).
        - harmonized_conllu: The modified CoNLL-U string.
        - change_log: List of Change objects describing each modification.
        - report: RunReport with per-rule statistics.
    """
    sentences = list(read_conllu_string(conllu_text))
    report = run_pipeline(sentences)
    return write_conllu_string(sentences), report.changes, report
