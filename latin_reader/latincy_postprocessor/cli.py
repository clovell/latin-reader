"""Command-line entry point.

Usage:
    python -m latincy_postprocessor.cli \\
        --input latincy_output.conllu \\
        --output postprocessed.conllu \\
        [--changelog changes.tsv]
"""
from __future__ import annotations
import argparse
import csv
from typing import List

from .conllu_io import read_conllu, write_conllu
from .pipeline import run_pipeline
from .sentence import Sentence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="LatinCy CoNLL-U input")
    ap.add_argument("--output", required=True, help="Post-processed CoNLL-U output")
    ap.add_argument("--changelog", default=None,
                    help="Optional TSV recording every change made")
    args = ap.parse_args()

    sentences: List[Sentence] = list(read_conllu(args.input))
    report = run_pipeline(sentences)
    write_conllu(sentences, args.output)

    print(report.summary())

    if args.changelog:
        with open(args.changelog, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["sent_id", "token_id", "form", "field",
                        "old_value", "new_value", "rule"])
            for c in report.changes:
                w.writerow([c.sent_id, c.token_id, c.token_form, c.field,
                            c.old_value, c.new_value, c.rule_name])
        print(f"Changelog written to {args.changelog}")


if __name__ == "__main__":
    main()