# LatinCy Post-processor

Rule-based post-processor that aligns LatinCy CoNLL-U output with
UDante annotation conventions.

## Install

    pip install -e .[dev]

## Run

    latincy-postprocess --input latincy.conllu --output fixed.conllu --changelog changes.tsv

## Test

    pytest