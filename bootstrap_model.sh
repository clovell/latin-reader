#!/usr/bin/env bash
# Downloads the LatinCy transformer model. Run once after `pip install -e .`
# Uses --no-deps because latincy-preprocess requires Python 3.10+, but
# we handle the missing uv_normalizer component with a stub factory.
set -euo pipefail
python -m pip install --no-deps https://huggingface.co/latincy/la_core_web_trf/resolve/main/la_core_web_trf-3.9.1-py3-none-any.whl
python -m pip install "spacy-transformers>=1.3.9,<1.4.0"
echo "LatinCy model installed."
