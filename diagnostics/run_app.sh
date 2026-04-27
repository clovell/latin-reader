#!/bin/bash
# run_app.sh — Start the LatinCy vs. UDante Evaluation App
#
# Any arguments after -- are forwarded to app.py as command-line options:
#
#   bash run_app.sh                                          # interactive only
#   bash run_app.sh -- --udante-path ./file.conllu          # pre-load a file
#   bash run_app.sh -- --udante-path ./file.conllu \
#                      --model la_core_web_trf \
#                      --max-sents 200                       # full pre-config
#
# Supported app options (all optional):
#   --udante-path PATH    Path to a .conllu file to pre-load
#   --model MODEL         LatinCy model name (default: la_core_web_trf)
#   --max-sents N         Max sentences to evaluate (default: 0 = all)

VENV_DIR="$HOME/.venvs/latin-dependency-parser"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Venv not found at $VENV_DIR"
    echo "Run 'bash setup_env.sh' inside latin-dependency-parser/ first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

PYTHON="$VENV_DIR/bin/python3"
STREAMLIT="$VENV_DIR/bin/streamlit"

echo "Activated venv: $PYTHON"
echo ""
echo "============================================"
echo "  LatinCy vs. UDante Evaluation App"
echo "  http://localhost:8501"
echo "============================================"
echo ""

cd "$SCRIPT_DIR"

# "$@" forwards any args you passed (e.g. -- --udante-path ./file.conllu)
# Streamlit passes everything after -- to the script as sys.argv.
"$STREAMLIT" run app.py \
    --server.port 8501 \
    --server.headless false \
    --theme.base dark \
    --theme.primaryColor "#388bfd" \
    --theme.backgroundColor "#0d1117" \
    --theme.secondaryBackgroundColor "#161b22" \
    --theme.textColor "#e6edf3" \
    "$@"
