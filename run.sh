#!/usr/bin/env bash
# run.sh — Start the Latin Reader Flask app
#
# Uses the local venv at ~/.venvs/latin-reader/
# Run bootstrap_model.sh first if you haven't set up the LatinCy model yet.

set -euo pipefail

VENV_DIR="$HOME/.venvs/latin-reader"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Venv not found at $VENV_DIR"
    echo "Create it with: python3 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install -e .[dev]"
    exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$SCRIPT_DIR"
export FLASK_APP=latin_reader.app:create_app
export FLASK_DEBUG=1
flask run --host=0.0.0.0 --port=5001
