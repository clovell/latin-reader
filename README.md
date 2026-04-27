---
title: Latin Reader
emoji: 📜
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Latin Reader

A Flask-based web application that helps intermediate Latin students read prose and poetry by automatically parsing input text and displaying visual dependency trees.

## Architecture

```
latin-reader/
├── latin_reader/           # Main Flask application package
│   ├── app.py              # App factory (create_app)
│   ├── config.py           # Configuration
│   ├── routes/             # Flask blueprints
│   │   ├── main.py         # User-facing routes (/)
│   │   ├── api.py          # JSON API endpoints (/api/*)
│   │   └── dev.py          # Developer tools (/dev/*)
│   ├── pipeline/           # NLP processing pipeline
│   │   ├── parser.py       # LatinCy wrapper, single nlp instance
│   │   ├── postprocessor.py  # Post-processor integration (Phase 2)
│   │   ├── chunker.py      # Dependency tree → nested chunks (Phase 3)
│   │   ├── renderer.py     # Chunk tree → SVG sentence map (Phase 4)
│   │   ├── vocab.py        # Vocabulary list extractor (Phase 5)
│   │   └── export.py       # CoNLL-U/PDF export (Phase 6)
│   ├── latincy_postprocessor/  # LatinCy→UDante harmonization rules
│   ├── treebanks/          # Treebank data loaders
│   ├── static/             # CSS, JS, data files
│   └── templates/          # Jinja2 templates
├── scripts/                # Utility scripts
├── tests/                  # Pytest test suite
└── diagnostics/            # Preserved LatinCy-vs-UDante evaluation app
```

## Quick Start

### 1. Create a virtual environment

```bash
python3 -m venv ~/.venvs/latin-reader
source ~/.venvs/latin-reader/bin/activate
```

### 2. Install the package

```bash
pip install -e .[dev]
```

### 3. Download the LatinCy model

```bash
bash bootstrap_model.sh
```

### 4. Run the app

```bash
bash run.sh
```

Open [http://localhost:5001](http://localhost:5001) in your browser.

## Development

### Running Tests

```bash
pytest
```

### Dev Mode

Access developer tools by appending `?dev=1` to any URL, or navigate to `/dev/`.

## Known Limitations

- Chunking is rule-based and continually improving (Phase 3+)
- Word-sense disambiguation is deferred to a future phase
- Context-sensitive vocabulary glosses are planned but not yet implemented
