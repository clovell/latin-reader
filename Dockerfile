FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (so logs appear in real-time)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy project metadata first (Docker layer caching)
COPY pyproject.toml README.md ./

# Copy the package
COPY latin_reader ./latin_reader
COPY scripts ./scripts

# Download large treebank data files from GitHub (too big for HF without LFS)
RUN mkdir -p latin_reader/static/data && \
    curl -L -o latin_reader/static/data/perseus_sentences.json \
        "https://raw.githubusercontent.com/clovell/latin-reader/main/latin_reader/static/data/perseus_sentences.json" && \
    curl -L -o latin_reader/static/data/udante.json \
        "https://raw.githubusercontent.com/clovell/latin-reader/main/latin_reader/static/data/udante.json"

# Install the package and its dependencies
RUN pip install --no-cache-dir -e .

# Install spacy-transformers + CPU-only PyTorch (needed for trf model)
RUN pip install --no-cache-dir \
    "spacy-transformers>=1.3.0,<1.4.0" \
    "torch>=2.0.0,<2.6.0" --index-url https://download.pytorch.org/whl/cpu \
    "transformers>=4.30.0,<5.0.0"

# Install HF hub with Xet support (model is stored on HF's Xet backend)
RUN pip install --no-cache-dir "huggingface_hub[hf_xet]"

# Install LatinCy trf model (transformer-based, best accuracy)
# --no-deps: skip latincy-preprocess dependency (we stub uv_normalizer in parser.py)
RUN pip install --no-cache-dir --no-deps \
    "https://huggingface.co/latincy/la_core_web_trf/resolve/main/la_core_web_trf-3.9.1-py3-none-any.whl"

# Install gunicorn for production serving
RUN pip install --no-cache-dir gunicorn

# Set model preference
ENV LATINCY_MODEL=la_core_web_trf

# Default port (overridden by platform)
EXPOSE 7860

# Run with gunicorn
# --timeout 120: transformer parsing can take a moment
# --workers 1:   single worker to fit memory constraints
# --preload:     load model before forking to save memory
CMD ["gunicorn", "latin_reader.app:create_app()", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "--preload"]
