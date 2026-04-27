"""Loader for the Perseus treebank sentences."""
from __future__ import annotations
import json
import os
import random
from typing import Optional

_CACHE = None


def _load() -> dict:
    """Load perseus_sentences.json from static data directory."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "static", "data", "perseus_sentences.json",
    )
    with open(data_path, encoding="utf-8") as f:
        _CACHE = json.load(f)
    return _CACHE


def get_authors() -> list[str]:
    """Return sorted list of available authors."""
    data = _load()
    return sorted(data.keys())


def get_random_sentence(author: str | None = None) -> Optional[dict]:
    """Return a random sentence dict, optionally filtered by author.

    Returns: {text: str, tokens: list, author: str} or None
    """
    data = _load()

    if author and author in data:
        sentences = data[author]
    elif author:
        return None
    else:
        # Pick a random author first, then a random sentence
        if not data:
            return None
        author = random.choice(list(data.keys()))
        sentences = data[author]

    if not sentences:
        return None

    sentence = random.choice(sentences)
    return {
        "text": sentence["text"],
        "tokens": sentence["tokens"],
        "author": author,
    }
