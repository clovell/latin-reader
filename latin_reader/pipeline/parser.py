"""LatinCy wrapper — single nlp instance loaded at startup."""
import logging
import spacy
from spacy.language import Language
from flask import Flask

logger = logging.getLogger(__name__)


# Register a passthrough stub for uv_normalizer (requires latincy-preprocess
# which needs specific Python versions; this stub lets us run without it).
@Language.factory(
    "uv_normalizer",
    default_config={"method": "rules", "overwrite": False},
)
def create_uv_normalizer(nlp, name, method, overwrite):
    return lambda doc: doc


def init_parser(app: Flask) -> None:
    """Load the LatinCy model once and stash on app.extensions.

    Tries la_core_web_trf first, falls back to la_core_web_lg.
    """
    model_name = app.config.get("LATINCY_MODEL", "la_core_web_trf")
    fallback = "la_core_web_lg"

    for name in (model_name, fallback):
        try:
            logger.info(f"Loading LatinCy model ({name})...")
            print(f"Loading LatinCy model ({name})...")
            nlp = spacy.load(name, exclude=["lookup_lemmatizer"])
            app.extensions["latin_reader_nlp"] = nlp
            logger.info("Model loaded successfully!")
            print("Model loaded successfully!")
            return
        except OSError:
            logger.warning(f"Model {name} not found, trying fallback...")
            print(f"Model {name} not found, trying fallback...")
            continue

    raise RuntimeError(
        f"Could not load LatinCy model. Tried: {model_name}, {fallback}. "
        f"Run 'bash bootstrap_model.sh' to install the model."
    )


def parse(text: str) -> "spacy.tokens.Doc":
    """Parse Latin text using the loaded nlp model.

    Must be called within a Flask application context.
    """
    from flask import current_app

    nlp = current_app.extensions["latin_reader_nlp"]
    return nlp(text)
