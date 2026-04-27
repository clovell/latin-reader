"""Flask configuration."""
import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key-change-in-production")
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max request size
    LATINCY_MODEL = os.environ.get("LATINCY_MODEL", "la_core_web_trf")
