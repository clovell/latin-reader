"""Flask application factory."""
from __future__ import annotations
from flask import Flask


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object("latin_reader.config.Config")
    if config_overrides:
        app.config.update(config_overrides)

    from latin_reader.pipeline.parser import init_parser
    init_parser(app)

    from latin_reader.routes import main, api, dev
    app.register_blueprint(main.bp)
    app.register_blueprint(api.bp, url_prefix="/api")
    app.register_blueprint(dev.bp, url_prefix="/dev")

    return app
