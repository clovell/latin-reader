"""User-facing routes."""
from flask import Blueprint, render_template

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    """Serve the main reader page."""
    return render_template("index.html")


@bp.route("/about")
def about():
    """Serve the about page."""
    return render_template("about.html")
