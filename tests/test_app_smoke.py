"""Smoke tests: app factory boots, basic routes return 200."""
import json


def test_app_factory_boots(app):
    """The app factory should create a Flask app without error."""
    assert app is not None
    assert "latin_reader_nlp" in app.extensions


def test_index_returns_200(client):
    """GET / should return 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Latin" in response.data


def test_analyze_returns_valid_json(client):
    """POST /api/analyze with a simple sentence should return valid JSON with html."""
    response = client.post(
        "/api/analyze",
        data=json.dumps({"text": "Gallia est omnis divisa in partes tres."}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "html" in data
    assert len(data["html"]) > 0
    assert "tokens" in data
    assert len(data["tokens"]) > 0


def test_analyze_empty_text_returns_400(client):
    """POST /api/analyze with empty text should return 400."""
    response = client.post(
        "/api/analyze",
        data=json.dumps({"text": ""}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_export_conllu_returns_text(client):
    """POST /api/export/conllu should return CoNLL-U formatted text."""
    response = client.post(
        "/api/export/conllu",
        data=json.dumps({"text": "Gallia est omnis divisa in partes tres."}),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert response.content_type == "text/plain; charset=utf-8"
    text = response.data.decode("utf-8")
    assert "# sent_id" in text
    assert "# text" in text


def test_dev_route_returns_200(client):
    """GET /dev/ should return 200."""
    response = client.get("/dev/")
    assert response.status_code == 200
