"""Pytest fixtures for Latin Reader tests."""
import pytest
from latin_reader.app import create_app


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app({"TESTING": True})
    yield app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()
