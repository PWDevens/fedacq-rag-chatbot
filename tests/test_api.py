import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app("local")
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_chat_stream_empty_question(client):
    r = client.post("/chat_stream", json={"question": ""})
    assert r.status_code == 200
    assert b"Please enter a question." in r.data


def test_chat_stream_question_too_long(client):
    r = client.post("/chat_stream", json={"question": "x" * 2001})
    assert b"too long" in r.data
