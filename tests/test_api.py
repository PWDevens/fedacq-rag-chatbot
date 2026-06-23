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


def test_chat_stream_cache_hit_replays_without_model(client, monkeypatch):
    """A cache hit streams the stored answer + citations without loading the LLM."""
    from rag import cache

    monkeypatch.setattr(
        cache,
        "get",
        lambda q: {
            "answer": "Certified cost or pricing data is required under FAR 15.403.",
            "citations": [{"index": 1, "regulation": "FAR", "section": "15.403"}],
        },
    )
    r = client.post("/chat_stream", json={"question": "cost or pricing data?"})
    body = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Certified cost or pricing data is required under FAR 15.403." in body
    assert '"citations"' in body
    assert "15.403" in body


def test_chat_stream_engine_init_error(client, monkeypatch):
    """A cache miss + engine init failure returns the user-safe error, not a stack trace."""
    from rag import cache
    import app.api as apimod

    monkeypatch.setattr(cache, "get", lambda q: None)

    def boom():
        raise RuntimeError("engine unavailable")

    monkeypatch.setattr(apimod, "get_engine", boom)
    r = client.post("/chat_stream", json={"question": "anything"})
    body = r.get_data(as_text=True)
    assert "Could not initialize the retrieval engine" in body
