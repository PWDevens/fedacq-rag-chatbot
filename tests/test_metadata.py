from rag.retrieval.metadata import normalize_metadata

def test_normalize_metadata():
    md = {"a": None, "b": ["x"], "c": 123}
    safe = normalize_metadata(md)

    # None values are omitted from metadata dict (not stored as string "None")
    # so that Chroma where-filters for missing keys work correctly.
    assert "a" not in safe

    # Lists and dicts are JSON-serialized to strings
    assert safe["b"] == '["x"]'

    # Other values are converted via str()
    assert safe["c"] == "123"
