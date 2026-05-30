## tests/test_metadata.py

from rag.retrieval.metadata import normalize_metadata

def test_normalize_metadata():
    md = {"a": None, "b": ["x"], "c": 123}
    safe = normalize_metadata(md)
    assert safe["a"] == "None"
    assert isinstance(safe["b"], str)
    assert safe["c"] == "123"

