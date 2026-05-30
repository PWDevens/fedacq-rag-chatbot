## tests/test_query_engine_load.py

from rag.retrieval.query_engine import load_query_engine

def test_query_engine_loads():
    qe = load_query_engine()
    assert qe is not None

