## tests/test_query_engine.py
from rag.retrieval.query_engine import load_query_engine

def test_load_query_engine():
    qe = load_query_engine(chroma_path="./test_chroma")
    assert qe is not None

