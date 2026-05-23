# rag/indexing/loader.py
from rag.retrieval.query_engine import load_query_engine

def get_query_engine():
    return load_query_engine()
