# rag/__init__.py

"""
Top-level package for the FAR/DFARS RAG engine.
Exposes high-level indexing and query engine utilities.
"""

from .indexing import build_index, get_query_engine
from .retrieval import load_query_engine
from .llm import init_models

__all__ = [
    "build_index",
    "get_query_engine",
    "load_query_engine",
    "init_models",
]

