# rag/indexing/__init__.py

"""
Indexing utilities for building and loading the Chroma vector index.
"""

from .builder import build_index
from .loader import get_query_engine

__all__ = ["build_index", "get_query_engine"]
