# rag/llm/__init__.py

"""
LLM and embedding model initialization for the RAG system.
This package exposes `init_models()` as the public entrypoint.
"""

from .models import init_models

__all__ = ["init_models"]
