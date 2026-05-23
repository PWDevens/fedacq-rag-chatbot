# rag/retrieval/__init__.py

"""
Retrieval utilities for the FAR/DFARS RAG system.
Includes:
- DITA parsing
- Metadata normalization
- Query engine loading
"""

from .parser_dita import (
    clone_if_needed,
    parse_ditamap,
    extract_fillins_and_text,
    infer_part_subpart_section,
    build_documents_from_repo,
)

from .metadata import normalize_metadata
from .query_engine import load_query_engine

__all__ = [
    "clone_if_needed",
    "parse_ditamap",
    "extract_fillins_and_text",
    "infer_part_subpart_section",
    "build_documents_from_repo",
    "normalize_metadata",
    "load_query_engine",
]
