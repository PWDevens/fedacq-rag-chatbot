"""
Public re-export of load_query_engine for the indexing package boundary.

Callers within the indexing package should import from here rather than
reaching into rag.retrieval directly, keeping the indexing/retrieval
sub-package boundary explicit.
"""

from rag.retrieval.query_engine import load_query_engine


def get_query_engine():
    """
    Return a loaded LlamaIndex retriever backed by the ChromaDB vector store.

    Delegates to load_query_engine() with default parameters. Provided as a
    stable public entry point for the indexing package so that callers are
    insulated from the retrieval module's internal API.

    Returns:
        llama_index retriever: Configured with similarity_top_k=5.
    """
    return load_query_engine()
