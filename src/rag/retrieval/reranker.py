"""
Cross-encoder reranker for the FAR/DFARS RAG pipeline.

Reorders retrieved chunks by query relevance using a small CPU cross-encoder.
Reuses the already-installed `sentence-transformers` package (no new heavy
dependency). The model is tiny (~80 MB) and loaded once as a process singleton.

Applied as a cross-cutting step after retrieval in any mode when RERANK=true,
which improves both the context injected into the LLM and the citations shown
to the user (citations are derived from the post-rerank chunk set).
"""

import logging
import threading

from rag.config import RagConfig

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def _get_model():
    """Load and cache the cross-encoder once (thread-safe)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import CrossEncoder
                logger.info("[reranker] loading cross-encoder: %s", RagConfig.RERANK_MODEL)
                _model = CrossEncoder(RagConfig.RERANK_MODEL)
    return _model


def rerank(query, chunks, top_n):
    """
    Reorder `chunks` by cross-encoder relevance to `query`, keep the top_n.

    Args:
        query (str): The user question.
        chunks (list): Objects exposing a `.text` attribute.
        top_n (int): Number of chunks to keep.

    Returns:
        list: The top_n most relevant chunks, highest first.
    """
    if not chunks:
        return chunks
    model = _get_model()
    scores = model.predict([(query, c.text) for c in chunks])
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in order[:top_n]]
