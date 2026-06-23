"""
RAG engine factory — selects the retrieval strategy from RagConfig.RAG_MODE.

Every engine implements the same interface:

    retrieve_context(question) -> (context_str: str, citations: list[dict])

so the serving layer (api.py) keeps a single generation + citation path for all
modes. Only retrieval differs:

    naive  : dense vector search (the original behavior)
    hybrid : dense + BM25 fused with Reciprocal Rank Fusion (pure CPU)
    graph  : GraphRAG via LightRAG (offline-built knowledge graph)

The cross-cutting reranker (RERANK, default ON) is applied to naive and hybrid.
"""

import logging
import re
import threading
from dataclasses import dataclass, field

from rag.config import RagConfig, runtime_chroma_path

logger = logging.getLogger(__name__)

_COLLECTION = "far_dfars_chroma"


@dataclass
class RetrievedChunk:
    """Uniform retrieval result so reranking/citations work across modes."""
    text: str
    metadata: dict = field(default_factory=dict)


def _to_chunks(nodes):
    """Convert LlamaIndex NodeWithScore objects to RetrievedChunk."""
    return [RetrievedChunk(text=n.text or "", metadata=n.metadata or {}) for n in nodes]


def nodes_to_citations(chunks):
    """Build the citation dicts the frontend expects from a chunk list."""
    cites = []
    for i, c in enumerate(chunks, start=1):
        m = c.metadata or {}
        cites.append(
            {
                "index": i,
                "regulation": m.get("regulation"),
                "part": m.get("part"),
                "section": m.get("section"),
                "source_path": m.get("source_path"),
            }
        )
    return cites


def _context_and_citations(chunks):
    context = (
        "\n\n".join(c.text for c in chunks)
        if chunks
        else "No relevant context retrieved."
    )
    return context, nodes_to_citations(chunks)


def _maybe_rerank(question, chunks):
    """Apply the cross-encoder when enabled, else truncate to the final count."""
    if not chunks:
        return chunks
    if RagConfig.RERANK:
        from rag.retrieval.reranker import rerank
        return rerank(question, chunks, RagConfig.RERANK_TOP_N)
    return chunks[: RagConfig.RERANK_TOP_N]


class NaiveEngine:
    """Dense vector retrieval (+ optional rerank). Original behavior."""

    def __init__(self):
        from rag.retrieval.query_engine import load_query_engine
        self._retriever = load_query_engine(
            chroma_path=runtime_chroma_path(), top_k=RagConfig.RETRIEVAL_TOP_K
        )

    def retrieve_context(self, question):
        chunks = _to_chunks(self._retriever.retrieve(question))
        chunks = _maybe_rerank(question, chunks)
        return _context_and_citations(chunks)


class HybridEngine:
    """Dense + BM25 fused with Reciprocal Rank Fusion (+ optional rerank)."""

    def __init__(self):
        from rag.retrieval.query_engine import load_query_engine
        path = runtime_chroma_path()
        self._vector = load_query_engine(
            chroma_path=path, top_k=RagConfig.RETRIEVAL_TOP_K
        )
        self._build_bm25(path)

    def _build_bm25(self, path):
        import chromadb
        from rank_bm25 import BM25Okapi

        client = chromadb.PersistentClient(path=str(path))
        col = client.get_collection(_COLLECTION)
        data = col.get(include=["documents", "metadatas"])
        self._ids = data.get("ids") or []
        self._docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        self._by_id = {
            _id: (doc, meta or {})
            for _id, doc, meta in zip(self._ids, self._docs, metas)
        }
        self._bm25 = BM25Okapi([self._tokenize(d) for d in self._docs])
        logger.info("[hybrid] BM25 index built over %d chunks", len(self._docs))

    @staticmethod
    def _tokenize(text):
        return [t for t in re.split(r"\W+", (text or "").lower()) if t]

    @staticmethod
    def _rrf(rank_lists, k0=60):
        """Reciprocal Rank Fusion over several ranked id lists."""
        scores = {}
        for lst in rank_lists:
            for rank, _id in enumerate(lst):
                scores[_id] = scores.get(_id, 0.0) + 1.0 / (k0 + rank)
        return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def retrieve_context(self, question):
        top_k = RagConfig.RETRIEVAL_TOP_K

        # Dense arm — also keep node text/metadata for fused results.
        vec_nodes = self._vector.retrieve(question)
        dense_map = {
            n.node.node_id: RetrievedChunk(n.node.text or "", n.node.metadata or {})
            for n in vec_nodes
        }
        dense_ids = list(dense_map.keys())

        # Sparse arm — BM25 over the same corpus.
        bm25_scores = self._bm25.get_scores(self._tokenize(question))
        bm25_order = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_k]
        bm25_ids = [self._ids[i] for i in bm25_order]

        fused_ids = self._rrf([dense_ids, bm25_ids])[:top_k]

        chunks = []
        for _id in fused_ids:
            if _id in dense_map:
                chunks.append(dense_map[_id])
            elif _id in self._by_id:
                doc, meta = self._by_id[_id]
                chunks.append(RetrievedChunk(doc or "", meta))

        chunks = _maybe_rerank(question, chunks)
        return _context_and_citations(chunks)


class GraphEngine:
    """GraphRAG via LightRAG. Retrieval-only; generation stays in api.py."""

    def __init__(self):
        from rag.retrieval.graph_lightrag import LightRagEngine
        self._impl = LightRagEngine()

    def retrieve_context(self, question):
        return self._impl.retrieve_context(question)


_engine = None
_engine_lock = threading.Lock()


def _build_engine(mode):
    mode = (mode or "naive").strip().lower()
    logger.info("[factory] building RAG engine: mode=%s rerank=%s", mode, RagConfig.RERANK)
    if mode == "naive":
        return NaiveEngine()
    if mode == "hybrid":
        return HybridEngine()
    if mode == "graph":
        return GraphEngine()
    raise ValueError(f"Unknown RAG_MODE '{mode}'. Use one of: naive, hybrid, graph.")


def get_engine(mode=None):
    """
    Return the shared RAG engine for the active mode, building it on first call.
    Thread-safe singleton (mirrors the previous get_query_engine pattern).
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = _build_engine(mode or RagConfig.RAG_MODE)
    return _engine
