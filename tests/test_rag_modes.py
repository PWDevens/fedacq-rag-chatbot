"""
Unit tests for the multi-mode RAG additions: engine factory dispatch, the
cross-encoder reranker, hybrid Reciprocal Rank Fusion, citation shaping, and the
persistent answer cache. These avoid loading the Phi-4 model or the real index
so they run fast in CI.
"""

import os

import pytest

from rag.config import RagConfig
from rag.retrieval import factory
from rag.retrieval.factory import (
    HybridEngine,
    RetrievedChunk,
    nodes_to_citations,
    _maybe_rerank,
)


def test_factory_unknown_mode_raises():
    with pytest.raises(ValueError):
        factory._build_engine("does-not-exist")


def test_nodes_to_citations_shape():
    chunks = [
        RetrievedChunk("text a", {"regulation": "FAR", "part": "15", "section": "15.403"}),
        RetrievedChunk("text b", {"regulation": "DFARS", "section": "215.403"}),
    ]
    cites = nodes_to_citations(chunks)
    assert [c["index"] for c in cites] == [1, 2]
    assert cites[0]["regulation"] == "FAR"
    assert cites[0]["section"] == "15.403"
    assert cites[1]["regulation"] == "DFARS"
    # Missing keys surface as None rather than raising.
    assert cites[1]["part"] is None


def test_maybe_rerank_truncates_when_disabled(monkeypatch):
    monkeypatch.setattr(RagConfig, "RERANK", False)
    monkeypatch.setattr(RagConfig, "RERANK_TOP_N", 2)
    chunks = [RetrievedChunk(f"c{i}", {}) for i in range(5)]
    out = _maybe_rerank("q", chunks)
    assert [c.text for c in out] == ["c0", "c1"]


def test_rerank_reorders_by_score(monkeypatch):
    from rag.retrieval import reranker

    class FakeModel:
        # Score later chunks higher so the order should reverse.
        def predict(self, pairs):
            return list(range(len(pairs)))

    monkeypatch.setattr(reranker, "_model", FakeModel())
    chunks = [RetrievedChunk(f"c{i}", {}) for i in range(4)]
    out = reranker.rerank("q", chunks, top_n=2)
    assert [c.text for c in out] == ["c3", "c2"]


def test_rrf_fusion_orders_by_combined_rank():
    # id "b" appears high in both lists -> should win.
    dense = ["a", "b", "c"]
    sparse = ["b", "d", "a"]
    fused = HybridEngine._rrf([dense, sparse])
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}


def test_tokenize_lowercases_and_splits():
    assert HybridEngine._tokenize("FAR 15.403, Cost-Data!") == [
        "far", "15", "403", "cost", "data"
    ]


def test_answer_cache_roundtrip(tmp_path, monkeypatch):
    from rag import cache

    monkeypatch.setattr(RagConfig, "ANSWER_CACHE", True)
    monkeypatch.setattr(RagConfig, "ANSWER_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setattr(cache, "_conn", None)

    assert cache.get("hello?") is None
    cache.put("hello?", "world", [{"index": 1, "regulation": "FAR"}])
    hit = cache.get("hello?")
    assert hit["answer"] == "world"
    assert hit["citations"][0]["regulation"] == "FAR"
    # Normalization: case/whitespace-insensitive key.
    assert cache.get("  HELLO?  ")["answer"] == "world"


def test_answer_cache_disabled_is_noop(tmp_path, monkeypatch):
    from rag import cache

    monkeypatch.setattr(RagConfig, "ANSWER_CACHE", False)
    monkeypatch.setattr(RagConfig, "ANSWER_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setattr(cache, "_conn", None)

    cache.put("q", "a", [])
    assert cache.get("q") is None


def test_rerank_empty_list_returns_empty():
    from rag.retrieval import reranker

    # No model load needed for an empty candidate set.
    assert reranker.rerank("q", [], top_n=5) == []


def test_cache_key_separates_by_mode(tmp_path, monkeypatch):
    """Same question under different RAG_MODE must not collide in the cache."""
    from rag import cache

    monkeypatch.setattr(RagConfig, "ANSWER_CACHE", True)
    monkeypatch.setattr(RagConfig, "ANSWER_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setattr(cache, "_conn", None)

    monkeypatch.setattr(RagConfig, "RAG_MODE", "naive")
    cache.put("q", "naive-answer", [])

    monkeypatch.setattr(RagConfig, "RAG_MODE", "hybrid")
    assert cache.get("q") is None  # different mode -> different key -> miss
    cache.put("q", "hybrid-answer", [])

    monkeypatch.setattr(RagConfig, "RAG_MODE", "naive")
    assert cache.get("q")["answer"] == "naive-answer"
