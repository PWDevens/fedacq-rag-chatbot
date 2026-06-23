"""
GraphRAG via LightRAG (lazy-imported so the other modes don't require it).

Reuses the local Phi-4 (generation) and BGE (embeddings) models already loaded
by the app. At runtime this is *retrieval-only*: we ask LightRAG for the
graph-assembled context (only_need_context=True) and let api.py do the Phi-4
streaming generation — so graph mode shares the same serving + citation path as
naive/hybrid. The knowledge graph itself is built offline by
scripts/build_graph.py.

Note: LightRAG is async; the synchronous Flask path drives it through a private
event loop. Requests are served one at a time in this demo, so a single loop is
sufficient. Query-time keyword extraction makes one local Phi-4 call, which adds
latency on CPU — acceptable for a portfolio demo of the technique.
"""

import asyncio
import logging
import os

from rag.config import RagConfig

logger = logging.getLogger(__name__)


def local_llm_func(llm):
    """Async adapter over the local Phi-4 model (used at query time)."""

    async def _local(prompt, system_prompt=None, history_messages=None, **kwargs):
        text = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
        return llm.complete(text).text

    return _local


def build_llm_func(llm):
    """
    LLM function for offline graph construction.

    Uses an OpenAI-compatible API model when GRAPH_BUILD_LLM is set (better
    entity/relation extraction, needs OPENAI_API_KEY); otherwise the local
    Phi-4 model. Extraction is an offline step, so a stronger build model does
    not affect CPU runtime latency.
    """
    model = RagConfig.GRAPH_BUILD_LLM
    if not model:
        return local_llm_func(llm)

    from lightrag.llm.openai import openai_complete_if_cache

    async def _api(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            **kwargs,
        )

    logger.info("[graph] using API build model: %s", model)
    return _api


def make_lightrag(embed_model, working_dir, llm_func):
    """Construct a LightRAG instance wired to the given LLM + BGE embeddings."""
    import numpy as np
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    dim = len(embed_model.get_text_embedding("dimension probe"))

    async def _embed_func(texts):
        return np.array(
            [embed_model.get_text_embedding(t) for t in texts], dtype=np.float32
        )

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=dim, max_token_size=512, func=_embed_func
        ),
    )


async def init_storages(rag):
    """Initialize LightRAG storage backends + pipeline status (required in 1.x)."""
    from lightrag.kg.shared_storage import initialize_pipeline_status

    await rag.initialize_storages()
    await initialize_pipeline_status()


class _LoopRunner:
    """Owns a private event loop so the sync Flask path can drive async LightRAG."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()

    def run(self, coro):
        return self._loop.run_until_complete(coro)


class LightRagEngine:
    """Runtime engine: retrieval-only over the prebuilt LightRAG graph."""

    def __init__(self):
        from rag.llm.models import init_models

        llm, embed_model = init_models()
        working_dir = RagConfig.LIGHTRAG_WORKING_DIR
        if not os.path.isdir(working_dir) or not os.listdir(working_dir):
            raise RuntimeError(
                f"LightRAG graph not found at '{working_dir}'. Build it first: "
                "python -m scripts.build_graph"
            )
        self._runner = _LoopRunner()
        self._rag = make_lightrag(embed_model, working_dir, local_llm_func(llm))
        self._runner.run(init_storages(self._rag))
        logger.info("[graph] LightRAG engine ready (working_dir=%s)", working_dir)

    def retrieve_context(self, question):
        from lightrag import QueryParam

        result = self._runner.run(
            self._rag.aquery(
                question,
                param=QueryParam(mode="hybrid", only_need_context=True),
            )
        )
        context_str = (
            result
            if isinstance(result, str) and result.strip()
            else "No relevant context retrieved."
        )
        # LightRAG context is graph-assembled (entities/relations/source chunks)
        # and is not mapped back to discrete FAR sections, so citations are
        # omitted for graph mode (documented known limitation).
        return context_str, []
