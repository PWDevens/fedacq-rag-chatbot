"""
Offline GraphRAG builder — constructs the LightRAG knowledge graph.

Pulls a SUBSET of chunks from the already-committed Chroma index (no repo
cloning needed) and inserts them into LightRAG, which runs LLM-based
entity/relation extraction to build the graph. Defaults to a small FAR subset
so the build completes in minutes on CPU and the artifacts are reproducible.
Raise GRAPH_BUILD_MAX_DOCS for fuller coverage, and set GRAPH_BUILD_LLM (with
OPENAI_API_KEY) to use a stronger extraction model.

Usage:
    python -m scripts.build_graph
    GRAPH_BUILD_MAX_DOCS=80 python -m scripts.build_graph
"""

import asyncio
import os

import chromadb

from rag.config import RagConfig, runtime_chroma_path
from rag.retrieval.graph_lightrag import build_llm_func, make_lightrag, init_storages


def _load_subset(max_docs, regulation="FAR"):
    """Return up to max_docs chunk texts from the committed Chroma index."""
    path = runtime_chroma_path()
    client = chromadb.PersistentClient(path=str(path))
    col = client.get_collection("far_dfars_chroma")
    data = col.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    out = []
    for doc, meta in zip(docs, metas):
        if regulation and (meta or {}).get("regulation") != regulation:
            continue
        if doc:
            out.append(doc)
        if len(out) >= max_docs:
            break
    return out


async def _build():
    from rag.llm.models import init_models

    llm, embed_model = init_models()
    working_dir = RagConfig.LIGHTRAG_WORKING_DIR
    os.makedirs(working_dir, exist_ok=True)

    rag = make_lightrag(embed_model, working_dir, build_llm_func(llm))
    await init_storages(rag)

    max_docs = int(os.environ.get("GRAPH_BUILD_MAX_DOCS", "40"))
    chunks = _load_subset(max_docs)
    print(
        f"[build_graph] inserting {len(chunks)} chunks into LightRAG "
        f"(working_dir={working_dir}) ... this runs LLM extraction and may take "
        f"several minutes on CPU."
    )
    await rag.ainsert(chunks)
    print("[build_graph] done. Graph artifacts written to", working_dir)


def main():
    asyncio.run(_build())


if __name__ == "__main__":
    main()
