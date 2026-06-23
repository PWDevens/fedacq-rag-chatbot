"""
GPU GraphRAG builder — constructs the LightRAG knowledge graph with a
transformers LLM on CUDA (entity/relation extraction is impractical on CPU).

This mirrors scripts/build_graph.py but swaps the extraction LLM for a GPU
transformers model (default Phi-4-mini-instruct, the same weights as the ONNX
runtime model) and the embedder onto CUDA. Pulls a topical subset of chunks
from the committed Chroma index and writes the graph to LIGHTRAG_WORKING_DIR.

Designed to run on a GPU host (e.g. a RunPod pod). See the README "Graph mode"
section for the runbook. Runtime serving still uses Phi-4 ONNX on CPU.

Usage (on a GPU host, repo root):
    pip install -r requirements_graph.txt transformers accelerate
    GRAPH_BUILD_PART=15 GRAPH_BUILD_MAX_DOCS=50 python -m scripts.build_graph_gpu

Env:
    GRAPH_BUILD_PART       restrict to a FAR part, e.g. "15" (default: all FAR)
    GRAPH_BUILD_MAX_DOCS   number of chunks to ingest (default: 50)
    BUILD_LLM              HF model id (default: microsoft/Phi-4-mini-instruct)
    BUILD_MAX_TOKENS       max new tokens per extraction call (default: 1024)
"""

import asyncio
import os
import time

import chromadb

from rag.config import RagConfig
from rag.retrieval.graph_lightrag import make_lightrag, init_storages

PART = os.environ.get("GRAPH_BUILD_PART")
MAX_DOCS = int(os.environ.get("GRAPH_BUILD_MAX_DOCS", "50"))
LLM_ID = os.environ.get("BUILD_LLM", "microsoft/Phi-4-mini-instruct")
GEN_TOKENS = int(os.environ.get("BUILD_MAX_TOKENS", "1024"))


def _build_transformers_llm():
    """Return an async LightRAG llm func backed by a CUDA transformers model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(LLM_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map=device
    )
    model.eval()

    async def _llm(prompt, system_prompt=None, history_messages=None, **kw):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for h in (history_messages or []):
            msgs.append(h)
        msgs.append({"role": "user", "content": prompt})
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=GEN_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id
            )
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return _llm, device


def _load_chunks(n):
    col = chromadb.PersistentClient(path=str(RagConfig.CHROMA_PATH)).get_collection("far_dfars_chroma")
    data = col.get(include=["documents", "metadatas"])
    out = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        m = meta or {}
        if m.get("regulation") != "FAR" or not doc:
            continue
        if PART and str(m.get("part")) != PART:
            continue
        out.append(doc)
        if len(out) >= n:
            break
    return out


async def _build():
    import torch
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = HuggingFaceEmbedding(model_name=RagConfig.EMBED_MODEL_NAME, device=device)
    llm_func, dev = _build_transformers_llm()
    print(f"[gpu-build] device={dev} model={LLM_ID} part={PART} max_docs={MAX_DOCS}", flush=True)

    working_dir = RagConfig.LIGHTRAG_WORKING_DIR
    os.makedirs(working_dir, exist_ok=True)
    rag = make_lightrag(embed, working_dir, llm_func)
    await init_storages(rag)

    chunks = _load_chunks(MAX_DOCS)
    print(f"[gpu-build] inserting {len(chunks)} chunks ...", flush=True)
    t = time.time()
    await rag.ainsert(chunks)
    print(f"[gpu-build] done in {time.time() - t:.0f}s -> {working_dir}", flush=True)


def main():
    asyncio.run(_build())


if __name__ == "__main__":
    main()
