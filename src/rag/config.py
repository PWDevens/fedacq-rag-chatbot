import os
import shutil
from pathlib import Path


def _as_bool(val, default=False):
    """Parse a truthy/falsy env string. None -> default."""
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


class RagConfig:
    CHROMA_PATH = os.environ.get("CHROMA_PATH", "./data/chroma")
    EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    PHI4_MODEL_DIR = os.environ.get(
        "PHI4_MODEL_DIR",
        "./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    )
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

    # --- Retrieval strategy (selectable via .env) ---
    #   naive  : dense vector search only (original behavior)
    #   hybrid : dense + BM25 fused with Reciprocal Rank Fusion (CPU-only)
    #   graph  : GraphRAG via LightRAG (offline-built knowledge graph)
    RAG_MODE = os.environ.get("RAG_MODE", "naive").strip().lower()

    # Cross-cutting reranker (applies to naive + hybrid). Default ON.
    RERANK = _as_bool(os.environ.get("RERANK"), default=True)
    RERANK_MODEL = os.environ.get(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Candidate pool retrieved before reranking, and the final number of chunks
    # fed to the LLM (also the count when reranking is disabled).
    RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "20"))
    RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "5"))

    # Generation latency lever (was hardcoded at 128).
    MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))

    # Persistent exact-match answer cache (repeat questions replay instantly).
    ANSWER_CACHE = _as_bool(os.environ.get("ANSWER_CACHE"), default=True)
    ANSWER_CACHE_PATH = os.environ.get("ANSWER_CACHE_PATH", "./data/cache.db")

    # GraphRAG (LightRAG) settings.
    LIGHTRAG_WORKING_DIR = os.environ.get("LIGHTRAG_WORKING_DIR", "./data/lightrag")
    # Optional model id used ONLY at offline graph-build time for higher-quality
    # entity/relation extraction. Empty -> use the local Phi-4 model.
    GRAPH_BUILD_LLM = os.environ.get("GRAPH_BUILD_LLM", "")


def runtime_chroma_path():
    """
    Return a writable copy of the committed Chroma index.

    ChromaDB checkpoints WAL pages back into chroma.sqlite3 on every query,
    which dirties the LFS-tracked data/chroma index. To keep the committed
    index pristine, the app reads from a gitignored runtime copy instead.

    The copy is created on first use (per machine) and reused afterwards. It is
    refreshed automatically when the committed chroma.sqlite3 is newer than the
    copy (e.g. after a rebuild). Returns the committed path unchanged when
    CHROMA_HOST is set (a remote server owns its own storage) or when the
    committed index does not exist yet.
    """
    if os.environ.get("CHROMA_HOST"):
        return RagConfig.CHROMA_PATH

    src = Path(RagConfig.CHROMA_PATH)
    if not src.exists():
        return RagConfig.CHROMA_PATH

    runtime = Path(os.environ.get("CHROMA_RUNTIME_PATH", "./data/chroma_runtime"))
    src_db = src / "chroma.sqlite3"
    run_db = runtime / "chroma.sqlite3"

    needs_copy = (not run_db.exists()) or (
        src_db.exists() and src_db.stat().st_mtime > run_db.stat().st_mtime
    )
    if needs_copy:
        if runtime.exists():
            shutil.rmtree(runtime, ignore_errors=True)
        shutil.copytree(src, runtime)

    return str(runtime)
