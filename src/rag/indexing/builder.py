"""
Index builder for the FAR/DFARS RAG pipeline.

Clones the FAR and DFARS regulation repositories, parses DITA source files,
and persists a ChromaDB vector store for use by the query engine at runtime.

ChromaDB Server Configuration:
  - CHROMA_HOST: Host where ChromaDB server is running (default: None - use local)
  - CHROMA_PORT: Port where ChromaDB server is running (default: None - use local)

Usage:
    python -m scripts.build_index
    build_index(test_mode=True)   # for CI / unit tests

To use with ChromaDB server:
    CHROMA_HOST=localhost CHROMA_PORT=8000 python -m scripts.build_index
"""

from pathlib import Path
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag.retrieval.parser_dita import (
    clone_if_needed,
    build_documents_from_repo,
    FAR_REPO_URL,
    DFARS_REPO_URL,
)
from rag.retrieval.metadata import normalize_metadata
from app.config import BaseConfig

# Chunking: split each DITA topic into passages that fit the embedding model's
# context window, instead of embedding whole files (which truncates ~34% of
# topics and dilutes meaning). 512 tokens with overlap keeps clauses coherent.
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64

# Lightweight metadata kept per chunk (used for citations/filtering). Heavy
# parser fields (fillins, rev_markers, fill_types) are dropped before chunking.
_KEEP_METADATA_KEYS = (
    "regulation", "href", "navtitle",
    "part", "subpart", "section", "clause", "source_path",
)

# Rollup/aggregate source files that concatenate many sections (the FAR matrix,
# definitions maps, print bundles, whole-part dumps). They duplicate the
# individual section topics and pollute retrieval, so they are excluded.
def _is_aggregate(href: str) -> bool:
    name = Path(href).stem.lower()
    if any(marker in name for marker in ("matrix", "4map", "_pdf")):
        return True
    if name.startswith("part_"):  # e.g. Part_52.dita duplicates 52.* clauses
        return True
    return False


def _get_chroma_client():
    """
    Get ChromaDB client, using HTTP if server is configured, else local PersistentClient.

    Returns:
        chromadb.Client: Either HttpClient (if CHROMA_HOST set) or PersistentClient.
    """
    chroma_host = os.environ.get("CHROMA_HOST")
    if chroma_host:
        # Use HTTP client for remote ChromaDB server
        chroma_port = int(os.environ.get("CHROMA_PORT", 8000))
        return chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        # Use local persistent client (legacy mode)
        return None  # Will be handled by caller with path


def build_index(chroma_path=None, test_mode=False):
    """
    Build and persist the FAR/DFARS ChromaDB vector index.

    Args:
        chroma_path (str | Path | None): Directory for ChromaDB persistence.
            Defaults to <project_root>/data/chroma.
        test_mode (bool): When True (or when RAG_TEST_MODE=1 env var is set),
            skips repository cloning and DITA parsing and writes a minimal
            two-node index for CI/unit-test purposes.

    Side effects:
        - Creates chroma_path directory if it does not exist.
        - Clones FAR/DFARS repos under <project_root>/data/regs/ (real mode).
        - Writes ChromaDB files under chroma_path.
    """
    is_test_mode = test_mode or os.getenv("RAG_TEST_MODE") == "1"

    if is_test_mode:
        Settings.embed_model = MockEmbedding(embed_dim=8)
        Settings.llm = None
        os.environ["IS_TESTING"] = "1"

    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"

    if chroma_path is None:
        chroma_path = data_dir / "chroma"

    (data_dir / "regs").mkdir(parents=True, exist_ok=True)
    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    if is_test_mode:
        _build_test_index(chroma_path)
        return

    _build_production_index(data_dir, chroma_path)


def _build_test_index(chroma_path: Path) -> None:
    """
    Write a minimal two-node ChromaDB index for test/CI use.

    Args:
        chroma_path (Path): Directory for ChromaDB persistence (ignored if using HTTP).
    """
    dummy_nodes = [
        TextNode(text="dummy FAR text", id_="1"),
        TextNode(text="dummy DFARS text", id_="2"),
    ]

    # Use HTTP client if configured, else local persistent client
    client = _get_chroma_client()
    if client is None:
        client = chromadb.PersistentClient(path=str(chroma_path))

    collection = _fresh_collection(client)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(dummy_nodes, storage_context=storage)


def _fresh_collection(client, name: str = "far_dfars_chroma"):
    """
    Return an empty collection, dropping any existing one first.

    A rebuild must start clean: appending to an existing collection would mix
    vectors from a previous (possibly different) embedding model with the new
    ones, silently corrupting similarity search.
    """
    try:
        client.delete_collection(name)
    except Exception:
        pass  # collection may not exist yet
    return client.get_or_create_collection(name)


def _build_production_index(data_dir: Path, chroma_path: Path) -> None:
    """
    Clone FAR/DFARS repos, parse DITA files, and build the full vector index.

    Args:
        data_dir (Path): Root data directory; repos cloned under data_dir/regs/.
        chroma_path (Path): Directory for ChromaDB persistence (ignored if using HTTP).
    """
    # Embedding model only — the LLM is not needed to build the index.
    # Must match the model used by the query path (app.config.EMBED_MODEL_NAME).
    Settings.embed_model = HuggingFaceEmbedding(model_name=BaseConfig.EMBED_MODEL_NAME)

    base_dir = data_dir / "regs"
    far_path = base_dir / "far"
    dfars_path = base_dir / "dfars"

    clone_if_needed(FAR_REPO_URL, far_path)
    clone_if_needed(DFARS_REPO_URL, dfars_path)

    far_docs = build_documents_from_repo(far_path, "FAR")
    dfars_docs = build_documents_from_repo(dfars_path, "DFARS")
    documents = far_docs + dfars_docs

    # Drop rollup/aggregate files that duplicate and dilute individual sections.
    documents = [d for d in documents if not _is_aggregate(d.metadata.get("href", ""))]

    # Trim metadata to lightweight, retrieval-relevant scalars before chunking.
    # The parser also attaches large fill-in / revision-marker lists that bloat
    # each chunk's metadata past the chunk size (the SentenceSplitter is
    # metadata-aware) and are unused at query time.
    for d in documents:
        d.metadata = {
            k: v for k, v in d.metadata.items()
            if k in _KEEP_METADATA_KEYS and v is not None
        }
        # Keep file paths/hrefs out of the embedded text — they add noise.
        d.excluded_embed_metadata_keys = ["href", "source_path"]
        d.excluded_llm_metadata_keys = ["href", "source_path"]

    # Chunk each topic into overlapping passages that fit the embedding window,
    # then normalize each chunk's metadata to Chroma-safe scalars.
    splitter = SentenceSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    for node in nodes:
        node.metadata = normalize_metadata(node.metadata)

    # Use HTTP client if configured, else local persistent client
    client = _get_chroma_client()
    if client is None:
        client = chromadb.PersistentClient(path=str(chroma_path))

    collection = _fresh_collection(client)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    index.storage_context.persist()
