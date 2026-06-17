"""
Index builder for the FAR/DFARS RAG pipeline.

Clones the FAR and DFARS regulation repositories, parses DITA source files,
and persists a ChromaDB vector store for use by the query engine at runtime.

Usage:
    python -m scripts.build_index
    build_index(test_mode=True)   # for CI / unit tests
"""

from pathlib import Path
import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.embeddings import MockEmbedding

from rag.retrieval.parser_dita import (
    clone_if_needed,
    build_documents_from_repo,
    FAR_REPO_URL,
    DFARS_REPO_URL,
)
from rag.retrieval.metadata import normalize_metadata
from rag.llm.models import init_models


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
        chroma_path (Path): Directory for ChromaDB persistence.
    """
    dummy_nodes = [
        TextNode(text="dummy FAR text", id_="1"),
        TextNode(text="dummy DFARS text", id_="2"),
    ]
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection("far_dfars_chroma")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(dummy_nodes, storage_context=storage)


def _build_production_index(data_dir: Path, chroma_path: Path) -> None:
    """
    Clone FAR/DFARS repos, parse DITA files, and build the full vector index.

    Args:
        data_dir (Path): Root data directory; repos cloned under data_dir/regs/.
        chroma_path (Path): Directory for ChromaDB persistence.
    """
    init_models()

    base_dir = data_dir / "regs"
    far_path = base_dir / "far"
    dfars_path = base_dir / "dfars"

    clone_if_needed(FAR_REPO_URL, far_path)
    clone_if_needed(DFARS_REPO_URL, dfars_path)

    # One Document per DITA file; each file is already a semantically coherent
    # regulation section, so we treat it as a single chunk.
    far_docs = build_documents_from_repo(far_path, "FAR")
    dfars_docs = build_documents_from_repo(dfars_path, "DFARS")
    documents = far_docs + dfars_docs

    nodes = []
    for doc in documents:
        meta = normalize_metadata(doc.metadata)
        nodes.append(TextNode(text=doc.text, metadata=meta))

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection("far_dfars_chroma")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    index.storage_context.persist()
