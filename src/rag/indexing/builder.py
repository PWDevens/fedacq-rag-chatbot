# src/rag/indexing/builder.py

from pathlib import Path
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from rag.retrieval.parser_dita import (
    clone_if_needed,
    build_documents_from_repo,
    FAR_REPO_URL,
    DFARS_REPO_URL,
)
from rag.retrieval.metadata import normalize_metadata
from rag.llm.models import init_models
from llama_index.core.node_parser import SentenceSplitter


def build_index(chroma_path=None, test_mode=False):
    """
    Build the FAR/DFARS Chroma index.

    Args:
        chroma_path (str | Path | None): Where to store the Chroma DB.
        test_mode (bool): If True, skip cloning and heavy operations.
    """

    # --- Normalize paths ---
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"

    if chroma_path is None:
        chroma_path = DATA_DIR / "chroma"
    else:
        chroma_path = Path(chroma_path)

    # --- Ensure directories exist ---
    (DATA_DIR / "regs").mkdir(parents=True, exist_ok=True)
    chroma_path.mkdir(parents=True, exist_ok=True)

    # --- Test mode: create a tiny dummy index ---
    if test_mode:
        dummy_nodes = []
        client = chromadb.PersistentClient(path=str(chroma_path))
        coll = client.get_or_create_collection("far_dfars_chroma")

        vs = ChromaVectorStore(chroma_collection=coll)
        storage = StorageContext.from_defaults(vector_store=vs)

        # Create an empty index (valid smoke test)
        VectorStoreIndex(dummy_nodes, storage_context=storage)
        storage.persist()
        return chroma_path

    # --- Full mode: real pipeline ---
    init_models()

    base_dir = DATA_DIR / "regs"
    far_path = base_dir / "far"
    dfars_path = base_dir / "dfars"

    clone_if_needed(FAR_REPO_URL, far_path)
    clone_if_needed(DFARS_REPO_URL, dfars_path)

    far_docs = build_documents_from_repo(far_path, "FAR")
    dfars_docs = build_documents_from_repo(dfars_path, "DFARS")
    documents = far_docs + dfars_docs

    splitter = SentenceSplitter(chunk_size=8192, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents(documents)

    for n in nodes:
        n.metadata = normalize_metadata(n.metadata)

    client = chromadb.PersistentClient(path=str(chroma_path))
    coll = client.get_or_create_collection("far_dfars_chroma")

    vs = ChromaVectorStore(chroma_collection=coll)
    storage = StorageContext.from_defaults(vector_store=vs)

    index = VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    storage.persist()

    return chroma_path

