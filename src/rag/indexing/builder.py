## src/rag/indexing/builder.py

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
    # -------------------------
    # Test Mode
    # -------------------------
    if test_mode or os.getenv("RAG_TEST_MODE") == "1":
        Settings.embed_model = MockEmbedding(embed_dim=8)
        Settings.llm = None
        os.environ["IS_TESTING"] = "1"

    # -------------------------
    # Paths
    # -------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"

    if chroma_path is None:
        chroma_path = DATA_DIR / "chroma"

    (DATA_DIR / "regs").mkdir(parents=True, exist_ok=True)
    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Test Mode: Skip cloning + parsing
    # -------------------------
    if test_mode or os.getenv("RAG_TEST_MODE") == "1":
        dummy_nodes = [
            TextNode(text="dummy FAR text", id_="1"),
            TextNode(text="dummy DFARS text", id_="2"),
        ]
        client = chromadb.PersistentClient(path=str(chroma_path))
        coll = client.get_or_create_collection("far_dfars_chroma")
        vs = ChromaVectorStore(chroma_collection=coll)
        storage = StorageContext.from_defaults(vector_store=vs)
        VectorStoreIndex(dummy_nodes, storage_context=storage)
        return

    # -------------------------
    # Real Mode
    # -------------------------
    init_models()

    base_dir = DATA_DIR / "regs"
    far_path = base_dir / "far"
    dfars_path = base_dir / "dfars"

    # Clone FAR + DFARS repos if missing
    clone_if_needed(FAR_REPO_URL, far_path)
    clone_if_needed(DFARS_REPO_URL, dfars_path)

    # Build one Document per DITA file (already semantically chunked)
    far_docs = build_documents_from_repo(far_path, "FAR")
    dfars_docs = build_documents_from_repo(dfars_path, "DFARS")
    documents = far_docs + dfars_docs

    # Convert each Document into a TextNode directly
    nodes = []
    for doc in documents:
        meta = normalize_metadata(doc.metadata)
        nodes.append(TextNode(text=doc.text, metadata=meta))

    # -------------------------
    # Build Chroma Index
    # -------------------------
    client = chromadb.PersistentClient(path=str(chroma_path))
    coll = client.get_or_create_collection("far_dfars_chroma")
    vs = ChromaVectorStore(chroma_collection=coll)
    storage = StorageContext.from_defaults(vector_store=vs)

    index = VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    index.storage_context.persist()
