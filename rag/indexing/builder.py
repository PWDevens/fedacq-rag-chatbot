# rag/indexing/builder.py
import os
from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import chromadb

from rag.retrieval.parser_dita import (
    clone_if_needed,
    build_documents_from_repo,
    FAR_REPO_URL,
    DFARS_REPO_URL,
)
from rag.retrieval.metadata import normalize_metadata
from rag.llm.models import init_models

def build_index(chroma_path="./data/chroma"):
    init_models()

    base_dir = Path("./regs")
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

    client = chromadb.PersistentClient(path=chroma_path)
    coll = client.get_or_create_collection("far_dfars_chroma")

    vs = ChromaVectorStore(chroma_collection=coll)
    storage = StorageContext.from_defaults(vector_store=vs)

    index = VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    return index
