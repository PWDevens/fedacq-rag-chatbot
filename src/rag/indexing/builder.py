## src/rag/indexing/builder.py

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

def build_index(chroma_path=None):
    # Correct project root (one level higher)
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"

    # Default chroma path
    if chroma_path is None:
        chroma_path = DATA_DIR / "chroma"

    # Ensure directories exist
    (DATA_DIR / "regs").mkdir(parents=True, exist_ok=True)
    chroma_path.mkdir(parents=True, exist_ok=True)

    # Initialize models
    init_models()

    # FAR/DFARS repo paths
    base_dir = DATA_DIR / "regs"
    far_path = base_dir / "far"
    dfars_path = base_dir / "dfars"

    # Clone if needed
    clone_if_needed(FAR_REPO_URL, far_path)
    clone_if_needed(DFARS_REPO_URL, dfars_path)

    # Build documents
    far_docs = build_documents_from_repo(far_path, "FAR")
    dfars_docs = build_documents_from_repo(dfars_path, "DFARS")
    documents = far_docs + dfars_docs

    # Chunking
    splitter = SentenceSplitter(chunk_size=8192, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents(documents)
    for n in nodes:
        n.metadata = normalize_metadata(n.metadata)

    # Build Chroma index
    client = chromadb.PersistentClient(path=str(chroma_path))
    coll = client.get_or_create_collection("far_dfars_chroma")

    vs = ChromaVectorStore(chroma_collection=coll)
    storage = StorageContext.from_defaults(vector_store=vs)

    index = VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    index.storage_context.persist()
