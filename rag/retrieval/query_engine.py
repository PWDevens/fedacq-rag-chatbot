# rag/retrieval/query_engine.py
from app.config import BaseConfig
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import chromadb
from .metadata import normalize_metadata
from rag.llm.models import init_models

def load_query_engine(chroma_path=None, collection="far_dfars_chroma"):
    """Load an existing Chroma index and return a query engine."""
    if chroma_path is None:
        chroma_path = BaseConfig.CHROMA_PATH
    
    init_models()

    client = chromadb.PersistentClient(path=chroma_path)
    coll = client.get_or_create_collection(collection)

    vs = ChromaVectorStore(chroma_collection=coll)
    storage = StorageContext.from_defaults(vector_store=vs)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vs,
        storage_context=storage,
    )

    return index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )
