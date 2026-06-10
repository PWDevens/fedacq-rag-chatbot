# rag/retrieval/query_engine.py

from app.config import BaseConfig
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import chromadb


def load_query_engine(chroma_path=None, collection="far_dfars_chroma"):
    """
    Load an existing Chroma index and return a LlamaIndex query engine.

    This version is compatible with:
    - ONNX Runtime GenAI LLM (Phi-4)
    - LlamaIndex 0.14.x
    - Flask SSE streaming
    """

    # Initialize global LLM + embedding models
    from rag.llm.models import init_models
    init_models()

    # Resolve Chroma path
    if chroma_path is None:
        chroma_path = BaseConfig.CHROMA_PATH

    # Connect to Chroma
    client = chromadb.PersistentClient(path=chroma_path)
    coll = client.get_or_create_collection(collection)

    # Wrap Chroma in LlamaIndex vector store
    vs = ChromaVectorStore(chroma_collection=coll)
    storage = StorageContext.from_defaults(vector_store=vs)

    # Build index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vs,
        storage_context=storage,
    )

    # Return a query engine configured for:
    # - top_k retrieval
    # - streaming enabled (backend handles token streaming)
    return index.as_query_engine(
        similarity_top_k=5,
        streaming=True,
        response_mode="compact",   # ensures clean text output
    )
