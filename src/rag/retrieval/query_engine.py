# rag/retrieval/query_engine.py

from app.config import BaseConfig
import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import Settings


def load_query_engine(chroma_path=None, collection="far_dfars_chroma"):
    """
    Load the Chroma index and return a query engine that ALWAYS uses
    the ONNX Phi-4 LLM loaded via init_models().
    """

    # 1. Load ONNX LLM + embedding model into Settings
    from rag.llm.models import init_models
    llm, embed_model = init_models()

    # 2. Resolve Chroma path
    if chroma_path is None:
        chroma_path = BaseConfig.CHROMA_PATH

    # 3. Connect to Chroma
    client = chromadb.PersistentClient(path=chroma_path)
    coll = client.get_or_create_collection(collection)

    # 4. Wrap Chroma in LlamaIndex vector store
    vs = ChromaVectorStore(chroma_collection=coll)
    storage = StorageContext.from_defaults(vector_store=vs)

    # 5. Build index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vs,
        storage_context=storage,
        embed_model=embed_model,   # <-- ensure correct embedding model
    )

    # 6. Return query engine with ONNX LLM attached
    return index.as_query_engine(
        llm=llm,                    # <-- FORCE use of your ONNX LLM
        similarity_top_k=5,
        streaming=True,
        response_mode="compact",
    )
