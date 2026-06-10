from app.config import BaseConfig
import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

from rag.retrieval.dummy_embed import DummyEmbedding   # <-- NEW


def load_query_engine(chroma_path=None, collection="far_dfars_chroma"):

    if chroma_path is None:
        chroma_path = BaseConfig.CHROMA_PATH

    client = chromadb.PersistentClient(path=chroma_path)
    coll = client.get_or_create_collection(collection)

    vs = ChromaVectorStore(chroma_collection=coll)
    storage = StorageContext.from_defaults(vector_store=vs)

    # use dummy embedding model to prevent auto-loading of OpenAI/HF embedding models, which can cause issues if not properly configured
    dummy = DummyEmbedding()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vs,
        storage_context=storage,
        embed_model=dummy,     # prevents OpenAI/HF auto-load
    )

    return index.as_retriever(similarity_top_k=5)
