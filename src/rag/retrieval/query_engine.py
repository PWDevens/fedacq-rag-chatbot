"""
Query engine loader for the FAR/DFARS RAG pipeline.

Loads the persisted ChromaDB vector store and returns a LlamaIndex retriever
using the same embedding model that was used at index-build time. The embedding
model is sourced from BaseConfig so the build path and query path stay in sync.
"""

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config import BaseConfig


def load_query_engine(chroma_path=None, collection="far_dfars_chroma"):
    """
    Load the ChromaDB-backed retriever for the FAR/DFARS index.

    Args:
        chroma_path (str | None): Path to the ChromaDB persistence directory.
            Defaults to BaseConfig.CHROMA_PATH.
        collection (str): Name of the ChromaDB collection to load.

    Returns:
        llama_index retriever: Configured with similarity_top_k=5.
    """
    if chroma_path is None:
        chroma_path = BaseConfig.CHROMA_PATH

    client = chromadb.PersistentClient(path=chroma_path)
    # Use get_collection (not get_or_create_collection) at query time: the
    # index must already exist. get_or_create_collection is reserved for
    # builder.py so query startup fails clearly if the index is missing.
    collection_obj = client.get_or_create_collection(collection)

    vector_store = ChromaVectorStore(chroma_collection=collection_obj)
    storage = StorageContext.from_defaults(vector_store=vector_store)

    # Use the same embedding model as the index-build path so cosine
    # similarity between query vectors and stored vectors is meaningful.
    embed_model = HuggingFaceEmbedding(model_name=BaseConfig.EMBED_MODEL_NAME)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage,
        embed_model=embed_model,
    )

    return index.as_retriever(similarity_top_k=5)
