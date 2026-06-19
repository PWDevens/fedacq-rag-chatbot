"""
Query engine loader for the FAR/DFARS RAG pipeline.

Loads the on-disk ChromaDB index (PersistentClient) and returns a LlamaIndex
retriever using the same embedding model used at build time. Set CHROMA_HOST
to opt into a remote HTTP server instead.

Environment variables:
  CHROMA_PATH  - Path to the on-disk ChromaDB directory (default: RagConfig.CHROMA_PATH)
  CHROMA_HOST  - If set, connect to a remote ChromaDB HTTP server instead
  CHROMA_PORT  - Port for the remote server (default: 8000)
"""

import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag.config import RagConfig


def load_query_engine(chroma_path=None, collection="far_dfars_chroma"):
    """
    Load the ChromaDB-backed retriever for the FAR/DFARS index.

    Args:
        chroma_path (str | None): Override for the on-disk ChromaDB directory.
            Defaults to RagConfig.CHROMA_PATH. Ignored when CHROMA_HOST is set.
        collection (str): Name of the ChromaDB collection to load.

    Returns:
        llama_index retriever: Configured with similarity_top_k=8.

    Raises:
        RuntimeError: If the collection is missing (index not built yet) or
            the remote ChromaDB server is not reachable.
    """
    # Default: read the on-disk index built by builder.py (PersistentClient).
    # This matches how the index is created and committed (data/chroma) and
    # needs no separate service. Set CHROMA_HOST to opt into a remote HTTP
    # ChromaDB server instead (for scaled/multi-process deployments).
    chroma_host = os.environ.get("CHROMA_HOST")
    if chroma_host:
        chroma_port = int(os.environ.get("CHROMA_PORT", 8000))
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    else:
        path = chroma_path or RagConfig.CHROMA_PATH
        client = chromadb.PersistentClient(path=str(path))

    # Use get_collection (not get_or_create_collection) at query time: the
    # index must already exist, so a missing/empty index fails loudly here
    # instead of silently returning zero results.
    target = chroma_host or chroma_path or RagConfig.CHROMA_PATH
    try:
        collection_obj = client.get_collection(collection)
    except Exception as exc:
        msg = (
            "ChromaDB collection '" + str(collection) + "' not found. "
            "Build the index first (python -m scripts.build_index) and ensure "
            "CHROMA_PATH points to it (current target: " + str(target) + ")."
        )
        raise RuntimeError(msg) from exc

    vector_store = ChromaVectorStore(chroma_collection=collection_obj)
    storage = StorageContext.from_defaults(vector_store=vector_store)

    # Use the same embedding model as the index-build path so cosine
    # similarity between query vectors and stored vectors is meaningful.
    # bge models expect an asymmetric query instruction prepended at search
    # time (passages are embedded plain at build time). Adding it for bge only.
    _embed_kwargs = {"model_name": RagConfig.EMBED_MODEL_NAME}
    if "bge" in RagConfig.EMBED_MODEL_NAME.lower():
        _embed_kwargs["query_instruction"] = (
            "Represent this sentence for searching relevant passages:"
        )
    embed_model = HuggingFaceEmbedding(**_embed_kwargs)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage,
        embed_model=embed_model,
    )

    return index.as_retriever(similarity_top_k=8)
