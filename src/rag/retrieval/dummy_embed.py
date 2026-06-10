from typing import ClassVar
from llama_index.core.embeddings import BaseEmbedding

class DummyEmbedding(BaseEmbedding):
    """A no-op embedding model for retrieval-only pipelines."""

    DIM: ClassVar[int] = 384  # <-- MATCHES YOUR CHROMA COLLECTION

    # ---- PUBLIC METHODS ----
    def get_text_embedding(self, text: str):
        return [0.0] * self.DIM

    async def aget_text_embedding(self, text: str):
        return self.get_text_embedding(text)

    def get_query_embedding(self, query: str):
        return [0.0] * self.DIM

    async def aget_query_embedding(self, query: str):
        return self.get_query_embedding(query)

    # ---- REQUIRED INTERNAL METHODS ----
    def _get_text_embedding(self, text: str):
        return self.get_text_embedding(text)

    async def _aget_text_embedding(self, text: str):
        return self.get_text_embedding(text)

    def _get_query_embedding(self, query: str):
        return self.get_query_embedding(query)

    async def _aget_query_embedding(self, query: str):
        return self.get_query_embedding(query)
