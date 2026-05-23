# rag/llm/models.py
import os
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def init_models():
    """Initialize LLM + embedding models and apply to global Settings."""
    llm = HuggingFaceLLM(
        model_name=QWEN_MODEL,
        tokenizer_name=QWEN_MODEL,
        device_map="auto",
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.1, "do_sample": False},
    )

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model
