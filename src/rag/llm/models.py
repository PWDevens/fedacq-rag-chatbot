import os
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

PHI_MODEL = "microsoft/Phi-3-mini-128k-instruct"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def init_models():
    """
    Initialize Phi-3-mini-128k-instruct + BGE embeddings
    and apply to global LlamaIndex Settings.
    """

    llm = HuggingFaceLLM(
        model_name=PHI_MODEL,
        tokenizer_name=PHI_MODEL,
        device_map="auto",
        max_new_tokens=256,
        generate_kwargs={
            "do_sample": False,
        },
        model_kwargs={
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "use_cache": False,
        },
    )

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Apply globally
    Settings.llm = llm
    Settings.embed_model = embed_model

    # CRITICAL: Force LlamaIndex to disable sampling
    Settings.llm.generate_kwargs = {"do_sample": False}

    return llm, embed_model
