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

    This configuration assumes a recent transformers version
    compatible with Phi-3 (e.g. >= 4.41).
    """

    llm = HuggingFaceLLM(
        model_name=PHI_MODEL,
        tokenizer_name=PHI_MODEL,
        device_map="auto",
        max_new_tokens=256,  # keep responses tight and fast
        generate_kwargs={
            # deterministic, compliance-style answers
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": False,
        },
        model_kwargs={
            "trust_remote_code": True,
            "torch_dtype": "auto",
            # use eager attention to avoid flash-attn issues
            "attn_implementation": "eager",
        },
    )

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model
