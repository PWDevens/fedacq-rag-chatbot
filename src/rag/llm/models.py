import os
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

PHI_MODEL = "microsoft/Phi-4-mini-flash-reasoning"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def init_models():
    """
    Initialize Phi-4-mini-flash-reasoning (4-bit) + BGE embeddings
    and apply to global LlamaIndex Settings.
    """

    llm = HuggingFaceLLM(
        model_name=PHI_MODEL,
        tokenizer_name=PHI_MODEL,
        device_map="auto",
        # Tactical cap to avoid rambling; generous but bounded
        max_new_tokens=256,
        # Deterministic, compliance-style answers
        generate_kwargs={
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": False,
        },
        # 4-bit quantization + remote code for Phi-4
        model_kwargs={
            "trust_remote_code": True,
            "load_in_4bit": True,
        },
    )

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model
