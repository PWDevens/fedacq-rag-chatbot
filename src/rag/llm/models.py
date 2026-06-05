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
        max_new_tokens=256,  # tactical cap to prevent rambling
        generate_kwargs={
            # Have to comment out generage kwargs uue to incompatibility between Phi‑3‑mini‑128k‑instruct and the version of Transformers that LlamaIndex     
            #"temperature": 0.1,
            #"top_p": 0.9,
            "do_sample": False,  # deterministic, compliance-style answers
        },
        model_kwargs={
            "trust_remote_code": True,
            "torch_dtype": "auto",
        },
    )

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model
