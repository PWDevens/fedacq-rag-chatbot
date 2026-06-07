import os
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

PHI_MODEL = "microsoft/Phi-4-mini-instruct-onnx"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def init_models():
    """
    Initialize microsoft/Phi-4-mini-instruct-onnx + BGE embeddings
    and apply to global LlamaIndex Settings.

    This configuration uses ONNX Runtime for CPU-only inference.
    """

    llm = HuggingFaceLLM(
        model_name=PHI_MODEL,
        tokenizer_name=PHI_MODEL,
        device_map="cpu",          # ONNXRuntime runs on CPU
        max_new_tokens=256,
        generate_kwargs={
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": False,    # deterministic, compliance-style answers
        },
        model_kwargs={
            "trust_remote_code": True,
            "provider": "CPUExecutionProvider",  # ONNXRuntime backend
        },
    )

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model
