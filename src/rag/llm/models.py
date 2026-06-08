import os
from llama_index.core import Settings
from rag.llm.phi4_onnx_llm import Phi4OnnxLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PHI_MODEL = "./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def init_models():
    """
    Initialize Phi-4-mini-instruct-onnx (CPU-only, ONNX Runtime GenAI)
    and BGE-small embeddings, then apply them to global LlamaIndex Settings.
    """

    llm = Phi4OnnxLLM(
        model_dir=PHI_MODEL,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
    )

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model
