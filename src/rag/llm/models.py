from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config import BaseConfig
from rag.llm.phi4_onnx_llm import Phi4OnnxLLM


def init_models():
    """
    Initialize and register the global LLM + embedding model.
    Returns: (llm, embed_model)
    """

    # SAFE CHECK — do NOT access Settings.llm (it triggers OpenAI)
    existing_llm = Settings.__dict__.get("_llm", None)
    existing_embed = Settings.__dict__.get("_embed_model", None)

    if existing_llm is not None and existing_embed is not None:
        return existing_llm, existing_embed

    # ONNX Phi-4 LLM
    llm = Phi4OnnxLLM(
        model_dir=BaseConfig.PHI4_MODEL_DIR,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
    )

    # Embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=BaseConfig.EMBED_MODEL_NAME
    )

    # Register WITHOUT triggering OpenAI
    Settings._llm = llm
    Settings._embed_model = embed_model

    return llm, embed_model
