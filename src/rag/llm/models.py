from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag.config import RagConfig
from rag.llm.phi4_onnx_llm import Phi4OnnxLLM


def init_models():
    """
    Initialize and register the global LLM and embedding model with LlamaIndex.

    Uses Settings.__dict__ for the existence check rather than the Settings.llm
    property accessor, because the property accessor auto-initializes an OpenAI
    client when no LLM has been configured — a side effect we want to avoid in
    a fully local ONNX deployment.

    Returns:
        tuple[Phi4OnnxLLM, HuggingFaceEmbedding]: (llm, embed_model)
    """

    # Check via __dict__ to avoid the property getter, which triggers OpenAI
    # client initialization as a side effect when Settings._llm is unset.
    existing_llm = Settings.__dict__.get("_llm", None)
    existing_embed = Settings.__dict__.get("_embed_model", None)

    if existing_llm is not None and existing_embed is not None:
        return existing_llm, existing_embed

    # ONNX Phi-4 LLM
    llm = Phi4OnnxLLM(
        model_dir=RagConfig.PHI4_MODEL_DIR,
        max_new_tokens=128,
        temperature=0.1,
        top_p=0.9,
    )

    # Embedding model — must match the model used at index-build time.
    embed_model = HuggingFaceEmbedding(
        model_name=RagConfig.EMBED_MODEL_NAME
    )

    # Assign via private attributes rather than the property setters to avoid
    # triggering OpenAI auto-initialization in LlamaIndex's Settings class.
    Settings._llm = llm
    Settings._embed_model = embed_model

    return llm, embed_model
