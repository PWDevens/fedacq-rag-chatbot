import os

class BaseConfig:
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")

    # RAG-specific paths
    CHROMA_PATH = os.environ.get("CHROMA_PATH", "./data/chroma")
    DITA_PATH = os.environ.get("DITA_PATH", "./data/dita")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

    # --- REQUIRED FOR YOUR ONNX LLM ---
    PHI4_MODEL_DIR = os.environ.get(
        "PHI4_MODEL_DIR",
        "./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    )

    # Embedding model used when building your Chroma index
    EMBED_MODEL_NAME = os.environ.get(
        "EMBED_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )


class LocalConfig(BaseConfig):
    DEBUG = True


class ProdConfig(BaseConfig):
    DEBUG = False


def get_config(name):
    return {
        "local": LocalConfig,
        "prod": ProdConfig,
        "default": LocalConfig,
    }.get(name, LocalConfig)
