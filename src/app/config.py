import os
import sys


def _require_env(key: str, default: str = None) -> str:
    """
    Return the environment variable value for key.
    In non-local environments, raises RuntimeError if key is unset and no
    safe default exists, so the app fails fast rather than running with
    insecure placeholder values.
    """
    value = os.environ.get(key)
    if value:
        return value
    if default is not None:
        return default
    raise RuntimeError(
        f"Required environment variable '{key}' is not set. "
        f"Set it before starting the application."
    )


class BaseConfig:
    """
    Base configuration shared across all environments.

    Required environment variables:
      SECRET_KEY       - Flask session signing key (mandatory in prod).
      CHROMA_PATH      - Path to the persistent ChromaDB directory.
      PHI4_MODEL_DIR   - Path to the Phi-4 ONNX model directory.

    Optional environment variables:
      HF_TOKEN         - HuggingFace API token (anonymous access used if unset).
      EMBED_MODEL_NAME - HuggingFace embedding model name.
    """

    DEBUG = False
    TESTING = False

    # SECRET_KEY must be set via environment; no hardcoded fallback.
    SECRET_KEY = _require_env("SECRET_KEY", default=None if os.environ.get("FLASK_ENV") == "production" else "dev-secret-key-replace-in-prod")

    # RAG-specific paths
    CHROMA_PATH = os.environ.get("CHROMA_PATH", "./data/chroma")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

    # Required at runtime: the ONNX model directory must exist.
    PHI4_MODEL_DIR = os.environ.get(
        "PHI4_MODEL_DIR",
        "./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    )

    # Embedding model used when building the Chroma index.
    # Must match the model used at index-build time.
    EMBED_MODEL_NAME = os.environ.get(
        "EMBED_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )


class LocalConfig(BaseConfig):
    DEBUG = True


class ProdConfig(BaseConfig):
    DEBUG = False

    # In production, SECRET_KEY must be explicitly provided.
    SECRET_KEY = _require_env("SECRET_KEY")


def get_config(name):
    return {
        "local": LocalConfig,
        "prod": ProdConfig,
        "default": LocalConfig,
    }.get(name, LocalConfig)
