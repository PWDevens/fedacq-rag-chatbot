import os


class RagConfig:
    CHROMA_PATH = os.environ.get("CHROMA_PATH", "./data/chroma")
    EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    PHI4_MODEL_DIR = os.environ.get(
        "PHI4_MODEL_DIR",
        "./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
    )
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
