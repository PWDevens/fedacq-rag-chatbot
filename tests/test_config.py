import importlib
import os


def test_ragconfig_defaults_without_env(monkeypatch):
    monkeypatch.delenv("EMBED_MODEL_NAME", raising=False)
    monkeypatch.delenv("CHROMA_PATH", raising=False)
    monkeypatch.delenv("PHI4_MODEL_DIR", raising=False)
    import rag.config as cfg
    importlib.reload(cfg)
    assert cfg.RagConfig.EMBED_MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert cfg.RagConfig.CHROMA_PATH == "./data/chroma"
