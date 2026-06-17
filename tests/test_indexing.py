import pytest
from rag.indexing.builder import build_index


def test_build_index_smoke(tmp_path):
    chroma_path = tmp_path / "test_chroma"

    # Allow the exception to propagate naturally so pytest shows the full
    # traceback rather than a generic AssertionError.
    build_index(chroma_path=str(chroma_path), test_mode=True)

    assert chroma_path.exists()
