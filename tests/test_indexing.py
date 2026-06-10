import shutil
from rag.indexing.builder import build_index

def test_build_index_smoke(tmp_path):
    chroma_path = tmp_path / "test_chroma"

    try:
        build_index(chroma_path=str(chroma_path), test_mode=True)
    except Exception as e:
        assert False, f"build_index raised exception: {e}"

    # Ensure index directory was created
    assert chroma_path.exists()
