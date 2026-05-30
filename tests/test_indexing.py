## tests/test_indexing.py

from rag.indexing.builder import build_index

def test_build_index_smoke():
    try:
        build_index(chroma_path="./test_chroma", test_mode=True)
    except Exception as e:
        assert False, f"build_index raised exception: {e}"