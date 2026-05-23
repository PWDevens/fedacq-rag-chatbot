from rag.indexing.builder import build_index

def test_build_index_smoke():
    # Smoke test: ensure function runs without raising
    try:
        build_index(chroma_path="./test_chroma")
    except Exception as e:
        assert False, f"build_index raised exception: {e}"

