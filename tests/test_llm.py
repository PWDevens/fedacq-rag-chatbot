from rag.llm import init_models

def test_init_models():
    llm, embed = init_models()
    assert llm is not None
    assert embed is not None

