import os
import pytest

@pytest.mark.skipif(
    not os.environ.get("ENABLE_PHI4_TESTS"),
    reason="Phi-4 model tests skipped unless ENABLE_PHI4_TESTS=1"
)
def test_init_models():
    from rag.llm import init_models
    llm, embed = init_models()
    assert llm is not None
    assert embed is not None
