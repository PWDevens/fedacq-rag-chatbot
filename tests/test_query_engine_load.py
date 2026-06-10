import os
import pytest
from rag.retrieval.query_engine import load_query_engine

@pytest.mark.skipif(
    not os.environ.get("ENABLE_PHI4_TESTS"),
    reason="Phi-4 model tests skipped unless ENABLE_PHI4_TESTS=1"
)
def test_query_engine_initialization():
    qe = load_query_engine()
    assert qe is not None
    assert callable(getattr(qe, "retrieve", None))
