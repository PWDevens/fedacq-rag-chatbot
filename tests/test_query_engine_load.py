## tests/test_query_engine_load.py

import os
import pytest

@pytest.mark.skipif(
    not os.environ.get("ENABLE_PHI4_TESTS"),
    reason="Phi-4 model tests skipped unless ENABLE_PHI4_TESTS=1"
)
def test_load_query_engine():
    from rag.retrieval.query_engine import load_query_engine
    qe = load_query_engine()
    assert qe is not None
