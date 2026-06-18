"""
Pytest configuration for fedacq-rag-chatbot tests.

Sets up test environment variables before any tests run.
"""

import os
import pytest


def pytest_configure(config):
    """
    Set required environment variables for tests before any imports.
    This ensures test environment is ready before modules are loaded.
    """
    # Set FLASK_ENV to 'local' for test runs to avoid production config checks
    if "FLASK_ENV" not in os.environ:
        os.environ["FLASK_ENV"] = "local"

    # Set SECRET_KEY for tests (can be any value since it's not production)
    if "SECRET_KEY" not in os.environ:
        os.environ["SECRET_KEY"] = "test-secret-key-12345"

    # Set CHROMA_PATH to avoid data directory issues in CI
    if "CHROMA_PATH" not in os.environ:
        os.environ["CHROMA_PATH"] = "./data/test_chroma"

    # Set PHI4_MODEL_DIR to avoid requiring model files
    if "PHI4_MODEL_DIR" not in os.environ:
        os.environ["PHI4_MODEL_DIR"] = "./cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"

    # Configure ChromaDB for local/in-process mode (not http-only client mode)
    # This is required for PersistentClient usage in tests
    if "CHROMA_API_IMPL" not in os.environ:
        os.environ["CHROMA_API_IMPL"] = "chromadb.api.segment.SegmentAPI"
