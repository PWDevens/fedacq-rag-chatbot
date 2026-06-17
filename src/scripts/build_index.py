"""
Entry point for the offline FAR/DFARS index build.

Clones the FAR and DFARS regulation repositories, parses all DITA source
files, and persists the ChromaDB vector index used by the query engine at
runtime.

Usage:
    python -m scripts.build_index
"""

import logging
from rag.indexing.builder import build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Building FAR/DFARS index...")
    build_index()
    logger.info("Index build complete.")

