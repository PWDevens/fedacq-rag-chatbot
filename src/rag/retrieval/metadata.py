"""
Metadata normalization for ChromaDB storage.

ChromaDB metadata values must be scalars (str, int, float, bool). This module
converts arbitrary Python values from the DITA parsing pipeline into
Chroma-safe string representations.
"""

import json


def normalize_metadata(metadata: dict) -> dict:
    """
    Convert metadata values into ChromaDB-safe scalar strings.

    Transformations applied:
      - None values: key is omitted entirely (not stored as the string "None")
        so that Chroma where-filters for missing keys work correctly.
      - list / dict values: JSON-serialized to a string.
      - All other values: converted via str().

    Args:
        metadata (dict): Raw metadata dict from the DITA parsing pipeline.

    Returns:
        dict[str, str]: Metadata safe for storage in ChromaDB.
    """
    safe = {}
    for key, value in metadata.items():
        if value is None:
            # Omit None values rather than storing the string "None", so
            # callers can use absence-of-key semantics in Chroma filters.
            continue
        elif isinstance(value, (list, dict)):
            safe[key] = json.dumps(value)
        else:
            safe[key] = str(value)
    return safe
