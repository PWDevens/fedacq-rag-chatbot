# rag/retrieval/metadata.py
import json

def normalize_metadata(md: dict):
    """Convert metadata values into Chroma-safe strings."""
    safe = {}
    for k, v in md.items():
        if v is None:
            safe[k] = "None"
        elif isinstance(v, (list, dict)):
            safe[k] = json.dumps(v)
        else:
            safe[k] = str(v)
    return safe
