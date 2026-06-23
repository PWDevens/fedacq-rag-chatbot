"""
ASGI entrypoint for Hypercorn.

Wraps the synchronous Flask WSGI app with AsyncioWSGIMiddleware so it can be
served by an ASGI server (Hypercorn). The config is selected from the
FLASK_ENV environment variable; defaults to 'local' if unset.

Usage:
    hypercorn --bind 0.0.0.0:7860 src.app.asgi:app
"""

import logging
import os
from app import create_app
from hypercorn.middleware import AsyncioWSGIMiddleware

# Select config based on environment; default to local (non-debug for prod
# requires FLASK_ENV=production and SECRET_KEY to be set).
_config_name = os.environ.get("FLASK_ENV", "local")

# Create the Flask WSGI app
flask_app = create_app(_config_name)

# Warm the models + retrieval engine at startup so the first request doesn't pay
# the 15-30s cold-start cost. Best-effort: log and continue if warming fails
# (e.g. graph mode with no prebuilt graph) so the server still starts and the
# error surfaces on the first request instead. Disable with RAG_NO_WARM=1.
if os.environ.get("RAG_NO_WARM") != "1":
    try:
        from app.api import warm_start
        warm_start()
    except Exception as _warm_err:  # noqa: BLE001 - best-effort warmup
        logging.getLogger(__name__).warning("warm_start failed: %s", _warm_err)

# Wrap it as an ASGI app for Hypercorn
app = AsyncioWSGIMiddleware(flask_app)
