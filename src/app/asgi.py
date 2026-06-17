"""
ASGI entrypoint for Hypercorn.

Wraps the synchronous Flask WSGI app with AsyncioWSGIMiddleware so it can be
served by an ASGI server (Hypercorn). The config is selected from the
FLASK_ENV environment variable; defaults to 'local' if unset.

Usage:
    hypercorn --bind 0.0.0.0:7860 src.app.asgi:app
"""

import os
from app import create_app
from hypercorn.middleware import AsyncioWSGIMiddleware

# Select config based on environment; default to local (non-debug for prod
# requires FLASK_ENV=production and SECRET_KEY to be set).
_config_name = os.environ.get("FLASK_ENV", "local")

# Create the Flask WSGI app
flask_app = create_app(_config_name)

# Wrap it as an ASGI app for Hypercorn
app = AsyncioWSGIMiddleware(flask_app)
