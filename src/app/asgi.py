# src/app/asgi.py

import os
from hypercorn.asyncio import serve
from hypercorn.config import Config
from app import create_app

# Create the Flask app (Flask 3.x supports ASGI natively)
app = create_app()

# Expose ASGI callable
asgi_app = app

if __name__ == "__main__":
    config = Config()
    config.bind = ["0.0.0.0:7860"]
    config.worker_class = "asyncio"
    config.use_reloader = False

    import asyncio
    asyncio.run(serve(asgi_app, config))

