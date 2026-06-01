from app import create_app
from hypercorn.middleware import AsyncioWSGIMiddleware

# Create the Flask WSGI app
flask_app = create_app()

# Wrap it as an ASGI app for Hypercorn
app = AsyncioWSGIMiddleware(flask_app)
