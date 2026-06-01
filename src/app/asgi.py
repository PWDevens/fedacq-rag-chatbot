from app import create_app
from hypercorn.middleware import wsgi

# Create the Flask WSGI app
flask_app = create_app()

# Wrap it as an ASGI app for Hypercorn
app = wsgi.WSGIMiddleware(flask_app)
