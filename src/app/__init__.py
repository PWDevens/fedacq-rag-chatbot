from flask import Flask
from .config import get_config
from .api import api_bp

def create_app(config_name="default"):
    """
    Application factory pattern.
    Creates the Flask app and mounts the API blueprint.
    """

    # Create Flask app
    app = Flask(
        __name__,
        static_folder="static",      # ensures Flask knows where static files live
        static_url_path="/static"
    )
    app.config.from_object(get_config(config_name))

    # Mount the API blueprint at root
    # This ensures:
    #   GET /            → index.html
    #   POST /chat_stream → works
    #   GET /static/...   → works
    app.register_blueprint