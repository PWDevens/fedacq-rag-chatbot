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
        static_folder="static",
        static_url_path="/static"
    )
    app.config.from_object(get_config(config_name))

    # Mount the API blueprint at root
    app.register_blueprint(api_bp, url_prefix="/")

    # Add a healthcheck route
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
