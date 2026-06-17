from flask import Flask
from .config import get_config
from .api import api_bp


def create_app(config_name="default"):
    """
    Application factory.

    Creates and configures the Flask application, then registers the API
    blueprint which provides all routes including /health, /chat_stream,
    and the static frontend.

    Args:
        config_name (str): Config profile to load — "local", "prod", or
            "default" (maps to LocalConfig). Defaults to "default".

    Returns:
        Flask: Configured application instance.
    """
    app = Flask(
        __name__,
        static_folder="static",
        static_url_path="/static"
    )
    app.config.from_object(get_config(config_name))

    # All routes (/, /chat_stream, /health, /static) are defined on api_bp.
    app.register_blueprint(api_bp, url_prefix="/")

    return app
