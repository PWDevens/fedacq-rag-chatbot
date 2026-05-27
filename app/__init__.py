## app/__init__.py

from flask import Flask
from .config import get_config

def create_app(config_name="default"):
    """
    Application factory pattern.
    Allows different configs for local/dev/prod.
    """
    app = Flask(__name__)
    app.config.from_object(get_config(config_name))

    # Import and register routes
    from .api import api_bp
    app.register_blueprint(api_bp)

    return app

# For convenience when running locally:
app = create_app()
