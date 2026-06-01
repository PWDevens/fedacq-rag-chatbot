from flask import Flask
from .api import api_bp

def create_app():
    # IMPORTANT: static_folder must point to the same folder used by api_bp
    app = Flask(
        __name__,
        static_folder="static",
        static_url_path="/static", 
        url_prefix="/"
    )

    # Mount the API blueprint at root
    app.register_blueprint(api_bp, url_prefix="/")

    # Add a healthcheck route
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
