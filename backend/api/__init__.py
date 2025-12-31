from flask import Flask
from flask_cors import CORS

def create_app(config):
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    CORS(app, origins=config.flask.cors_origins)
    
    app.config['DEBUG'] = config.flask.debug
    app.config['JSON_SORT_KEYS'] = False
    
    from .routes import register_routes
    register_routes(app, config)
    
    return app