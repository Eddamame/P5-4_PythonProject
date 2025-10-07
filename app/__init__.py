from flask import Flask
import logging 
from app import routes

# Initialize the app
app = Flask(__name__)

def configure_logging():
    # Set up handlers, formatters, and set the root logger level
    logging.basicConfig(level=logging.INFO, ...)

def create_app():
    # First step: configure logging
    configure_logging() 
    
    # Second step: create the Flask instance
    app = Flask(__name__)
    
    # Third step: register modules/blueprints
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app


