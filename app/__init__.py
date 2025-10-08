# P5-4_PythonProject/app/__init__.py

from flask import Flask

def create_app():
    app = Flask(__name__)

    # ... app configuration (e.g., app.config, secret key) ...
    
    # 1. IMPORT the routes module from the local package
    # The leading dot (.) indicates a relative import within the 'app' package.
    from . import routes 
    
    # Alternatively, and often clearer if using Blueprints directly:
    from .routes import main as main_blueprint 
    
    # 2. REGISTER the Blueprint with the application instance
    app.register_blueprint(main_blueprint)

    return app