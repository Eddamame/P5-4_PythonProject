import os
from app import create_app
from config import Config 

# Initialize the application using the factory function
# This function registers routes from routes.py and loads config
app = create_app(config_class=Config)

if __name__ == '__main__':
    """
    Starts the Flask web server. 
    This block is the application's entry point, handling the host and port 
    configuration required for deployment environments like Railway.
    """
    
    # Read the dynamic port provided by the host. 
    # It defaults to 5000 for local testing.
    port = int(os.environ.get("PORT", 5000))
    
    # Bind to '0.0.0.0' so the external proxy (Railway) can connect.
    # This resolves the "connection refused" (502) error.
    app.run(
        debug=False,
        host='0.0.0.0', 
        port=port 
    )