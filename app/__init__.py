from flask import Flask

# Initialize the app
app = Flask(__name__)

# Import the routes to register them with the app
from app import routes