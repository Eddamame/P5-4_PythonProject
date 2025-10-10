# This file serves as the serverless function entry point for Vercel.
from app.routes import create_app

app = create_app()