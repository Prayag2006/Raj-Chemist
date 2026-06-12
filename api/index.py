"""
Vercel serverless entry point.
This file re-exports the Flask `app` object so Vercel can discover it
without changing the existing project structure.
"""
from app import app  # noqa: F401
