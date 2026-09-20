"""
WSGI entry point for production servers.

The development server (``python app.py``) is single-threaded and explicitly
not suitable for real traffic. In production a WSGI server such as gunicorn
imports this module and serves the ``app`` object:

    gunicorn wsgi:app

``create_app()`` is called at import time so the model and vectorizer are
loaded once per worker process at start-up, rather than on the first request.
"""

from app import app, create_app

# Initialise directories and load the model artefacts.
create_app()

# gunicorn looks for a module-level WSGI callable; both names are provided
# because different platforms default to different ones.
application = app

__all__ = ["app", "application"]
