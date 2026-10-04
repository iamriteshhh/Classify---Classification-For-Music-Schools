"""
wsgi.py
=======
Production WSGI entry point for CLASSIFY.
Suitable for running behind Gunicorn, Waitress, or uWSGI.
Example:
    gunicorn "wsgi:app" --bind 0.0.0.0:8000 --workers 4
    waitress-serve --port=8000 wsgi:app
Part of Phase 12 for CLASSIFY.
"""

from classify import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
