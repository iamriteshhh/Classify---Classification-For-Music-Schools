"""
app.py
======
Entry point for CLASSIFY — Music Classification System for Music Schools.
Uses the application factory pattern (classify.create_app()) per structure.md §2.
"""

import os
from classify import create_app

# Default application instance
app = create_app()

if __name__ == "__main__":
    # Run development server on port 5000
    print("Starting CLASSIFY on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)
