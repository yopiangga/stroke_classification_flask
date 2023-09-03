#!/usr/bin/env python3

import sys
import logging
from pathlib import Path

# Add the directory containing your Flask app to the Python path
app_directory = Path(__file__).resolve().parents[0]
sys.path.append(str(app_directory))

# Initialize and configure your Flask app
from your_app import app  # Import your Flask app instance

# Configure logging (optional but recommended for debugging)
logging.basicConfig(stream=sys.stderr)
logging.getLogger().setLevel(logging.DEBUG)

# Define the WSGI application
def application(environ, start_response):
    # You can modify the WSGI environment if needed
    # For example, you can set the SCRIPT_NAME to the subpath if your app is not at the root URL
    # environ['SCRIPT_NAME'] = '/myapp'

    # Pass the request to your Flask app
    return app(environ, start_response)

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple("localhost", 4000, application)
