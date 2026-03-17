"""
Main entry point for running the TERRA UGLA application
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from terra_ugla.app import app

if __name__ == '__main__':
    print("Starting TERRA UGLA application...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
