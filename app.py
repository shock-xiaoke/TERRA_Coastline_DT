"""
Backward compatibility wrapper for app.py
Redirects to the new modular structure
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from terra_ugla.app import app

if __name__ == '__main__':
    print("=" * 60)
    print("NOTICE: Using restructured TERRA UGLA application")
    print("The codebase has been reorganized into a modular structure")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
