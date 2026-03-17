#!/usr/bin/env python3
"""
TERRA - Minimal Flask App for Testing (NumPy 2.x compatible)
This version excludes the advanced VE detection to avoid import issues
"""
from flask import Flask, render_template, request, jsonify, send_file
import requests
import base64
import json
import os
from datetime import datetime, timedelta
import numpy as np
from sentinelhub.api.process import SentinelHubRequest
from sentinelhub.data_collections import DataCollection
from sentinelhub.constants import MimeType, CRS
from sentinelhub.geometry import BBox
from sentinelhub.api.catalog import SentinelHubCatalog
from sentinelhub.config import SHConfig
from sentinelhub.geo_utils import bbox_to_dimensions
from sentinelhub import SentinelHubRequest, DataCollection, MimeType, BBox, CRS, bbox_to_dimensions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.plot import show
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Create directories for storing data
os.makedirs('data/shorelines', exist_ok=True)
os.makedirs('data/transects', exist_ok=True)
os.makedirs('data/satellite_images', exist_ok=True)
os.makedirs('data/exports', exist_ok=True)

# Sentinel Hub configuration (read from config file)
config = SHConfig()

def load_sentinel_hub_config():
    """Load Sentinel Hub configuration from config.json file"""
    config_file = 'config.json'

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Configuration file '{config_file}' not found. "
            f"Please copy 'config.example.json' to '{config_file}' and add your Sentinel Hub credentials."
        )

    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        sentinel_config = config_data.get('sentinel_hub', {})

        if not sentinel_config.get('client_id') or sentinel_config.get('client_id') == 'your_client_id_here':
            raise ValueError("Please set your Sentinel Hub client_id in config.json")

        if not sentinel_config.get('client_secret') or sentinel_config.get('client_secret') == 'your_client_secret_here':
            raise ValueError("Please set your Sentinel Hub client_secret in config.json")

        return sentinel_config

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config.json: {e}")
    except Exception as e:
        raise Exception(f"Error loading configuration: {e}")

# Load Sentinel Hub configuration
sentinel_hub_available = False
try:
    sentinel_config = load_sentinel_hub_config()

    config = SHConfig()
    config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
    config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    config.sh_auth_base_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect'
    config.sh_client_id = sentinel_config['client_id']
    config.sh_client_secret = sentinel_config['client_secret']

    sentinel_hub_available = True
    print("Sentinel Hub configuration loaded successfully")
except Exception as e:
    print(f"Warning: Sentinel Hub configuration error: {e}")
    print("Satellite data functionality will use demo mode")
    sentinel_hub_available = False

@app.route('/')
def index():
    """Main page with the interactive map"""
    return render_template('index.html')

@app.route('/save_shoreline', methods=['POST'])
def save_shoreline():
    """Save the drawn shoreline coordinates"""
    try:
        data = request.get_json()
        coordinates = data.get('coordinates', [])
        name = data.get('name', f'shoreline_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

        if len(coordinates) < 2:
            return jsonify({'error': 'At least 2 points required for a shoreline'}), 400

        # Create a LineString from coordinates
        geojson_coords = [[coord[1], coord[0]] for coord in coordinates]

        # Save as GeoJSON
        geojson_data = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    "name": name,
                    "created": datetime.now().isoformat(),
                    "point_count": len(coordinates)
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": geojson_coords
                }
            }]
        }

        # Save to file
        filename = f"data/shorelines/{name}.geojson"
        with open(filename, 'w') as f:
            json.dump(geojson_data, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'Shoreline saved as {name}',
            'filename': filename,
            'point_count': len(coordinates)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_transects', methods=['POST'])
def generate_transects():
    """Generate transects perpendicular to the shoreline"""
    try:
        # Simplified transect generation without geopandas
        data = request.get_json()
        shoreline_name = data.get('shoreline_name')

        return jsonify({
            'success': True,
            'message': 'Transect generation temporarily disabled due to NumPy compatibility issues',
            'note': 'Please fix NumPy version first, then use the full app.py'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_vegetation_edge', methods=['POST'])
def analyze_vegetation_edge():
    """Simplified vegetation analysis"""
    try:
        data = request.get_json()

        return jsonify({
            'success': False,
            'error': 'Advanced VE detection requires NumPy 1.x compatibility. Please downgrade NumPy first.',
            'instructions': 'Run: pip install "numpy<2" or conda install numpy=1.26.4'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/list_shorelines')
def list_shorelines():
    """List all saved shorelines"""
    try:
        shorelines = []
        shoreline_dir = 'data/shorelines'

        if os.path.exists(shoreline_dir):
            for filename in os.listdir(shoreline_dir):
                if filename.endswith('.geojson'):
                    filepath = os.path.join(shoreline_dir, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        feature = data['features'][0]
                        shorelines.append({
                            'filename': filename,
                            'name': feature['properties']['name'],
                            'created': feature['properties']['created'],
                            'point_count': feature['properties']['point_count'],
                            'coordinates': feature['geometry']['coordinates']
                        })

        return jsonify(shorelines)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_numpy_status')
def check_numpy_status():
    """Check NumPy version and compatibility"""
    try:
        import numpy

        # Try importing problematic packages
        issues = []
        try:
            import geopandas
        except ImportError as e:
            issues.append(f"geopandas: {str(e)}")

        try:
            import pandas
        except ImportError as e:
            issues.append(f"pandas: {str(e)}")

        try:
            from sklearn.neighbors import KernelDensity
        except ImportError as e:
            issues.append(f"scikit-learn: {str(e)}")

        return jsonify({
            'numpy_version': numpy.__version__,
            'numpy_compatible': numpy.__version__.startswith('1.'),
            'issues': issues,
            'recommendation': 'Downgrade to NumPy 1.x: pip install "numpy<2"' if not numpy.__version__.startswith('1.') else 'NumPy version is compatible'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("TERRA - Minimal Flask App (NumPy 2.x compatible)")
    print("="*50)
    print("Note: Advanced VE detection is disabled due to NumPy compatibility issues")
    print("To enable full functionality:")
    print("1. Run: pip install 'numpy<2'")
    print("2. Restart the app using: python app.py")
    print("="*50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)