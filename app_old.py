from flask import Flask, render_template, request, jsonify, send_file
import requests
import base64
import json
import os
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
import pandas as pd
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
# Import scientific packages only when needed to avoid startup issues
# from rasterio.features import rasterize
# from scipy import signal
# from scipy.spatial.distance import cdist
# from skimage import measure
# from sklearn.neighbors import KernelDensity
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

    # CRITICAL: Set ALL the required URLs for CDSE
    config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
    config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    config.sh_auth_base_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect'

    # THEN set credentials
    config.sh_client_id = sentinel_config['client_id']
    config.sh_client_secret = sentinel_config['client_secret']

    # DO NOT set instance_id for CDSE
    # Remove this line: config.instance_id = sentinel_config['instance_id']

    sentinel_hub_available = True
    print("Sentinel Hub configuration loaded successfully")
except Exception as e:
    print(f"Warning: Sentinel Hub configuration error: {e}")
    print("Satellite data functionality will use demo mode")
    sentinel_hub_available = False


def get_cdse_access_token(client_id, client_secret):
    """Get access token from CDSE using direct HTTP request"""

    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }

    try:
        response = requests.post(token_url, data=data, timeout=30)

        if response.status_code == 200:
            token_data = response.json()
            return token_data.get('access_token')
        else:
            print(f"❌ Token request failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"❌ Token request error: {e}")
        return None


def request_satellite_data_direct(bbox_coords, start_date, end_date, max_cloud_coverage, image_type, shoreline_name,
                                  client_id, client_secret):
    """
    Request satellite data using direct HTTP API calls to CDSE
    This bypasses the problematic sentinelhub-py library
    """

    print(f"🚀 Direct API request to CDSE...")
    print(f"   Area: {bbox_coords}")
    print(f"   Time: {start_date} to {end_date}")

    # Get access token
    access_token = get_cdse_access_token(client_id, client_secret)
    if not access_token:
        raise Exception("Failed to get access token")

    print("✅ Access token obtained")

    # Create evalscript based on image type
    if image_type == 'true_color':
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04"],
        output: { bands: 3 }
    };
}

function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02];
}
"""
    elif image_type == 'ndvi':
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B04", "B08"],
        output: { bands: 3 }
    };
}

function evaluatePixel(sample) {
    let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
    return colorBlend(ndvi, 
        [-0.5, 0, 0.2, 0.4, 0.6, 1],
        [[0.05, 0.05, 0.05], [165/255, 0, 38/255], [215/255, 48/255, 39/255], 
         [244/255, 109/255, 67/255], [253/255, 174/255, 97/255], [254/255, 224/255, 139/255]]
    );
}
"""
    else:
        # Default to true color
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04"],
        output: { bands: 3 }
    };
}

function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02];
}
"""

    # Calculate appropriate image size
    lon_diff = bbox_coords[2] - bbox_coords[0]
    lat_diff = bbox_coords[3] - bbox_coords[1]

    # Use conservative size calculation
    pixels_per_degree = 1000  # Conservative estimate
    width = max(10, min(500, int(lon_diff * pixels_per_degree)))
    height = max(10, min(500, int(lat_diff * pixels_per_degree)))

    print(f"   Image size: {width}x{height}")

    # Create the request payload
    request_payload = {
        "input": {
            "bounds": {
                "bbox": bbox_coords,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                }
            },
            "data": [
                {
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date}T00:00:00Z",
                            "to": f"{end_date}T23:59:59Z"
                        },
                        "maxCloudCoverage": max_cloud_coverage
                    },
                    "type": "sentinel-2-l2a"
                }
            ]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/png"
                    }
                }
            ]
        },
        "evalscript": evalscript
    }

    # API endpoint
    process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"

    # Headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'Accept': 'image/png'
    }

    try:
        print("📡 Sending request to Process API...")
        response = requests.post(
            process_url,
            json=request_payload,
            headers=headers,
            timeout=120
        )

        print(f"   Response status: {response.status_code}")

        if response.status_code == 200:
            print("✅ Satellite data request successful!")

            # Save the image
            output_folder = f"data/satellite_images/{shoreline_name}/{image_type}"
            os.makedirs(output_folder, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{shoreline_name}_{image_type}_{timestamp}.png"
            filepath = os.path.join(output_folder, filename)

            with open(filepath, 'wb') as f:
                f.write(response.content)

            print(f"   Saved to: {filepath}")

            # Create metadata
            metadata = [{
                'id': f'direct_api_{timestamp}',
                'date': end_date,
                'cloud_coverage': max_cloud_coverage,
                'platform': 'Sentinel-2',
                'source': 'Direct CDSE API',
                'size': (width, height),
                'filename': filename
            }]

            return {
                'success': True,
                'images': [filepath],
                'metadata': metadata,
                'bbox': bbox_coords,
                'size': (width, height),
                'output_folder': output_folder,
                'image_type': image_type,
                'error': None
            }

        else:
            error_msg = f"API request failed: {response.status_code}"
            if response.content:
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data}"
                except:
                    error_msg += f" - {response.text[:200]}"

            print(f"❌ {error_msg}")
            raise Exception(error_msg)

    except Exception as e:
        error_msg = f"Direct API request failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)


def request_copernicus_data_direct_api(bbox_coords, start_date, end_date, max_cloud_coverage, image_type,
                                       shoreline_name):
    """
    Updated function for your Flask app using direct API calls
    """

    # Check if credentials are available
    if not sentinel_hub_available:
        error_msg = "Sentinel Hub not configured. Please check your credentials in config.json"
        print(f"ERR_MSG: {error_msg}")
        demo_data = generate_demo_satellite_data(bbox_coords, image_type, shoreline_name, start_date, end_date)
        return {
            'success': False,
            'images': demo_data['images'],
            'metadata': demo_data['metadata'],
            'bbox': bbox_coords,
            'size': demo_data['size'],
            'output_folder': demo_data['output_folder'],
            'image_type': image_type,
            'error': error_msg,
            'demo_mode': True
        }

    try:
        # Load credentials
        sentinel_config = load_sentinel_hub_config()
        client_id = sentinel_config['client_id']
        client_secret = sentinel_config['client_secret']

        # Use direct API implementation
        result = request_satellite_data_direct(
            bbox_coords, start_date, end_date, max_cloud_coverage,
            image_type, shoreline_name, client_id, client_secret
        )

        return result

    except Exception as e:
        error_msg = str(e)
        print(f"ERR_MSG: {error_msg}")
        print("Using demo data for demonstration purposes")

        demo_data = generate_demo_satellite_data(bbox_coords, image_type, shoreline_name, start_date, end_date)

        return {
            'success': False,
            'images': demo_data['images'],
            'metadata': demo_data['metadata'],
            'bbox': bbox_coords,
            'size': demo_data['size'],
            'output_folder': demo_data['output_folder'],
            'image_type': image_type,
            'error': error_msg,
            'demo_mode': True
        }

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
        # Coordinates come as [lat, lng] from Leaflet, convert to [lng, lat] for GeoJSON
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
        data = request.get_json()
        shoreline_name = data.get('shoreline_name')
        transect_spacing = data.get('transect_spacing', 100)  # meters
        transect_length = data.get('transect_length', 500)  # meters (total length)
        offshore_ratio = data.get('offshore_ratio', 0.7)  # 70% offshore, 30% onshore

        if not shoreline_name:
            return jsonify({'error': 'Shoreline name is required'}), 400

        # Load shoreline
        shoreline_file = f"data/shorelines/{shoreline_name}.geojson"
        if not os.path.exists(shoreline_file):
            return jsonify({'error': 'Shoreline file not found'}), 404

        with open(shoreline_file, 'r') as f:
            shoreline_data = json.load(f)

        shoreline_coords = shoreline_data['features'][0]['geometry']['coordinates']
        shoreline = LineString(shoreline_coords)

        # Generate transects
        transects = generate_transects_from_shoreline(
            shoreline, transect_spacing, transect_length, offshore_ratio
        )

        # Save transects
        transects_geojson = create_transects_geojson(transects, shoreline_name)
        transects_filename = f"data/transects/{shoreline_name}_transects.geojson"

        with open(transects_filename, 'w') as f:
            json.dump(transects_geojson, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'Generated {len(transects)} transects',
            'transects_file': transects_filename,
            'transects_count': len(transects),
            'transects_data': transects_geojson
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_satellite_data', methods=['POST'])
def get_satellite_data():
    """Retrieve satellite imagery for the area of interest"""
    try:
        data = request.get_json()
        shoreline_name = data.get('shoreline_name')
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2023-12-31')
        max_cloud_coverage = data.get('max_cloud_coverage', 30)
        image_type = data.get('image_type', 'true_color')  # 'true_color', 'ndvi', 'false_color'

        if not shoreline_name:
            return jsonify({'error': 'Shoreline name is required', 'success': False}), 400

        # Load shoreline to determine bbox
        shoreline_file = f"data/shorelines/{shoreline_name}.geojson"
        if not os.path.exists(shoreline_file):
            return jsonify({'error': 'Shoreline file not found', 'success': False}), 404

        with open(shoreline_file, 'r') as f:
            shoreline_data = json.load(f)

        # Calculate bounding box from shoreline with buffer
        coords = shoreline_data['features'][0]['geometry']['coordinates']
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]

        # Add buffer (approximately 1km)
        buffer = 0.01  # degrees
        bbox_coords = [
            min(lons) - buffer,  # min_lon
            min(lats) - buffer,  # min_lat
            max(lons) + buffer,  # max_lon
            max(lats) + buffer  # max_lat
        ]

        # Get satellite data
        satellite_data = request_copernicus_data_direct_api(
            bbox_coords=bbox_coords,
            start_date=start_date,
            end_date=end_date,
            max_cloud_coverage=max_cloud_coverage,
            image_type=image_type,
            shoreline_name=shoreline_name
        )

        # Check if the request was successful
        if satellite_data.get('success', False):
            return jsonify({
                'success': True,
                'message': f'Retrieved {len(satellite_data["images"])} satellite images',
                'data': satellite_data
            })
        else:
            # Handle error case with demo data
            error_msg = satellite_data.get('error', 'Unknown error')
            demo_mode = satellite_data.get('demo_mode', False)
            
            if demo_mode:
                return jsonify({
                    'success': False,
                    'message': f'ERR_MSG: {error_msg}. Using default for demonstration purposes.',
                    'error': error_msg,
                    'data': satellite_data,
                    'demo_mode': True,
                    'warning': 'This is demo data. Please check your Sentinel Hub credentials for real satellite data.'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'ERR_MSG: {error_msg}',
                    'error': error_msg,
                    'data': satellite_data
                }), 500

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


def generate_transects_from_shoreline(shoreline, spacing_meters=100, length_meters=500, offshore_ratio=0.7):
    """
    Generate transects perpendicular to shoreline

    Parameters:
    - shoreline: Shapely LineString
    - spacing_meters: Distance between transects in meters
    - length_meters: Total length of each transect in meters
    - offshore_ratio: Proportion of transect extending offshore (0.7 = 70% offshore, 30% onshore)
    """
    transects = []

    # Convert meters to approximate degrees (rough conversion for mid-latitudes)
    # 1 degree latitude ≈ 111,320 meters
    # 1 degree longitude ≈ 111,320 * cos(latitude) meters
    lat_center = np.mean([coord[1] for coord in shoreline.coords])
    spacing_deg = spacing_meters / 111320
    length_deg = length_meters / 111320

    # Calculate distances along the shoreline
    total_length = shoreline.length
    num_transects = int(total_length / spacing_deg)

    for i in range(num_transects):
        # Position along shoreline (0 to 1)
        position = i / (num_transects - 1) if num_transects > 1 else 0

        # Get point on shoreline
        point_on_shoreline = shoreline.interpolate(position, normalized=True)

        # Calculate perpendicular direction
        # Get a small segment around the point to calculate direction
        segment_start = max(0, position - 0.01)
        segment_end = min(1, position + 0.01)

        start_point = shoreline.interpolate(segment_start, normalized=True)
        end_point = shoreline.interpolate(segment_end, normalized=True)

        # Calculate shoreline direction vector
        dx = end_point.x - start_point.x
        dy = end_point.y - start_point.y

        # Normalize
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 0:
            dx /= length
            dy /= length

        # Perpendicular vector (rotate 90 degrees)
        perp_dx = -dy
        perp_dy = dx

        # Create transect endpoints
        offshore_length = length_deg * offshore_ratio
        onshore_length = length_deg * (1 - offshore_ratio)

        # Onshore point (negative direction)
        onshore_point = Point(
            point_on_shoreline.x - perp_dx * onshore_length,
            point_on_shoreline.y - perp_dy * onshore_length
        )

        # Offshore point (positive direction)
        offshore_point = Point(
            point_on_shoreline.x + perp_dx * offshore_length,
            point_on_shoreline.y + perp_dy * offshore_length
        )

        # Create transect line
        transect = LineString([onshore_point, offshore_point])
        transects.append({
            'id': i,
            'geometry': transect,
            'shoreline_point': point_on_shoreline,
            'properties': {
                'transect_id': i,
                'spacing_meters': spacing_meters,
                'length_meters': length_meters,
                'offshore_ratio': offshore_ratio
            }
        })

    return transects


def create_transects_geojson(transects, shoreline_name):
    """Create GeoJSON representation of transects"""
    features = []

    for transect in transects:
        feature = {
            "type": "Feature",
            "properties": {
                "transect_id": transect['id'],
                "shoreline_name": shoreline_name,
                "created": datetime.now().isoformat(),
                **transect['properties']
            },
            "geometry": {
                "type": "LineString",
                "coordinates": list(transect['geometry'].coords)
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


def get_evalscript(image_type):
    """Get evaluation script for different image types"""

    if image_type == 'true_color':
        return """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04"],
                output: { bands: 3 }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
        """

    elif image_type == 'ndvi':
        return """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B08"],
                output: { bands: 3 }
            };
        }

        function evaluatePixel(sample) {
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
            return colorBlend(ndvi, 
                [-0.5, 0, 0.2, 0.4, 0.6, 1],
                [[0.05, 0.05, 0.05], [165/255, 0, 38/255], [215/255, 48/255, 39/255], 
                 [244/255, 109/255, 67/255], [253/255, 174/255, 97/255], [254/255, 224/255, 139/255]]
            );
        }
        """

    elif image_type == 'false_color':
        return """
        //VERSION=3
        function setup() {
            return {
                input: ["B03", "B04", "B08"],
                output: { bands: 3 }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B08, sample.B04, sample.B03];
        }
        """

    else:
        # Default to true color
        return get_evalscript('true_color')


def request_copernicus_data(bbox_coords, start_date, end_date, max_cloud_coverage, image_type, shoreline_name):
    """
    Request satellite data from Copernicus using Sentinel Hub Process API
    Updated to use only the working Process API (not Catalog API)
    """

    # Check if Sentinel Hub is available
    if not sentinel_hub_available:
        error_msg = "Sentinel Hub not configured. Please check your credentials in config.json"
        print(f"ERR_MSG: {error_msg}")
        print("Using default for demonstration purposes")

        # Generate demo data for demonstration
        demo_data = generate_demo_satellite_data(bbox_coords, image_type, shoreline_name, start_date, end_date)

        return {
            'success': False,
            'images': demo_data['images'],
            'metadata': demo_data['metadata'],
            'bbox': bbox_coords,
            'size': demo_data['size'],
            'output_folder': demo_data['output_folder'],
            'image_type': image_type,
            'error': error_msg,
            'demo_mode': True
        }

    try:
        from sentinelhub import SentinelHubRequest, DataCollection, MimeType, BBox, CRS, bbox_to_dimensions

        # Create BBox
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)

        # Calculate appropriate size (resolution)
        size = bbox_to_dimensions(bbox, resolution=10)  # 10m resolution

        # Limit size to prevent too large images
        max_size = 1000
        if size[0] > max_size:
            ratio = max_size / size[0]
            size = (max_size, int(size[1] * ratio))
        if size[1] > max_size:
            ratio = max_size / size[1]
            size = (int(size[0] * ratio), max_size)

        # Time interval
        time_interval = (start_date, end_date)

        # Get evaluation script
        evalscript = get_evalscript(image_type)

        # Create output directory
        output_folder = f"data/satellite_images/{shoreline_name}/{image_type}"
        os.makedirs(output_folder, exist_ok=True)

        # Create request using Process API (which works!)
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    maxcc=max_cloud_coverage / 100.0  # Convert percentage to ratio
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.PNG)
            ],
            bbox=bbox,
            size=size,
            config=config,  # Use the working config
            data_folder=output_folder
        )

        print(f"📡 Requesting satellite data...")
        print(f"   Area: {bbox_coords}")
        print(f"   Time: {start_date} to {end_date}")
        print(f"   Size: {size}")
        print(f"   Max cloud cover: {max_cloud_coverage}%")

        # Execute request
        data = request.get_data(save_data=True)

        print(f"✅ Successfully downloaded {len(data)} image(s)")

        # Create simple metadata (since Catalog API doesn't work)
        images_info = []
        for i, img_data in enumerate(data):
            images_info.append({
                'id': f'sentinel2_image_{i}',
                'date': end_date,  # Use end date as approximation
                'cloud_coverage': max_cloud_coverage,  # Use max as approximation
                'platform': 'Sentinel-2',
                'source': 'Process API'
            })

        return {
            'success': True,
            'images': data,
            'metadata': images_info,
            'bbox': bbox_coords,
            'size': size,
            'output_folder': output_folder,
            'image_type': image_type,
            'error': None
        }

    except Exception as e:
        error_msg = str(e)
        print(f"ERR_MSG: {error_msg}")
        print("Using default for demonstration purposes")

        # Generate demo data for demonstration
        demo_data = generate_demo_satellite_data(bbox_coords, image_type, shoreline_name, start_date, end_date)

        return {
            'success': False,
            'images': demo_data['images'],
            'metadata': demo_data['metadata'],
            'bbox': bbox_coords,
            'size': demo_data['size'],
            'output_folder': demo_data['output_folder'],
            'image_type': image_type,
            'error': error_msg,
            'demo_mode': True
        }


@app.route('/test_minimal_satellite')
def test_minimal_satellite():
    """Test minimal satellite request to debug 405 error"""
    try:
        from sentinelhub import SentinelHubRequest, DataCollection, MimeType, BBox, CRS

        # Tiny test area
        bbox = BBox([0.1, 51.5, 0.11, 51.51], crs=CRS.WGS84)

        # Minimal evalscript
        evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B04"],
        output: { bands: 1 }
    };
}
function evaluatePixel(sample) {
    return [sample.B04];
}
"""

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=('2024-06-01', '2024-06-05')
                )
            ],
            responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
            bbox=bbox,
            size=(10, 10),  # Tiny size
            config=config
        )

        print("Testing minimal request...")
        data = request.get_data()

        return jsonify({
            'success': True,
            'message': f'Minimal request worked! Downloaded {len(data)} images',
            'size': '10x10 pixels',
            'area': 'London test area'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'message': 'Minimal request failed'
        })


def generate_demo_satellite_data(bbox_coords, image_type, shoreline_name, start_date=None, end_date=None):
    """
    Generate demo satellite data for demonstration purposes when real data is unavailable
    """
    # Create output directory
    output_folder = f"data/satellite_images/{shoreline_name}/{image_type}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate size based on bbox
    bbox_width = bbox_coords[2] - bbox_coords[0]
    bbox_height = bbox_coords[3] - bbox_coords[1]
    
    # Create a simple demo image (placeholder)
    size = (512, 512)  # Standard demo size
    
    # Use actual date range for demo metadata
    if end_date:
        demo_date = end_date
        demo_date_str = end_date.replace('-', '')
    else:
        demo_date = '2023-06-15'
        demo_date_str = '20230615'

    # Create demo metadata with actual date
    demo_metadata = [{
        'id': f'demo_sentinel_2a_{demo_date_str}',
        'date': f'{demo_date}T10:30:00Z',
        'cloud_coverage': 5.0,
        'platform': 'Sentinel-2A',
        'demo': True,
        'start_date': start_date,
        'end_date': end_date
    }]

    # Create demo image file with date in filename
    timestamp = datetime.now().strftime("%Y%m%dT%H%M")
    demo_image_path = os.path.join(output_folder, f'demo_{image_type}_{timestamp}.png')
    try:
        # Create a simple colored image for demo
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a realistic vegetation pattern for NDVI
        if image_type == 'ndvi':
            # Create realistic coastal vegetation pattern
            img = np.zeros((512, 512, 3))

            # Create NDVI-like data (single channel first)
            ndvi_data = np.zeros((512, 512))

            # Create a coastal pattern: shore at bottom, vegetation inland
            for i in range(512):
                for j in range(512):
                    # Distance from shore (bottom of image)
                    dist_from_shore = i / 512.0

                    # Add some randomness
                    noise = np.random.normal(0, 0.1)

                    if dist_from_shore < 0.2:  # Water/wet sand (low NDVI)
                        ndvi_data[i, j] = -0.3 + noise
                    elif dist_from_shore < 0.4:  # Dry sand/rocky shore
                        ndvi_data[i, j] = 0.1 + noise
                    elif dist_from_shore < 0.6:  # Sparse vegetation transition
                        ndvi_data[i, j] = 0.3 + noise * 0.5
                    else:  # Dense vegetation
                        ndvi_data[i, j] = 0.6 + noise * 0.3

            # Clip to valid NDVI range
            ndvi_data = np.clip(ndvi_data, -1, 1)

            # Convert NDVI to RGB visualization
            # Use standard NDVI color scheme: blue for water, red/yellow for land, green for vegetation
            for i in range(512):
                for j in range(512):
                    ndvi_val = ndvi_data[i, j]
                    if ndvi_val < 0:  # Water (blue)
                        img[i, j] = [0, 0, 0.8]
                    elif ndvi_val < 0.2:  # Bare soil (brown/red)
                        img[i, j] = [0.6, 0.3, 0.1]
                    elif ndvi_val < 0.4:  # Light vegetation (yellow-green)
                        img[i, j] = [0.8, 0.8, 0.2]
                    else:  # Dense vegetation (green)
                        img[i, j] = [0.2, 0.8, 0.2]
        elif image_type == 'false_color':
            # Blue to red gradient for false color
            img = np.random.rand(512, 512, 3)
            img[:, :, 0] = np.linspace(0, 1, 512).reshape(1, -1)  # Red channel
            img[:, :, 1] = 0  # Green channel
            img[:, :, 2] = np.linspace(1, 0, 512).reshape(1, -1)  # Blue channel
        else:
            # Natural color gradient for true color
            img = np.random.rand(512, 512, 3)
            img[:, :, 0] = np.linspace(0.2, 0.8, 512).reshape(1, -1)  # Red channel
            img[:, :, 1] = np.linspace(0.3, 0.9, 512).reshape(1, -1)  # Green channel
            img[:, :, 2] = np.linspace(0.1, 0.7, 512).reshape(1, -1)  # Blue channel
        
        plt.imsave(demo_image_path, img)

        # Also save raw NDVI data for realistic analysis
        if image_type == 'ndvi' and 'ndvi_data' in locals():
            ndvi_raw_path = os.path.join(output_folder, 'demo_ndvi_raw.npy')
            np.save(ndvi_raw_path, ndvi_data)
            print(f"   Saved raw NDVI data to: {ndvi_raw_path}")

        demo_images = [demo_image_path]
        
    except Exception as e:
        print(f"Could not create demo image: {e}")
        demo_images = []
    
    return {
        'images': demo_images,
        'metadata': demo_metadata,
        'size': size,
        'output_folder': output_folder
    }


@app.route('/analyze_vegetation_edge', methods=['POST'])
def analyze_vegetation_edge():
    """Analyze vegetation edge along transects using satellite imagery"""
    try:
        data = request.get_json()
        shoreline_name = data.get('shoreline_name')
        image_date = data.get('image_date')  # Specific date or 'latest'
        analysis_method = data.get('analysis_method', 'weighted_peaks')
        ndvi_threshold = data.get('ndvi_threshold', 0.3)  # Only used for manual method

        if not shoreline_name:
            return jsonify({'error': 'Shoreline name is required'}), 400

        # Load transects
        transects_file = f"data/transects/{shoreline_name}_transects.geojson"
        if not os.path.exists(transects_file):
            return jsonify({'error': 'Transects file not found. Generate transects first.'}), 404

        # Load NDVI imagery
        ndvi_folder = f"data/satellite_images/{shoreline_name}/ndvi"
        if not os.path.exists(ndvi_folder):
            return jsonify({'error': 'NDVI imagery not found. Request satellite data first.'}), 404

        # Perform vegetation edge detection
        vegetation_edges = detect_vegetation_edges_along_transects(
            transects_file, ndvi_folder, ndvi_threshold, image_date, analysis_method
        )

        # Save results
        results_file = f"data/exports/{shoreline_name}_vegetation_edges.geojson"
        with open(results_file, 'w') as f:
            json.dump(vegetation_edges, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'Analyzed vegetation edges along {len(vegetation_edges["features"])} transects',
            'results_file': results_file,
            'vegetation_edges': vegetation_edges
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def find_weighted_peaks_threshold(veg_values, nonveg_values):
    """
    Core vegetation edge detection using weighted peaks algorithm
    Based on the established VE detection method
    """
    try:
        # Import scipy and sklearn only when needed
        from scipy import signal
        from sklearn.neighbors import KernelDensity
        # Combine all values for probability distribution
        all_values = np.concatenate([veg_values, nonveg_values]).reshape(-1, 1)

        if len(all_values) < 10:  # Need sufficient data points
            return np.mean([np.mean(veg_values), np.mean(nonveg_values)])

        # Create kernel density estimator with Gaussian kernel
        model = KernelDensity(bandwidth=0.01, kernel='gaussian')
        model.fit(all_values)

        # Generate test points for evaluation
        x_vals = np.linspace(all_values.min(), all_values.max(), 1000).reshape(-1, 1)
        probabilities = model.score_samples(x_vals)

        # Find peaks in the probability distribution
        peaks_idx, properties = signal.find_peaks(probabilities, prominence=0.5)

        if len(peaks_idx) >= 2:
            # Get the two most prominent peaks
            prominences = properties['prominences']
            top_peaks_idx = peaks_idx[np.argsort(prominences)[-2:]]
            peak_values = x_vals[top_peaks_idx].flatten()

            # Sort peaks (vegetation typically has higher NDVI values)
            peak_values = np.sort(peak_values)

            # Calculate threshold using 0.2:0.8 weighting between peaks
            # Lower weight for lower peak (non-vegetation), higher for upper peak (vegetation)
            threshold = 0.2 * peak_values[0] + 0.8 * peak_values[1]
        else:
            # Fallback: use midpoint between mean values
            threshold = (np.mean(nonveg_values) + np.mean(veg_values)) / 2

        return float(threshold)

    except Exception as e:
        print(f"Error in weighted peaks calculation: {e}")
        # Fallback to simple threshold
        return float((np.mean(nonveg_values) + np.mean(veg_values)) / 2)


def extract_ndvi_along_transect(ndvi_image, georef, transect_coords):
    """
    Extract NDVI values along a transect line from raster image
    """
    try:
        from rasterio.transform import from_bounds

        # Convert transect coordinates to pixel coordinates
        start_lon, start_lat = transect_coords[0]
        end_lon, end_lat = transect_coords[1]

        print(f"   Transect coords: [{start_lon:.6f}, {start_lat:.6f}] to [{end_lon:.6f}, {end_lat:.6f}]")

        # Create line samples (interpolate between start and end points)
        num_samples = 100  # Sample 100 points along the transect

        lons = np.linspace(start_lon, end_lon, num_samples)
        lats = np.linspace(start_lat, end_lat, num_samples)

        # Convert geographic coordinates to pixel coordinates
        height, width = ndvi_image.shape

        if isinstance(georef, dict) and 'transform' in georef:
            transform = georef['transform']
            print(f"   Using rasterio transform: {transform}")
        else:
            # For demo images, use actual transect bounds with buffer
            buffer = max(abs(end_lon - start_lon), abs(end_lat - start_lat)) * 0.5
            west = min(start_lon, end_lon) - buffer
            east = max(start_lon, end_lon) + buffer
            south = min(start_lat, end_lat) - buffer
            north = max(start_lat, end_lat) + buffer

            print(f"   Creating transform for bounds: W={west:.6f}, S={south:.6f}, E={east:.6f}, N={north:.6f}")
            print(f"   Image size: {width}x{height}")

            transform = rasterio.transform.from_bounds(west, south, east, north, width, height)

        # Sample NDVI values at each point
        ndvi_values = []
        distances = []
        valid_pixels = 0

        for i, (lon, lat) in enumerate(zip(lons, lats)):
            try:
                # Convert to pixel coordinates
                col, row = rasterio.transform.rowcol(transform, lon, lat)

                # Check bounds
                if 0 <= row < ndvi_image.shape[0] and 0 <= col < ndvi_image.shape[1]:
                    ndvi_val = ndvi_image[row, col]
                    if not np.isnan(ndvi_val) and not np.isinf(ndvi_val):
                        ndvi_values.append(ndvi_val)
                        # Calculate distance from transect start
                        dist = np.sqrt((lon - start_lon)**2 + (lat - start_lat)**2)
                        distances.append(dist)
                        valid_pixels += 1

            except Exception as e:
                print(f"   Error processing point {i}: {e}")
                continue

        print(f"   Extracted {len(ndvi_values)} valid NDVI values from {valid_pixels} pixels")
        if len(ndvi_values) > 0:
            print(f"   NDVI range: {min(ndvi_values):.3f} to {max(ndvi_values):.3f}")

        return np.array(ndvi_values), np.array(distances)

    except Exception as e:
        print(f"   Error extracting NDVI along transect: {e}")
        return np.array([]), np.array([])


def find_vegetation_edge_point(ndvi_values, distances, threshold, transect_coords):
    """
    Find the vegetation edge point along a transect using the threshold
    """
    if len(ndvi_values) == 0:
        # Fallback to midpoint if no valid data
        mid_point = [
            (transect_coords[0][0] + transect_coords[1][0]) / 2,
            (transect_coords[0][1] + transect_coords[1][1]) / 2
        ]
        return mid_point

    # Find first point where NDVI exceeds threshold (vegetation edge)
    edge_indices = np.where(ndvi_values > threshold)[0]

    if len(edge_indices) > 0:
        edge_idx = edge_indices[0]  # First point above threshold
        edge_distance = distances[edge_idx]

        # Interpolate coordinates at the edge distance
        start_lon, start_lat = transect_coords[0]
        end_lon, end_lat = transect_coords[1]

        total_distance = np.sqrt((end_lon - start_lon)**2 + (end_lat - start_lat)**2)

        if total_distance > 0:
            ratio = edge_distance / total_distance
            edge_lon = start_lon + ratio * (end_lon - start_lon)
            edge_lat = start_lat + ratio * (end_lat - start_lat)
            return [edge_lon, edge_lat]

    # Fallback to midpoint if no edge found
    mid_point = [
        (transect_coords[0][0] + transect_coords[1][0]) / 2,
        (transect_coords[0][1] + transect_coords[1][1]) / 2
    ]
    return mid_point


def detect_vegetation_edges_along_transects(transects_file, ndvi_folder, threshold, image_date, analysis_method='weighted_peaks'):
    """
    Detect vegetation edges along transects using NDVI imagery and weighted peaks algorithm
    """
    print(f"🌱 Starting vegetation edge detection...")
    print(f"   Transects: {transects_file}")
    print(f"   NDVI folder: {ndvi_folder}")

    with open(transects_file, 'r') as f:
        transects_data = json.load(f)

    vegetation_edges = {
        "type": "FeatureCollection",
        "features": []
    }

    # Find the most recent NDVI image
    ndvi_image_path = None
    georef_data = None

    # Look for NDVI image files (prefer raw NDVI data if available)
    import glob

    # First check for raw NDVI numpy files
    raw_ndvi_files = glob.glob(os.path.join(ndvi_folder, '*_raw.npy'))
    image_files = []
    for ext in ['*.png', '*.tiff', '*.tif', '*.jpg']:
        image_files.extend(glob.glob(os.path.join(ndvi_folder, ext)))

    if not raw_ndvi_files and not image_files:
        print("⚠️  No NDVI image files found. Using fallback method.")
        # Fallback: use simple threshold-based detection
        return detect_vegetation_edges_fallback(transects_data, threshold, image_date)

    # Prefer raw NDVI data for better accuracy
    if raw_ndvi_files:
        ndvi_image_path = max(raw_ndvi_files, key=os.path.getmtime)
        print(f"   Using raw NDVI data: {ndvi_image_path}")
        is_raw_ndvi = True
    else:
        ndvi_image_path = max(image_files, key=os.path.getmtime)
        print(f"   Using NDVI image: {ndvi_image_path}")
        is_raw_ndvi = False

    try:
        # Load NDVI data
        if is_raw_ndvi and ndvi_image_path.endswith('.npy'):
            # Load raw NDVI array directly
            ndvi_array = np.load(ndvi_image_path)
            print(f"   Loaded raw NDVI array: {ndvi_array.shape}")

            # Create georeferencing based on actual coordinates if we have transects
            if len(transects_data['features']) > 0:
                # Get coordinate bounds from all transects
                all_coords = []
                for feature in transects_data['features']:
                    all_coords.extend(feature['geometry']['coordinates'])

                lons = [coord[0] for coord in all_coords]
                lats = [coord[1] for coord in all_coords]

                buffer = max(max(lons) - min(lons), max(lats) - min(lats)) * 0.1
                georef_data = {
                    'west': min(lons) - buffer,
                    'east': max(lons) + buffer,
                    'south': min(lats) - buffer,
                    'north': max(lats) + buffer,
                    'width': ndvi_array.shape[1],
                    'height': ndvi_array.shape[0]
                }
                print(f"   Georef bounds: W={georef_data['west']:.6f}, E={georef_data['east']:.6f}, S={georef_data['south']:.6f}, N={georef_data['north']:.6f}")
            else:
                # Fallback georeferencing
                georef_data = {
                    'west': -1, 'east': 1, 'south': -1, 'north': 1,
                    'width': ndvi_array.shape[1], 'height': ndvi_array.shape[0]
                }

        elif ndvi_image_path.endswith('.png'):
            # For PNG images, we need to load as RGB and extract the NDVI-like information
            from PIL import Image
            img = Image.open(ndvi_image_path)
            ndvi_array = np.array(img)

            # Convert to single channel (use green channel as proxy for vegetation)
            if len(ndvi_array.shape) == 3:
                ndvi_array = ndvi_array[:, :, 1]  # Green channel

            # Normalize to NDVI-like range [-1, 1]
            ndvi_array = (ndvi_array.astype(float) / 255.0) * 2 - 1

            # Create approximate georeferencing
            georef_data = {
                'west': -180, 'east': 180, 'south': -90, 'north': 90,
                'width': ndvi_array.shape[1], 'height': ndvi_array.shape[0]
            }

        else:
            # For GeoTIFF files
            with rasterio.open(ndvi_image_path) as dataset:
                ndvi_array = dataset.read(1)  # Read first band
                georef_data = {
                    'transform': dataset.transform,
                    'bounds': dataset.bounds,
                    'width': dataset.width,
                    'height': dataset.height
                }

        print(f"   NDVI array shape: {ndvi_array.shape}")
        print(f"   NDVI value range: {ndvi_array.min():.3f} to {ndvi_array.max():.3f}")

        # Collect all NDVI values from all transects to compute global threshold
        all_veg_values = []
        all_nonveg_values = []

        # First pass: collect values for threshold calculation
        for feature in transects_data['features']:
            transect_coords = feature['geometry']['coordinates']
            ndvi_vals, _ = extract_ndvi_along_transect(ndvi_array, georef_data, transect_coords)

            if len(ndvi_vals) > 0:
                # Simple heuristic: higher NDVI values are vegetation, lower are non-vegetation
                median_ndvi = np.median(ndvi_vals)
                veg_mask = ndvi_vals > median_ndvi
                nonveg_mask = ndvi_vals <= median_ndvi

                all_veg_values.extend(ndvi_vals[veg_mask])
                all_nonveg_values.extend(ndvi_vals[nonveg_mask])

        # Calculate threshold based on analysis method
        if analysis_method == 'weighted_peaks':
            if len(all_veg_values) > 10 and len(all_nonveg_values) > 10:
                optimal_threshold = find_weighted_peaks_threshold(
                    np.array(all_veg_values),
                    np.array(all_nonveg_values)
                )
                print(f"   Calculated optimal threshold (weighted peaks): {optimal_threshold:.3f}")
            else:
                optimal_threshold = 0.3  # Default fallback
                print(f"   Using default threshold (insufficient data): {optimal_threshold:.3f}")
        else:  # manual_threshold
            optimal_threshold = threshold
            print(f"   Using manual threshold: {optimal_threshold:.3f}")

        # Second pass: detect vegetation edges using the calculated threshold
        for feature in transects_data['features']:
            transect_coords = feature['geometry']['coordinates']
            transect_id = feature['properties']['transect_id']

            # Extract NDVI values along the transect
            ndvi_vals, distances = extract_ndvi_along_transect(ndvi_array, georef_data, transect_coords)

            # Find vegetation edge point
            edge_coords = find_vegetation_edge_point(
                ndvi_vals, distances, optimal_threshold, transect_coords
            )

            edge_feature = {
                "type": "Feature",
                "properties": {
                    "transect_id": transect_id,
                    "ndvi_threshold": optimal_threshold,
                    "detection_method": analysis_method,
                    "detection_date": datetime.now().isoformat(),
                    "image_date": image_date,
                    "ndvi_samples": len(ndvi_vals),
                    "mean_ndvi": float(np.mean(ndvi_vals)) if len(ndvi_vals) > 0 else None
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": edge_coords
                }
            }

            vegetation_edges['features'].append(edge_feature)

        print(f"✅ Detected {len(vegetation_edges['features'])} vegetation edges")
        return vegetation_edges

    except Exception as e:
        print(f"❌ Error in vegetation edge detection: {e}")
        print("   Using fallback method...")
        return detect_vegetation_edges_fallback(transects_data, threshold, image_date)


def detect_vegetation_edges_fallback(transects_data, threshold, image_date):
    """
    Fallback vegetation edge detection when image processing fails
    """
    vegetation_edges = {
        "type": "FeatureCollection",
        "features": []
    }

    for feature in transects_data['features']:
        transect_coords = feature['geometry']['coordinates']
        transect_id = feature['properties']['transect_id']

        # Use midpoint as fallback
        mid_point = [
            (transect_coords[0][0] + transect_coords[1][0]) / 2,
            (transect_coords[0][1] + transect_coords[1][1]) / 2
        ]

        edge_feature = {
            "type": "Feature",
            "properties": {
                "transect_id": transect_id,
                "ndvi_threshold": threshold,
                "detection_method": "fallback_midpoint",
                "detection_date": datetime.now().isoformat(),
                "image_date": image_date,
                "warning": "Used fallback method - no NDVI data available"
            },
            "geometry": {
                "type": "Point",
                "coordinates": mid_point
            }
        }

        vegetation_edges['features'].append(edge_feature)

    return vegetation_edges


def extract_vegetation_contours_marching_squares(ndvi_array, threshold, georef_data):
    """
    Extract vegetation contours using Marching Squares algorithm
    Based on the established VE detection method
    """
    try:
        # Import skimage only when needed
        from skimage import measure
        print(f"   Extracting contours using Marching Squares (threshold: {threshold:.3f})...")

        # Apply threshold to create binary mask
        vegetation_mask = ndvi_array > threshold

        # Extract contours using scikit-image's marching squares implementation
        contours = measure.find_contours(ndvi_array, threshold)

        print(f"   Found {len(contours)} contour segments")

        # Convert pixel coordinates to geographic coordinates
        geographic_contours = []

        for contour in contours:
            if len(contour) < 3:  # Skip very short contours
                continue

            geographic_coords = []

            for point in contour:
                row, col = point

                # Convert pixel coordinates to geographic coordinates
                if isinstance(georef_data, dict) and 'transform' in georef_data:
                    # Use rasterio transform
                    lon, lat = rasterio.transform.xy(georef_data['transform'], row, col)
                else:
                    # Use simple linear mapping
                    height, width = ndvi_array.shape
                    west = georef_data.get('west', -180)
                    east = georef_data.get('east', 180)
                    south = georef_data.get('south', -90)
                    north = georef_data.get('north', 90)

                    lon = west + (col / width) * (east - west)
                    lat = north - (row / height) * (north - south)

                geographic_coords.append([lon, lat])

            if len(geographic_coords) > 2:
                geographic_contours.append(geographic_coords)

        return geographic_contours

    except Exception as e:
        print(f"❌ Error in contour extraction: {e}")
        return []


def create_vegetation_contour_geojson(contours, shoreline_name, threshold, image_date, analysis_method='weighted_peaks'):
    """
    Create GeoJSON representation of vegetation contours
    """
    features = []

    for i, contour in enumerate(contours):
        feature = {
            "type": "Feature",
            "properties": {
                "contour_id": i,
                "shoreline_name": shoreline_name,
                "ndvi_threshold": threshold,
                "detection_method": analysis_method,
                "contour_algorithm": "marching_squares",
                "detection_date": datetime.now().isoformat(),
                "image_date": image_date,
                "point_count": len(contour)
            },
            "geometry": {
                "type": "LineString",
                "coordinates": contour
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


@app.route('/extract_vegetation_contours', methods=['POST'])
def extract_vegetation_contours():
    """
    Extract continuous vegetation contours using Marching Squares algorithm
    This complements the transect-based approach
    """
    try:
        data = request.get_json()
        shoreline_name = data.get('shoreline_name')
        image_date = data.get('image_date', 'latest')
        analysis_method = data.get('analysis_method', 'weighted_peaks')
        ndvi_threshold = data.get('ndvi_threshold', 0.3)  # Only used for manual method

        if not shoreline_name:
            return jsonify({'error': 'Shoreline name is required'}), 400

        # Load NDVI imagery
        ndvi_folder = f"data/satellite_images/{shoreline_name}/ndvi"
        if not os.path.exists(ndvi_folder):
            return jsonify({'error': 'NDVI imagery not found. Request satellite data first.'}), 404

        # Find NDVI image files (prefer raw NDVI data)
        import glob

        # Check for raw NDVI numpy files first
        raw_ndvi_files = glob.glob(os.path.join(ndvi_folder, '*_raw.npy'))
        image_files = []
        for ext in ['*.png', '*.tiff', '*.tif', '*.jpg']:
            image_files.extend(glob.glob(os.path.join(ndvi_folder, ext)))

        if not raw_ndvi_files and not image_files:
            return jsonify({'error': 'No NDVI image files found.'}), 404

        # Prefer raw NDVI data for better accuracy
        if raw_ndvi_files:
            ndvi_image_path = max(raw_ndvi_files, key=os.path.getmtime)
            print(f"Using raw NDVI data for contours: {ndvi_image_path}")
            is_raw_ndvi = True
        else:
            ndvi_image_path = max(image_files, key=os.path.getmtime)
            print(f"Using NDVI image for contours: {ndvi_image_path}")
            is_raw_ndvi = False

        # Load and process NDVI data
        if is_raw_ndvi and ndvi_image_path.endswith('.npy'):
            # Load raw NDVI array directly
            ndvi_array = np.load(ndvi_image_path)
            print(f"Loaded raw NDVI array for contours: {ndvi_array.shape}")

            # Use a reasonable coordinate system for demo
            georef_data = {
                'west': -1, 'east': 1, 'south': -1, 'north': 1,
                'width': ndvi_array.shape[1], 'height': ndvi_array.shape[0]
            }

        elif ndvi_image_path.endswith('.png'):
            from PIL import Image
            img = Image.open(ndvi_image_path)
            ndvi_array = np.array(img)

            if len(ndvi_array.shape) == 3:
                ndvi_array = ndvi_array[:, :, 1]  # Green channel

            ndvi_array = (ndvi_array.astype(float) / 255.0) * 2 - 1

            georef_data = {
                'west': -180, 'east': 180, 'south': -90, 'north': 90,
                'width': ndvi_array.shape[1], 'height': ndvi_array.shape[0]
            }
        else:
            with rasterio.open(ndvi_image_path) as dataset:
                ndvi_array = dataset.read(1)
                georef_data = {
                    'transform': dataset.transform,
                    'bounds': dataset.bounds,
                    'width': dataset.width,
                    'height': dataset.height
                }

        print(f"NDVI array stats for contours: min={ndvi_array.min():.3f}, max={ndvi_array.max():.3f}")

        # Calculate threshold based on analysis method
        if analysis_method == 'weighted_peaks':
            # Sample NDVI values for threshold calculation
            sample_indices = np.random.choice(ndvi_array.size, min(10000, ndvi_array.size), replace=False)
            sample_values = ndvi_array.flat[sample_indices]
            valid_values = sample_values[~np.isnan(sample_values) & ~np.isinf(sample_values)]

            if len(valid_values) > 100:
                median_val = np.median(valid_values)
                veg_values = valid_values[valid_values > median_val]
                nonveg_values = valid_values[valid_values <= median_val]

                if len(veg_values) > 50 and len(nonveg_values) > 50:
                    optimal_threshold = find_weighted_peaks_threshold(veg_values, nonveg_values)
                    print(f"   Calculated optimal threshold (weighted peaks): {optimal_threshold:.3f}")
                else:
                    optimal_threshold = 0.3
                    print(f"   Using default threshold (insufficient data): {optimal_threshold:.3f}")
            else:
                optimal_threshold = 0.3
                print(f"   Using default threshold (no valid data): {optimal_threshold:.3f}")
        else:  # manual_threshold
            optimal_threshold = ndvi_threshold
            print(f"   Using manual threshold: {optimal_threshold:.3f}")

        # Extract contours
        contours = extract_vegetation_contours_marching_squares(
            ndvi_array, optimal_threshold, georef_data
        )

        print(f"Extracted {len(contours)} contour segments")

        # Create GeoJSON
        contours_geojson = create_vegetation_contour_geojson(
            contours, shoreline_name, optimal_threshold, image_date, analysis_method
        )

        print(f"Created GeoJSON with {len(contours_geojson['features'])} features")

        # Save results
        results_file = f"data/exports/{shoreline_name}_vegetation_contours.geojson"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(contours_geojson, f, indent=2)

        return jsonify({
            'success': True,
            'message': f'Extracted {len(contours)} vegetation contour segments',
            'results_file': results_file,
            'contours': contours_geojson
        })

    except Exception as e:
        print(f"Error in extract_vegetation_contours: {e}")
        import traceback
        traceback.print_exc()
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


@app.route('/load_shoreline/<filename>')
def load_shoreline(filename):
    """Load a specific shoreline"""
    try:
        filepath = f"data/shorelines/{filename}"

        if not os.path.exists(filepath):
            return jsonify({'error': 'Shoreline file not found'}), 404

        with open(filepath, 'r') as f:
            data = json.load(f)

        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete_shoreline/<filename>', methods=['DELETE'])
def delete_shoreline(filename):
    """Delete a shoreline file"""
    try:
        filepath = f"data/shorelines/{filename}"

        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': f'Deleted {filename}'})
        else:
            return jsonify({'error': 'File not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/check_sentinel_hub_status')
def check_sentinel_hub_status():
    """Check if Sentinel Hub is properly configured"""
    try:
        if sentinel_hub_available:
            return jsonify({
                'status': 'configured',
                'message': 'Sentinel Hub is properly configured',
                'available': True
            })
        else:
            return jsonify({
                'status': 'not_configured',
                'message': 'Sentinel Hub is not configured. Check config.json for credentials.',
                'available': False,
                'demo_mode': True
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error checking Sentinel Hub status: {str(e)}',
            'available': False
        }), 500


# (optional)
def check_data_availability(bbox_coords, start_date, end_date):
    """
    Check what satellite data is available using a simple Process API request
    This replaces the problematic Catalog API
    """
    try:
        from sentinelhub import SentinelHubRequest, DataCollection, MimeType, BBox, CRS

        # Create a very small request just to check availability
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)

        # Minimal evalscript just to test
        test_evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02"],
                output: { bands: 1 }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B02];
        }
        """

        request = SentinelHubRequest(
            evalscript=test_evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(start_date, end_date)
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.PNG)
            ],
            bbox=bbox,
            size=(10, 10),  # Very small size
            config=config
        )

        # Check if request can be created (indicates data availability)
        download_list = request.download_list
        if download_list:
            return {'available': True, 'message': 'Satellite data available for this area and time period'}
        else:
            return {'available': False, 'message': 'No satellite data available for this area and time period'}

    except Exception as e:
        return {'available': False, 'message': f'Error checking availability: {str(e)}'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)