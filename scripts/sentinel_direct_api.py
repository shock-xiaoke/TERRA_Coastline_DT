import requests
import json
import os
import base64
from datetime import datetime


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
        demo_data = generate_demo_satellite_data(bbox_coords, image_type, shoreline_name)
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

        demo_data = generate_demo_satellite_data(bbox_coords, image_type, shoreline_name)

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


# Test function to verify the direct API works
def test_direct_api():
    """Test the direct API implementation"""

    # Load credentials
    try:
        with open('config.json', 'r') as f:
            config_data = json.load(f)

        sentinel_config = config_data.get('sentinel_hub', {})
        client_id = sentinel_config['client_id']
        client_secret = sentinel_config['client_secret']

        # Small test area (London)
        bbox_coords = [0.1, 51.5, 0.11, 51.51]

        print("🧪 Testing direct API implementation...")

        result = request_satellite_data_direct(
            bbox_coords=bbox_coords,
            start_date='2024-06-01',
            end_date='2024-06-05',
            max_cloud_coverage=50,
            image_type='true_color',
            shoreline_name='test',
            client_id=client_id,
            client_secret=client_secret
        )

        if result['success']:
            print("🎉 Direct API test successful!")
            return True
        else:
            print("❌ Direct API test failed")
            return False

    except Exception as e:
        print(f"❌ Direct API test error: {e}")
        return False


if __name__ == "__main__":
    # Run test
    test_direct_api()