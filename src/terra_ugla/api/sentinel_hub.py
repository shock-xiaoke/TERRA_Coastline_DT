"""
Sentinel Hub API client for satellite data retrieval
"""

import os
import requests
from datetime import datetime
from sentinelhub import SentinelHubRequest, DataCollection, MimeType, BBox, CRS, bbox_to_dimensions

from ..config import load_sentinel_hub_config
from ..utils.demo_data import generate_demo_satellite_data


def get_cdse_access_token(client_id, client_secret):
    """Get access token from CDSE using direct HTTP request."""

    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }

    last_error = None
    for _ in range(2):
        try:
            response = requests.post(token_url, data=data, timeout=30)

            if response.status_code != 200:
                snippet = (response.text or "")[:500]
                last_error = f"CDSE token request failed ({response.status_code}): {snippet}"
                continue

            token_data = response.json()
            access_token = token_data.get("access_token")
            if not access_token:
                last_error = "CDSE token response missing access_token"
                continue
            return access_token

        except requests.RequestException as exc:
            last_error = f"CDSE token request error: {exc}"
            continue
        except ValueError as exc:
            last_error = f"CDSE token response parse error: {exc}"
            continue

    raise RuntimeError(last_error or "CDSE token request failed for unknown reason")


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


def request_satellite_data_direct(bbox_coords, start_date, end_date, max_cloud_coverage, image_type, shoreline_name,
                                  client_id, client_secret):
    """
    Request satellite data using direct HTTP API calls to CDSE
    This bypasses the problematic sentinelhub-py library
    """

    print(f"Direct API request to CDSE...")
    print(f"   Area: {bbox_coords}")
    print(f"   Time: {start_date} to {end_date}")

    # Get access token
    access_token = get_cdse_access_token(client_id, client_secret)
    if not access_token:
        raise Exception("Failed to get access token")

    print("Access token obtained")

    # Create evalscript based on image type
    evalscript = get_evalscript(image_type)

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
        print("Sending request to Process API...")
        response = requests.post(
            process_url,
            json=request_payload,
            headers=headers,
            timeout=120
        )

        print(f"   Response status: {response.status_code}")

        if response.status_code == 200:
            print("Satellite data request successful!")

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

            print(f"Error: {error_msg}")
            raise Exception(error_msg)

    except Exception as e:
        error_msg = f"Direct API request failed: {str(e)}"
        print(f"Error: {error_msg}")
        raise Exception(error_msg)


def request_copernicus_data_direct_api(bbox_coords, start_date, end_date, max_cloud_coverage, image_type,
                                       shoreline_name, sentinel_hub_available):
    """
    Request satellite data using direct API calls
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


def request_copernicus_data(bbox_coords, start_date, end_date, max_cloud_coverage, image_type, shoreline_name,
                           sentinel_hub_available, config):
    """
    Request satellite data from Copernicus using Sentinel Hub Process API
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

        # Create request using Process API
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
            config=config,
            data_folder=output_folder
        )

        print(f"Requesting satellite data...")
        print(f"   Area: {bbox_coords}")
        print(f"   Time: {start_date} to {end_date}")
        print(f"   Size: {size}")
        print(f"   Max cloud cover: {max_cloud_coverage}%")

        # Execute request
        data = request.get_data(save_data=True)

        print(f"Successfully downloaded {len(data)} image(s)")

        # Create simple metadata
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


def check_data_availability(bbox_coords, start_date, end_date, config):
    """
    Check what satellite data is available using a simple Process API request
    """
    try:
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
