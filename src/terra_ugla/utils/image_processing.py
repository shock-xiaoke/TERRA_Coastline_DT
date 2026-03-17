"""
Image and raster processing utilities
"""

import numpy as np
import rasterio
from PIL import Image


def extract_ndvi_along_transect(ndvi_image, georef, transect_coords):
    """
    Extract NDVI values along a transect line from raster image
    """
    try:
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


def load_ndvi_data(ndvi_image_path, transects_data=None):
    """
    Load NDVI data from various file formats
    Returns: (ndvi_array, georef_data)
    """
    is_raw_ndvi = ndvi_image_path.endswith('.npy')

    if is_raw_ndvi:
        # Load raw NDVI array directly
        ndvi_array = np.load(ndvi_image_path)
        print(f"   Loaded raw NDVI array: {ndvi_array.shape}")

        # Create georeferencing based on actual coordinates if we have transects
        if transects_data and len(transects_data['features']) > 0:
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
        # For PNG images, extract NDVI-like information
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

    return ndvi_array, georef_data


def convert_pixel_to_geographic(row, col, ndvi_array_shape, georef_data):
    """Convert pixel coordinates to geographic coordinates"""
    if isinstance(georef_data, dict) and 'transform' in georef_data:
        # Use rasterio transform
        lon, lat = rasterio.transform.xy(georef_data['transform'], row, col)
    else:
        # Use simple linear mapping
        height, width = ndvi_array_shape
        west = georef_data.get('west', -180)
        east = georef_data.get('east', 180)
        south = georef_data.get('south', -90)
        north = georef_data.get('north', 90)

        lon = west + (col / width) * (east - west)
        lat = north - (row / height) * (north - south)

    return lon, lat
