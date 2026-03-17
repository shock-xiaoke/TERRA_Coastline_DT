"""
Demo data generation for testing and demonstration purposes
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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
