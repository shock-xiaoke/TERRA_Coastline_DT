"""
Vegetation edge detection service
"""

import os
import json
import glob
import numpy as np
from datetime import datetime

from ..utils.image_processing import extract_ndvi_along_transect, load_ndvi_data, convert_pixel_to_geographic
from ..models.geojson import create_vegetation_edge_feature


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
    print(f"Starting vegetation edge detection...")
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
    # First check for raw NDVI numpy files
    raw_ndvi_files = glob.glob(os.path.join(ndvi_folder, '*_raw.npy'))
    image_files = []
    for ext in ['*.png', '*.tiff', '*.tif', '*.jpg']:
        image_files.extend(glob.glob(os.path.join(ndvi_folder, ext)))

    if not raw_ndvi_files and not image_files:
        print("No NDVI image files found. Using fallback method.")
        return detect_vegetation_edges_fallback(transects_data, threshold, image_date)

    # Prefer raw NDVI data for better accuracy
    if raw_ndvi_files:
        ndvi_image_path = max(raw_ndvi_files, key=os.path.getmtime)
        print(f"   Using raw NDVI data: {ndvi_image_path}")
    else:
        ndvi_image_path = max(image_files, key=os.path.getmtime)
        print(f"   Using NDVI image: {ndvi_image_path}")

    try:
        # Load NDVI data
        ndvi_array, georef_data = load_ndvi_data(ndvi_image_path, transects_data)

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

            edge_feature = create_vegetation_edge_feature(
                transect_id=transect_id,
                edge_coords=edge_coords,
                threshold=optimal_threshold,
                detection_method=analysis_method,
                image_date=image_date,
                ndvi_samples=len(ndvi_vals),
                mean_ndvi=float(np.mean(ndvi_vals)) if len(ndvi_vals) > 0 else None
            )

            vegetation_edges['features'].append(edge_feature)

        print(f"Detected {len(vegetation_edges['features'])} vegetation edges")
        return vegetation_edges

    except Exception as e:
        print(f"Error in vegetation edge detection: {e}")
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
                lon, lat = convert_pixel_to_geographic(row, col, ndvi_array.shape, georef_data)
                geographic_coords.append([lon, lat])

            if len(geographic_coords) > 2:
                geographic_contours.append(geographic_coords)

        return geographic_contours

    except Exception as e:
        print(f"Error in contour extraction: {e}")
        return []


def extract_vegetation_contours(shoreline_name, ndvi_folder, image_date, analysis_method, ndvi_threshold):
    """
    Extract continuous vegetation contours using Marching Squares algorithm
    """
    # Find NDVI image files (prefer raw NDVI data)
    raw_ndvi_files = glob.glob(os.path.join(ndvi_folder, '*_raw.npy'))
    image_files = []
    for ext in ['*.png', '*.tiff', '*.tif', '*.jpg']:
        image_files.extend(glob.glob(os.path.join(ndvi_folder, ext)))

    if not raw_ndvi_files and not image_files:
        raise FileNotFoundError('No NDVI image files found.')

    # Prefer raw NDVI data for better accuracy
    if raw_ndvi_files:
        ndvi_image_path = max(raw_ndvi_files, key=os.path.getmtime)
        print(f"Using raw NDVI data for contours: {ndvi_image_path}")
    else:
        ndvi_image_path = max(image_files, key=os.path.getmtime)
        print(f"Using NDVI image for contours: {ndvi_image_path}")

    # Load and process NDVI data
    ndvi_array, georef_data = load_ndvi_data(ndvi_image_path)

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

    return contours, optimal_threshold
