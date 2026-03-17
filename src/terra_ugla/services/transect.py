"""
Transect generation service
"""

import numpy as np
from shapely.geometry import LineString, Point


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
