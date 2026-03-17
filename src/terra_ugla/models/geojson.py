"""
GeoJSON model helpers
"""

from datetime import datetime


def create_shoreline_geojson(coordinates, name):
    """Create GeoJSON representation of a shoreline"""
    return {
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
                "coordinates": coordinates
            }
        }]
    }


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


def create_vegetation_edge_feature(transect_id, edge_coords, threshold, detection_method, image_date,
                                   ndvi_samples=None, mean_ndvi=None):
    """Create a vegetation edge point feature"""
    return {
        "type": "Feature",
        "properties": {
            "transect_id": transect_id,
            "ndvi_threshold": threshold,
            "detection_method": detection_method,
            "detection_date": datetime.now().isoformat(),
            "image_date": image_date,
            "ndvi_samples": ndvi_samples,
            "mean_ndvi": mean_ndvi
        },
        "geometry": {
            "type": "Point",
            "coordinates": edge_coords
        }
    }


def create_vegetation_contour_geojson(contours, shoreline_name, threshold, image_date, analysis_method='weighted_peaks'):
    """Create GeoJSON representation of vegetation contours"""
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
