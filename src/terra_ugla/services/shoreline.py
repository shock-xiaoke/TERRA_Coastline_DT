"""
Shoreline management service
"""

import os
import json


def list_shorelines():
    """List all saved shorelines"""
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

    return shorelines


def load_shoreline(filename):
    """Load a specific shoreline"""
    filepath = f"data/shorelines/{filename}"

    if not os.path.exists(filepath):
        raise FileNotFoundError('Shoreline file not found')

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def delete_shoreline(filename):
    """Delete a shoreline file"""
    filepath = f"data/shorelines/{filename}"

    if not os.path.exists(filepath):
        raise FileNotFoundError('File not found')

    os.remove(filepath)
    return True


def save_shoreline(geojson_data, filename):
    """Save shoreline to file"""
    filepath = f"data/shorelines/{filename}"
    with open(filepath, 'w') as f:
        json.dump(geojson_data, f, indent=2)
    return filepath
