"""AOI management service for polygon-first workflows."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pyproj import Transformer
from shapely.geometry import Polygon, mapping, shape
from shapely.validation import explain_validity

AOI_DIR = Path("data") / "aoi"
FIXED_AOI_ID = "fixed_model_aoi"
FIXED_AOI_NAME = "st_andrews_model_aoi"
FIXED_AOI_POLYGON_LATLNG = [
    [56.3495, -2.8875],
    [56.3495, -2.8060],
    [56.3825, -2.8060],
    [56.3825, -2.8875],
]


def _polygon_to_latlng(polygon: Polygon) -> list[list[float]]:
    coords = list(polygon.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[float(lat), float(lon)] for lon, lat in coords]


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "aoi"


def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def _validate_polygon_latlng(polygon_latlng: list[list[float]], close_polygon: bool) -> Polygon:
    if not isinstance(polygon_latlng, list) or len(polygon_latlng) < 3:
        raise ValueError("AOI polygon requires at least 3 vertices")

    coords_lonlat: list[tuple[float, float]] = []
    for idx, vertex in enumerate(polygon_latlng):
        if not isinstance(vertex, list) or len(vertex) != 2:
            raise ValueError(f"Invalid vertex at index {idx}")
        lat, lon = float(vertex[0]), float(vertex[1])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"Invalid latitude/longitude at index {idx}")
        coords_lonlat.append((lon, lat))

    if close_polygon and coords_lonlat[0] != coords_lonlat[-1]:
        coords_lonlat.append(coords_lonlat[0])

    polygon = Polygon(coords_lonlat)
    if not polygon.is_valid:
        raise ValueError(f"Invalid AOI polygon: {explain_validity(polygon)}")
    if polygon.area <= 0:
        raise ValueError("AOI polygon area must be greater than zero")

    # Validate area in meters in local UTM CRS.
    centroid = polygon.centroid
    utm_epsg = _utm_epsg_for_lonlat(centroid.x, centroid.y)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    projected = Polygon([transformer.transform(x, y) for x, y in polygon.exterior.coords])
    if projected.area < 1_000:
        raise ValueError("AOI is too small. Minimum area is 1,000 square meters")

    return polygon


def create_aoi(name: str, polygon_latlng: list[list[float]], close_polygon: bool = True) -> dict[str, Any]:
    """Validate and persist an AOI polygon."""
    AOI_DIR.mkdir(parents=True, exist_ok=True)

    polygon = _validate_polygon_latlng(polygon_latlng, close_polygon)
    centroid = polygon.centroid
    utm_epsg = _utm_epsg_for_lonlat(centroid.x, centroid.y)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    aoi_id = f"{_slugify(name)}_{timestamp}"

    feature = {
        "type": "Feature",
        "properties": {
            "aoi_id": aoi_id,
            "name": name,
            "created": datetime.utcnow().isoformat() + "Z",
            "utm_epsg": utm_epsg,
            "bbox_wgs84": [min_lon, min_lat, max_lon, max_lat],
            "vertex_count": len(polygon.exterior.coords) - 1,
        },
        "geometry": mapping(polygon),
    }

    collection = {"type": "FeatureCollection", "features": [feature]}
    path = AOI_DIR / f"{aoi_id}.geojson"
    with path.open("w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)

    return {
        "aoi_id": aoi_id,
        "name": name,
        "bbox_wgs84": [min_lon, min_lat, max_lon, max_lat],
        "utm_epsg": utm_epsg,
        "filepath": str(path),
    }


def create_or_get_fixed_aoi(force_recreate: bool = False) -> dict[str, Any]:
    """Create or return the model-fixed AOI used by segmentation/prediction v1."""
    AOI_DIR.mkdir(parents=True, exist_ok=True)
    path = AOI_DIR / f"{FIXED_AOI_ID}.geojson"

    if path.exists() and not force_recreate:
        loaded = load_aoi(FIXED_AOI_ID)
        loaded["is_fixed_model_aoi"] = True
        return loaded

    polygon = _validate_polygon_latlng(FIXED_AOI_POLYGON_LATLNG, close_polygon=True)
    centroid = polygon.centroid
    utm_epsg = _utm_epsg_for_lonlat(centroid.x, centroid.y)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    feature = {
        "type": "Feature",
        "properties": {
            "aoi_id": FIXED_AOI_ID,
            "name": FIXED_AOI_NAME,
            "created": datetime.utcnow().isoformat() + "Z",
            "utm_epsg": utm_epsg,
            "bbox_wgs84": [min_lon, min_lat, max_lon, max_lat],
            "vertex_count": len(polygon.exterior.coords) - 1,
            "is_fixed_model_aoi": True,
        },
        "geometry": mapping(polygon),
    }

    collection = {"type": "FeatureCollection", "features": [feature]}
    with path.open("w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)

    return {
        "aoi_id": FIXED_AOI_ID,
        "name": FIXED_AOI_NAME,
        "bbox_wgs84": [min_lon, min_lat, max_lon, max_lat],
        "utm_epsg": utm_epsg,
        "filepath": str(path),
        "is_fixed_model_aoi": True,
        "polygon_latlng": FIXED_AOI_POLYGON_LATLNG,
    }


def load_aoi(aoi_id: str) -> dict[str, Any]:
    path = AOI_DIR / f"{aoi_id}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"AOI not found: {aoi_id}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    feature = data["features"][0]
    polygon = shape(feature["geometry"])
    props = feature.get("properties", {})
    bbox_from_geometry = [float(v) for v in polygon.bounds]
    utm_epsg = props.get("utm_epsg")
    if utm_epsg is None:
        centroid = polygon.centroid
        utm_epsg = _utm_epsg_for_lonlat(centroid.x, centroid.y)
    return {
        "aoi_id": props.get("aoi_id", aoi_id),
        "name": props.get("name", aoi_id),
        "polygon": polygon,
        "bbox_wgs84": bbox_from_geometry,
        "utm_epsg": int(utm_epsg),
        "filepath": str(path),
        "geojson": data,
        "is_fixed_model_aoi": bool(props.get("is_fixed_model_aoi", False)),
        "polygon_latlng": _polygon_to_latlng(polygon),
    }


def list_aois() -> list[dict[str, Any]]:
    AOI_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for geojson_path in sorted(AOI_DIR.glob("*.geojson"), reverse=True):
        try:
            with geojson_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            feature = data["features"][0]
            props = feature.get("properties", {})
            items.append(
                {
                    "aoi_id": props.get("aoi_id", geojson_path.stem),
                    "name": props.get("name", geojson_path.stem),
                    "created": props.get("created"),
                    "utm_epsg": props.get("utm_epsg"),
                    "bbox_wgs84": props.get("bbox_wgs84"),
                    "is_fixed_model_aoi": bool(props.get("is_fixed_model_aoi", False)),
                }
            )
        except Exception:
            continue
    return items
