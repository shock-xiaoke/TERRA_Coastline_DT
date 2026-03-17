"""Baseline management service for coastal LineString-first workflows."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point, mapping, shape
from shapely.validation import explain_validity

BASELINE_DIR = Path("data") / "baselines"
TRANSECT_DIR = Path("data") / "transects"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "baseline"


def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def _validate_line_latlng(line_latlng: list[list[float]]) -> LineString:
    if not isinstance(line_latlng, list) or len(line_latlng) < 2:
        raise ValueError("Baseline polyline requires at least 2 vertices")

    coords_lonlat: list[tuple[float, float]] = []
    for idx, vertex in enumerate(line_latlng):
        if not isinstance(vertex, list) or len(vertex) != 2:
            raise ValueError(f"Invalid vertex at index {idx}")
        lat, lon = float(vertex[0]), float(vertex[1])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"Invalid latitude/longitude at index {idx}")
        coords_lonlat.append((lon, lat))

    line = LineString(coords_lonlat)
    if line.length <= 0:
        raise ValueError("Baseline line length must be greater than zero")
    return line


def _line_to_utm(line_wgs84: LineString, utm_epsg: int) -> LineString:
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    coords_utm = [to_utm.transform(x, y) for x, y in line_wgs84.coords]
    return LineString(coords_utm)


def _line_to_wgs84(line_utm: LineString, utm_epsg: int) -> LineString:
    to_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    coords_wgs84 = [to_wgs84.transform(x, y) for x, y in line_utm.coords]
    return LineString(coords_wgs84)


def _sample_distances(total_length_m: float, spacing_m: float) -> list[float]:
    if total_length_m <= 0:
        return []

    distances = list(np.arange(0.0, total_length_m, spacing_m))
    if not distances or (total_length_m - distances[-1]) > 1e-6:
        distances.append(float(total_length_m))
    return [float(d) for d in distances]


def _normal_unit_vector(line_utm: LineString, distance_m: float) -> tuple[float, float]:
    """
    Compute local normal using a moving-window secant tangent.
    This mimics legacy segment_start/segment_end behavior in metric CRS.
    """
    # Window size: 10 m minimum or 1% of baseline length.
    delta = max(10.0, float(line_utm.length) * 0.01)
    start_d = max(0.0, float(distance_m) - delta)
    end_d = min(float(line_utm.length), float(distance_m) + delta)
    if (end_d - start_d) < 1e-6:
        # Fallback window for very short lines.
        start_d = max(0.0, float(distance_m) - 0.5)
        end_d = min(float(line_utm.length), float(distance_m) + 0.5)
    if (end_d - start_d) < 1e-9:
        raise ValueError("Cannot compute normal vector from a zero-length secant window")

    p0 = line_utm.interpolate(float(start_d))
    p1 = line_utm.interpolate(float(end_d))
    dx = float(p1.x - p0.x)
    dy = float(p1.y - p0.y)
    norm = float(np.hypot(dx, dy))
    if norm <= 0:
        raise ValueError("Cannot compute normal vector on a zero-length baseline secant")

    # Perpendicular vector: rotate tangent 90 degrees.
    return (-(dy / norm), dx / norm)


def create_baseline(name: str, line_latlng: list[list[float]]) -> dict[str, Any]:
    """Validate and persist a baseline LineString."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    line = _validate_line_latlng(line_latlng)
    if not line.is_valid:
        raise ValueError(f"Invalid baseline line: {explain_validity(line)}")
    centroid = line.centroid
    utm_epsg = _utm_epsg_for_lonlat(centroid.x, centroid.y)
    min_lon, min_lat, max_lon, max_lat = line.bounds

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    baseline_id = f"{_slugify(name)}_{timestamp}"

    feature = {
        "type": "Feature",
        "properties": {
            "baseline_id": baseline_id,
            "name": name,
            "created": datetime.utcnow().isoformat() + "Z",
            "utm_epsg": utm_epsg,
            "bbox_wgs84": [min_lon, min_lat, max_lon, max_lat],
            "vertex_count": len(line.coords),
        },
        "geometry": mapping(line),
    }

    collection = {"type": "FeatureCollection", "features": [feature]}
    path = BASELINE_DIR / f"{baseline_id}.geojson"
    with path.open("w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)

    return {
        "baseline_id": baseline_id,
        "name": name,
        "bbox_wgs84": [min_lon, min_lat, max_lon, max_lat],
        "utm_epsg": utm_epsg,
        "filepath": str(path),
    }


def load_baseline(baseline_id: str) -> dict[str, Any]:
    path = BASELINE_DIR / f"{baseline_id}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_id}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    feature = data["features"][0]
    line = shape(feature["geometry"])
    if line.geom_type != "LineString":
        raise ValueError(f"Baseline geometry must be LineString, got: {line.geom_type}")

    props = feature.get("properties", {})
    return {
        "baseline_id": props.get("baseline_id", baseline_id),
        "name": props.get("name", baseline_id),
        "line": line,
        "bbox_wgs84": props.get("bbox_wgs84", list(line.bounds)),
        "utm_epsg": int(props["utm_epsg"]),
        "filepath": str(path),
        "geojson": data,
    }


def list_baselines() -> list[dict[str, Any]]:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for geojson_path in sorted(BASELINE_DIR.glob("*.geojson"), reverse=True):
        try:
            with geojson_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            feature = data["features"][0]
            props = feature.get("properties", {})
            items.append(
                {
                    "baseline_id": props.get("baseline_id", geojson_path.stem),
                    "name": props.get("name", geojson_path.stem),
                    "created": props.get("created"),
                    "utm_epsg": props.get("utm_epsg"),
                    "bbox_wgs84": props.get("bbox_wgs84"),
                }
            )
        except Exception:
            continue
    return items


def generate_transects(
    baseline_id: str,
    spacing_m: float = 20.0,
    transect_length_m: float = 500.0,
    offshore_ratio: float = 0.7,
) -> dict[str, Any]:
    """Generate perpendicular transects from a saved baseline and persist as GeoJSON."""
    TRANSECT_DIR.mkdir(parents=True, exist_ok=True)

    if spacing_m <= 0:
        raise ValueError("spacing_m must be greater than zero")
    if transect_length_m <= 0:
        raise ValueError("transect_length_m must be greater than zero")
    if not (0.0 <= float(offshore_ratio) <= 1.0):
        raise ValueError("offshore_ratio must be between 0 and 1")

    baseline = load_baseline(baseline_id)
    line_wgs84: LineString = baseline["line"]
    utm_epsg = int(baseline["utm_epsg"])

    # All geometry math is done in projected UTM (meters).
    line_utm = _line_to_utm(line_wgs84, utm_epsg)
    if line_utm.length <= 0:
        raise ValueError("Baseline has zero length after projection")

    offshore_length = float(transect_length_m) * float(offshore_ratio)
    onshore_length = float(transect_length_m) * (1.0 - float(offshore_ratio))

    transect_features: list[dict[str, Any]] = []
    distances = _sample_distances(float(line_utm.length), float(spacing_m))

    for idx, distance_m in enumerate(distances, start=1):
        origin = line_utm.interpolate(distance_m)
        nx, ny = _normal_unit_vector(line_utm, distance_m)

        onshore_point = Point(
            float(origin.x - (nx * onshore_length)),
            float(origin.y - (ny * onshore_length)),
        )
        offshore_point = Point(
            float(origin.x + (nx * offshore_length)),
            float(origin.y + (ny * offshore_length)),
        )

        transect_utm = LineString([onshore_point, offshore_point])
        transect_wgs84 = _line_to_wgs84(transect_utm, utm_epsg)

        transect_features.append(
            {
                "type": "Feature",
                "properties": {
                    "transect_id": f"T{idx:03d}",
                    "transect_index": idx - 1,
                    "baseline_id": baseline_id,
                    "distance_along_baseline": float(distance_m),
                    "spacing_m": float(spacing_m),
                    "transect_length_m": float(transect_length_m),
                    "offshore_ratio": float(offshore_ratio),
                    "offshore_length_m": float(offshore_length),
                    "onshore_length_m": float(onshore_length),
                    "utm_epsg": utm_epsg,
                },
                "geometry": mapping(transect_wgs84),
            }
        )

    collection = {
        "type": "FeatureCollection",
        "features": transect_features,
        "metadata": {
            "baseline_id": baseline_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "utm_epsg": utm_epsg,
            "spacing_m": float(spacing_m),
            "transect_length_m": float(transect_length_m),
            "offshore_ratio": float(offshore_ratio),
            "offshore_length_m": float(offshore_length),
            "onshore_length_m": float(onshore_length),
            "baseline_length_m": float(line_utm.length),
            "transect_count": len(transect_features),
        },
    }

    path = TRANSECT_DIR / f"{baseline_id}_transects.geojson"
    with path.open("w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)

    return {
        "baseline_id": baseline_id,
        "transect_count": len(transect_features),
        "utm_epsg": utm_epsg,
        "spacing_m": float(spacing_m),
        "transect_length_m": float(transect_length_m),
        "offshore_ratio": float(offshore_ratio),
        "offshore_length_m": float(offshore_length),
        "onshore_length_m": float(onshore_length),
        "filepath": str(path),
        "geojson": collection,
    }
