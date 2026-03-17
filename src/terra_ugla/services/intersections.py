"""Transect generation and line/transect intersection services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point, mapping
from shapely.ops import transform as shapely_transform


@dataclass
class Transect:
    transect_id: int
    line_utm: LineString
    line_wgs84: LineString


def transform_geometry(geom, src_epsg: int, dst_epsg: int):
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
    return shapely_transform(transformer.transform, geom)


def _perpendicular_unit_vector(line: LineString, distance: float) -> tuple[float, float]:
    start_d = max(0.0, distance - 1.0)
    end_d = min(line.length, distance + 1.0)
    p0 = line.interpolate(start_d)
    p1 = line.interpolate(end_d)

    dx = p1.x - p0.x
    dy = p1.y - p0.y
    norm = np.hypot(dx, dy)
    if norm == 0:
        return 0.0, 1.0
    dx /= norm
    dy /= norm
    return -dy, dx


def generate_transects_from_baseline(
    baseline_line_wgs84: LineString,
    utm_epsg: int,
    spacing_m: float = 100.0,
    length_m: float = 500.0,
    offshore_ratio: float = 0.7,
) -> list[Transect]:
    """Generate fixed baseline transects in projected CRS and return both CRS views."""
    baseline_utm = transform_geometry(baseline_line_wgs84, 4326, utm_epsg)
    if baseline_utm.length <= 0:
        return []

    spacing_m = max(float(spacing_m), 1.0)
    length_m = max(float(length_m), 10.0)
    offshore_ratio = min(max(float(offshore_ratio), 0.0), 1.0)

    distances = list(np.arange(0, baseline_utm.length, spacing_m))
    if not distances or distances[-1] != baseline_utm.length:
        distances.append(baseline_utm.length)

    transects: list[Transect] = []
    onshore_len = length_m * (1.0 - offshore_ratio)
    offshore_len = length_m * offshore_ratio

    for i, d in enumerate(distances):
        center = baseline_utm.interpolate(float(d))
        perp_dx, perp_dy = _perpendicular_unit_vector(baseline_utm, float(d))

        start = Point(center.x - (perp_dx * onshore_len), center.y - (perp_dy * onshore_len))
        end = Point(center.x + (perp_dx * offshore_len), center.y + (perp_dy * offshore_len))
        line_utm = LineString([start, end])
        line_wgs84 = transform_geometry(line_utm, utm_epsg, 4326)
        transects.append(Transect(transect_id=i, line_utm=line_utm, line_wgs84=line_wgs84))

    return transects


def _as_points(intersection_geom) -> list[Point]:
    if intersection_geom.is_empty:
        return []
    gtype = intersection_geom.geom_type
    if gtype == "Point":
        return [intersection_geom]
    if gtype == "MultiPoint":
        return list(intersection_geom.geoms)
    if gtype == "LineString":
        return [intersection_geom.interpolate(0.5, normalized=True)]
    if gtype == "MultiLineString":
        pts = [line.interpolate(0.5, normalized=True) for line in intersection_geom.geoms if line.length > 0]
        return pts
    if hasattr(intersection_geom, "geoms"):
        pts: list[Point] = []
        for geom in intersection_geom.geoms:
            pts.extend(_as_points(geom))
        return pts
    return []


def intersect_line_with_transects(
    target_line_utm: LineString | MultiLineString | None,
    transects: list[Transect],
    utm_epsg: int,
) -> dict[int, dict[str, Any] | None]:
    """Intersect one extracted line with fixed transects and compute signed distances."""
    result: dict[int, dict[str, Any] | None] = {}

    if target_line_utm is None or target_line_utm.is_empty:
        for transect in transects:
            result[transect.transect_id] = None
        return result

    for transect in transects:
        intersection = transect.line_utm.intersection(target_line_utm)
        points = _as_points(intersection)
        if not points:
            result[transect.transect_id] = None
            continue

        midpoint = transect.line_utm.interpolate(0.5, normalized=True)
        point = min(points, key=lambda p: p.distance(midpoint))

        dist_from_start = transect.line_utm.project(point)
        signed_dist = float(dist_from_start - (transect.line_utm.length / 2.0))

        point_wgs84 = transform_geometry(point, utm_epsg, 4326)
        result[transect.transect_id] = {
            "point_utm": point,
            "point_wgs84": point_wgs84,
            "distance_signed_m": signed_dist,
            "distance_from_start_m": float(dist_from_start),
        }

    return result


def build_intersection_timeseries(
    per_scene_results: list[dict[str, Any]],
    transects: list[Transect],
    utm_epsg: int,
) -> pd.DataFrame:
    """Build tabular timeseries from VE/WL transect intersections."""
    rows: list[dict[str, Any]] = []

    for scene in per_scene_results:
        wl_map = intersect_line_with_transects(scene.get("waterline_utm"), transects, utm_epsg)
        ve_map = intersect_line_with_transects(scene.get("vegline_utm"), transects, utm_epsg)

        for transect in transects:
            wl = wl_map.get(transect.transect_id)
            ve = ve_map.get(transect.transect_id)
            row = {
                "run_id": scene["run_id"],
                "scene_id": scene["scene_id"],
                "datetime": scene["datetime"],
                "transect_id": transect.transect_id,
                "WL_distance_m": wl["distance_signed_m"] if wl else np.nan,
                "VE_distance_m": ve["distance_signed_m"] if ve else np.nan,
                "wl_lon": wl["point_wgs84"].x if wl else np.nan,
                "wl_lat": wl["point_wgs84"].y if wl else np.nan,
                "ve_lon": ve["point_wgs84"].x if ve else np.nan,
                "ve_lat": ve["point_wgs84"].y if ve else np.nan,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values(["datetime", "transect_id"]).reset_index(drop=True)
    return df


def transects_to_geojson(transects: list[Transect], run_id: str) -> dict[str, Any]:
    features = []
    for transect in transects:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "run_id": run_id,
                    "transect_id": transect.transect_id,
                },
                "geometry": mapping(transect.line_wgs84),
            }
        )
    return {"type": "FeatureCollection", "features": features}
