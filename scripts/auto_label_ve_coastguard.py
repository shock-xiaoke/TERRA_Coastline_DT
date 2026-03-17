#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-label vegetation edge (ve) using COASTGUARD-style extraction.

This script is designed for monthly PNG previews like:
  data/labelme_work/aoi_01_02_2016_2020/*.png

Workflow (aligned with VedgeSat_DriverTemplate extraction logic):
1. Parse each PNG name -> run_id + scene_id.
2. Locate matching 5-band TIF under data/runs/<run_id>/imagery/<scene_id>.tif
3. Vegetation/non-vegetation classification.
4. NDVI computation.
5. Weighted-peaks threshold contour extraction (COASTGUARD WP method).
6. Keep dominant vegetation edge and export LabelMe JSON ("ve" linestrip).

Optional:
- Save overlay QC images.
- Save per-image CSV report.

Example:
  python scripts/auto_label_ve_coastguard.py ^
      --png-dir data/labelme_work/aoi_01_02_2016_2020 ^
      --runs-dir data/runs ^
      --output-dir data/labelme_work/aoi_01_02_2016_2020
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from scipy import ndimage, signal
from skimage import measure, morphology
from sklearn.neighbors import KernelDensity
from shapely.geometry import LineString, Point
from shapely.ops import transform as shp_transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.terra_ugla.coastguard_port import classify_image_nn, nd_index  # noqa: E402
from src.terra_ugla.coastguard_port.classification import load_classifier  # noqa: E402


LABELME_VERSION = "5.11.4"
PNG_SUFFIX = ".png"
DEFAULT_MODEL_CANDIDATES = [
    PROJECT_ROOT / "data" / "models" / "coastguard" / "MLPClassifier_Veg_L5L8S2.pkl",
    PROJECT_ROOT / "COASTGUARD" / "Classification" / "models" / "MLPClassifier_Veg_L5L8S2.pkl",
    PROJECT_ROOT / "COASTGUARD" / "Classification" / "models" / "MLPClassifier_Veg_L8S2.pkl",
]


@dataclass
class RefLineState:
    line: LineString
    crs: CRS
    source: str = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label vegetation edges using COASTGUARD weighted-peaks contour extraction."
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "labelme_work" / "aoi_01_02_2016_2020",
        help="Input directory containing PNG previews to label.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "runs",
        help="Root run directory that stores matching TIF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for JSON/PNG pairs. Default: same as --png-dir.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional vegetation classifier model path (.pkl).",
    )
    parser.add_argument(
        "--max-dist-ref-m",
        type=float,
        default=150.0,
        help="Reference line buffer distance (meters), like VedgeSat max_dist_ref.",
    )
    parser.add_argument(
        "--min-line-length-m",
        type=float,
        default=500.0,
        help="Minimum accepted ve line length (meters), like VedgeSat min_length_sl.",
    )
    parser.add_argument(
        "--min-patch-size-px",
        type=int,
        default=30,
        help=(
            "Minimum connected component size (pixels) kept after classification. "
            "Default 30 follows src/terra_ugla/services/extraction.py."
        ),
    )
    parser.add_argument(
        "--min-beach-area-m2",
        type=float,
        default=200.0,
        help=(
            "Deprecated fallback for min patch sizing in m^2. "
            "Only used when --min-patch-size-px <= 0."
        ),
    )
    parser.add_argument(
        "--shape-mode",
        choices=["profile", "contour"],
        default="contour",
        help=(
            "'profile': one y per x (stable for downstream timeseries). "
            "'contour': raw contour points from extracted ve."
        ),
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=0,
        help=(
            "If > 0, force fixed final points for each ve linestrip. "
            "If <= 0, use adaptive point count."
        ),
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=64,
        help="Minimum final points when adaptive point-count mode is used.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=512,
        help="Maximum final points when adaptive point-count mode is used.",
    )
    parser.add_argument(
        "--points-per-100px",
        type=float,
        default=35.0,
        help="Adaptive density: points per 100 pixels of ve polyline length.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="Savitzky-Golay smoothing window (odd, >=5) for ve line regularization.",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order for ve line smoothing.",
    )
    parser.add_argument(
        "--smooth-passes",
        type=int,
        default=2,
        help="Number of smoothing passes for ve line regularization.",
    )
    parser.add_argument(
        "--manual-ref-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory of LabelMe JSON seed files (one manual reference line per AOI). "
            "When provided, these reference lines are used as fixed buffers."
        ),
    )
    parser.add_argument(
        "--manual-ref-label",
        type=str,
        default="refline",
        help="Label name in manual ref JSON shapes (default: refline).",
    )
    parser.add_argument(
        "--transect-spacing-m",
        type=float,
        default=40.0,
        help="Spacing (meters) for fixed transects generated from the reference line.",
    )
    parser.add_argument(
        "--transect-length-m",
        type=float,
        default=260.0,
        help="Total transect length (meters) used for contour scoring.",
    )
    parser.add_argument(
        "--transect-offshore-ratio",
        type=float,
        default=0.7,
        help="Offshore part ratio of transect length (0~1).",
    )
    parser.add_argument(
        "--transect-corridor-m",
        type=float,
        default=18.0,
        help="Half-width (meters) of transect sampling corridor used for thresholding.",
    )
    parser.add_argument(
        "--min-transect-hit-ratio",
        type=float,
        default=0.08,
        help="Minimum transect hit ratio for candidate contour acceptance when manual ref exists.",
    )
    parser.add_argument(
        "--transect-reconstruct-min-hit-ratio",
        type=float,
        default=0.35,
        help="Minimum transect hit ratio to accept transect-based full-line reconstruction.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy source PNG into output dir when output dir differs from input dir.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Optional directory to save quick QC overlays.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Optional CSV report path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extraction but do not write outputs.",
    )
    return parser.parse_args()


def resolve_model_path(user_model_path: Path | None) -> Path | None:
    if user_model_path is not None:
        return user_model_path if user_model_path.exists() else None
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def parse_png_name(png_path: Path) -> tuple[str, str] | None:
    name = png_path.name
    if "__" not in name or not name.lower().endswith(PNG_SUFFIX):
        return None
    stem = name[: -len(PNG_SUFFIX)]
    run_id, scene_id = stem.split("__", 1)
    if not run_id or not scene_id:
        return None
    return run_id, scene_id


def parse_aoi_key(run_id: str) -> str:
    match = re.match(r"^(aoi_\d+)", run_id, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    parts = run_id.split("_")
    return "_".join(parts[:2]).lower() if len(parts) >= 2 else run_id.lower()


def extract_points_from_shape(shape: dict[str, Any]) -> list[list[float]]:
    pts = shape.get("points", []) or []
    out: list[list[float]] = []
    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                out.append([float(p[0]), float(p[1])])
            except Exception:
                continue
    return out


def resolve_image_from_labelme_json(json_path: Path, payload: dict[str, Any], png_dir: Path) -> Path | None:
    image_path = str(payload.get("imagePath", "") or "").strip()
    candidates: list[Path] = []

    if image_path:
        candidates.append((json_path.parent / image_path).resolve())
        candidates.append((png_dir / image_path).resolve())

    candidates.append((json_path.with_suffix(".png")).resolve())
    candidates.append((png_dir / f"{json_path.stem}.png").resolve())

    for c in candidates:
        if c.exists():
            return c
    return None


def find_tif_path(runs_dir: Path, run_id: str, scene_id: str) -> Path | None:
    direct = runs_dir / run_id / "imagery" / f"{scene_id}.tif"
    if direct.exists():
        return direct

    manifest = runs_dir / run_id / "manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            scenes = payload.get("scenes", []) or []
            for scene in scenes:
                download = scene.get("download", {}) or {}
                if str(download.get("scene_id", "")) == scene_id:
                    rel = str(download.get("filepath", "")).replace("\\", "/")
                    if rel:
                        candidate = (PROJECT_ROOT / rel).resolve()
                        if candidate.exists():
                            return candidate
        except Exception:
            pass

    candidates = list(runs_dir.glob(f"**/imagery/{scene_id}.tif"))
    if candidates:
        return candidates[0]
    return None


def load_manual_reference_lines(
    manual_ref_dir: Path,
    png_dir: Path,
    runs_dir: Path,
    manual_ref_label: str,
) -> dict[str, RefLineState]:
    """
    Load manual reference lines from LabelMe JSON files.

    Expected workflow:
    - User labels one line per AOI in LabelMe with label == manual_ref_label.
    - JSON imagePath points to a PNG scene in png_dir (or same folder).
    - We map JSON pixel coordinates -> matched scene TIF world coords.
    - Reference is cached per AOI key, fixed for all months.
    """
    refs: dict[str, RefLineState] = {}
    if not manual_ref_dir.exists():
        return refs

    json_files = sorted([p for p in manual_ref_dir.glob("*.json") if p.is_file()])
    for js in json_files:
        try:
            payload = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue

        shapes = payload.get("shapes", []) or []
        selected_points: list[list[float]] = []
        best_len = -1.0
        for shp in shapes:
            label = str(shp.get("label", "")).strip().lower()
            if label != manual_ref_label.strip().lower():
                continue
            shape_type = str(shp.get("shape_type", "line")).strip().lower()
            if shape_type not in {"line", "linestrip", "polyline"}:
                continue
            pts = extract_points_from_shape(shp)
            if len(pts) >= 2 and len(pts) > best_len:
                selected_points = pts
                best_len = len(pts)
        if len(selected_points) < 2:
            continue

        png_path = resolve_image_from_labelme_json(js, payload, png_dir=png_dir)
        if png_path is None:
            continue

        parsed = parse_png_name(png_path)
        if parsed is None:
            continue
        run_id, scene_id = parsed
        aoi_key = parse_aoi_key(run_id)
        if aoi_key in refs:
            # Keep first seed per AOI.
            continue

        tif_path = find_tif_path(runs_dir, run_id, scene_id)
        if tif_path is None:
            continue

        img_w = int(payload.get("imageWidth") or 0)
        img_h = int(payload.get("imageHeight") or 0)
        if img_w <= 0 or img_h <= 0:
            try:
                with Image.open(png_path) as img:
                    img_w, img_h = img.size
            except Exception:
                continue
        if img_w <= 1 or img_h <= 1:
            continue

        try:
            with rasterio.open(tif_path) as src:
                src_crs = CRS.from_user_input(src.crs) if src.crs is not None else CRS.from_epsg(4326)
                h_tif, w_tif = src.height, src.width
                sx = float(w_tif) / float(img_w)
                sy = float(h_tif) / float(img_h)

                rows = []
                cols = []
                for x, y in selected_points:
                    cols.append(float(np.clip(x * sx, 0.0, w_tif - 1.0)))
                    rows.append(float(np.clip(y * sy, 0.0, h_tif - 1.0)))

                coords = []
                for r, c in zip(rows, cols):
                    xw, yw = rasterio.transform.xy(src.transform, r, c)
                    coords.append((float(xw), float(yw)))

                if len(coords) < 2:
                    continue
                line = LineString(coords)
                if not line.is_valid or line.length <= 0:
                    continue
                refs[aoi_key] = RefLineState(line=line, crs=src_crs, source="manual")
        except Exception:
            continue

    return refs


def make_cloud_mask(im_ms: np.ndarray) -> np.ndarray:
    invalid = np.any(~np.isfinite(im_ms), axis=2)
    nodata = np.all(im_ms == 0, axis=2)
    return np.logical_or(invalid, nodata)


def meters_to_crs_units(value_m: float, crs: CRS, lat_hint: float) -> float:
    if value_m <= 0:
        return 0.0
    if crs.is_geographic:
        cos_lat = max(abs(math.cos(math.radians(lat_hint))), 0.1)
        return value_m / (111_320.0 * cos_lat)
    return value_m


def reproject_line(line: LineString, src_crs: CRS, dst_crs: CRS) -> LineString:
    if src_crs == dst_crs:
        return line
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(transformer.transform, line)


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((float(lon) + 180.0) // 6.0) + 1
    return (32600 + zone) if float(lat) >= 0.0 else (32700 + zone)


def choose_metric_crs_for_line(line: LineString, src_crs: CRS) -> CRS:
    if not src_crs.is_geographic:
        units = [str(axis.unit_name or "").lower() for axis in (src_crs.axis_info or [])]
        if any(("metre" in u) or ("meter" in u) for u in units):
            return src_crs

    wgs84 = CRS.from_epsg(4326)
    line_wgs84 = reproject_line(line, src_crs, wgs84)
    centroid = line_wgs84.centroid
    epsg = utm_epsg_for_lonlat(float(centroid.x), float(centroid.y))
    return CRS.from_epsg(epsg)


def perpendicular_unit_vector_metric(line: LineString, distance_m: float) -> tuple[float, float]:
    delta = max(10.0, float(line.length) * 0.01)
    start_d = max(0.0, float(distance_m) - delta)
    end_d = min(float(line.length), float(distance_m) + delta)
    if (end_d - start_d) < 1e-6:
        start_d = max(0.0, float(distance_m) - 0.5)
        end_d = min(float(line.length), float(distance_m) + 0.5)

    p0 = line.interpolate(start_d)
    p1 = line.interpolate(end_d)
    dx = float(p1.x - p0.x)
    dy = float(p1.y - p0.y)
    norm = float(np.hypot(dx, dy))
    if norm <= 0:
        return 0.0, 1.0
    return (-(dy / norm), dx / norm)


def generate_transects_from_refline(
    ref_line_src: LineString,
    src_crs: CRS,
    spacing_m: float,
    length_m: float,
    offshore_ratio: float,
) -> tuple[CRS | None, list[LineString], list[LineString]]:
    if (ref_line_src is None) or (ref_line_src.length <= 0):
        return None, [], []

    spacing_m = max(float(spacing_m), 1.0)
    length_m = max(float(length_m), 10.0)
    offshore_ratio = min(max(float(offshore_ratio), 0.0), 1.0)

    metric_crs = choose_metric_crs_for_line(ref_line_src, src_crs)
    ref_line_metric = reproject_line(ref_line_src, src_crs, metric_crs)
    if (not ref_line_metric.is_valid) or (ref_line_metric.length <= 0):
        return metric_crs, [], []

    distances = list(np.arange(0.0, float(ref_line_metric.length), spacing_m))
    if not distances or (float(ref_line_metric.length) - float(distances[-1])) > 1e-6:
        distances.append(float(ref_line_metric.length))

    onshore_len = float(length_m) * (1.0 - offshore_ratio)
    offshore_len = float(length_m) * offshore_ratio

    transects_metric: list[LineString] = []
    transects_src: list[LineString] = []

    for d in distances:
        center = ref_line_metric.interpolate(float(d))
        nx, ny = perpendicular_unit_vector_metric(ref_line_metric, float(d))
        start = Point(float(center.x - (nx * onshore_len)), float(center.y - (ny * onshore_len)))
        end = Point(float(center.x + (nx * offshore_len)), float(center.y + (ny * offshore_len)))
        line_metric = LineString([start, end])
        if (not line_metric.is_valid) or (line_metric.length <= 0):
            continue
        transects_metric.append(line_metric)
        transects_src.append(reproject_line(line_metric, metric_crs, src_crs))

    return metric_crs, transects_metric, transects_src


def intersection_points(intersection_geom) -> list[Point]:
    if intersection_geom.is_empty:
        return []
    gtype = intersection_geom.geom_type
    if gtype == "Point":
        return [intersection_geom]
    if gtype == "MultiPoint":
        return list(intersection_geom.geoms)
    if gtype == "LineString":
        if intersection_geom.length <= 0:
            return []
        return [intersection_geom.interpolate(0.5, normalized=True)]
    if gtype == "MultiLineString":
        points: list[Point] = []
        for geom in intersection_geom.geoms:
            if geom.length > 0:
                points.append(geom.interpolate(0.5, normalized=True))
        return points
    if hasattr(intersection_geom, "geoms"):
        points = []
        for geom in intersection_geom.geoms:
            points.extend(intersection_points(geom))
        return points
    return []


def contour_transect_stats(line_metric: LineString, transects_metric: list[LineString]) -> tuple[float, float, int]:
    if (not transects_metric) or (line_metric is None) or line_metric.is_empty:
        return 0.0, float("inf"), 0

    hits = 0
    abs_offsets: list[float] = []

    for transect in transects_metric:
        intersection = transect.intersection(line_metric)
        points = intersection_points(intersection)
        if not points:
            continue

        midpoint = transect.interpolate(0.5, normalized=True)
        point = min(points, key=lambda p: p.distance(midpoint))
        dist_from_start = float(transect.project(point))
        signed_offset = dist_from_start - (float(transect.length) / 2.0)
        abs_offsets.append(abs(float(signed_offset)))
        hits += 1

    hit_ratio = float(hits) / float(len(transects_metric))
    median_abs_offset = float(np.median(abs_offsets)) if abs_offsets else float("inf")
    return hit_ratio, median_abs_offset, hits


def build_transect_corridor_mask(
    src: rasterio.io.DatasetReader,
    cloud_mask: np.ndarray,
    transects_src: list[LineString],
    src_crs: CRS,
    corridor_m: float,
) -> np.ndarray | None:
    if (not transects_src) or (float(corridor_m) <= 0.0):
        return None

    if src_crs.is_geographic:
        lat_hint = float((src.bounds.bottom + src.bounds.top) * 0.5)
    else:
        lat_hint = float(transects_src[0].centroid.y)
    dist_units = meters_to_crs_units(float(corridor_m), src_crs, lat_hint)
    if dist_units <= 0:
        return None

    geoms = []
    for line in transects_src:
        if line is None or line.length <= 0:
            continue
        geoms.append((line.buffer(dist_units), 1))
    if not geoms:
        return None

    mask = rasterize(
        geoms,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    mask = np.logical_and(mask, ~cloud_mask)
    if float(mask.mean()) <= 0:
        return None
    return mask


def build_ref_mask_edge_distance_px(ref_mask: np.ndarray) -> np.ndarray | None:
    if ref_mask is None or ref_mask.size == 0:
        return None

    se = morphology.disk(1)
    edge = np.logical_xor(
        morphology.binary_dilation(ref_mask, se),
        morphology.binary_erosion(ref_mask, se),
    )
    if not np.any(edge):
        return None
    return ndimage.distance_transform_edt(~edge)


def build_ref_buffer_mask(
    src: rasterio.io.DatasetReader,
    cloud_mask: np.ndarray,
    ref_state: RefLineState | None,
    max_dist_ref_m: float,
) -> np.ndarray:
    if ref_state is None:
        return ~cloud_mask

    src_crs = CRS.from_user_input(src.crs) if src.crs is not None else CRS.from_epsg(4326)
    ref_in_src = reproject_line(ref_state.line, ref_state.crs, src_crs)
    lat_hint = float(ref_in_src.centroid.y)
    dist_units = meters_to_crs_units(max_dist_ref_m, src_crs, lat_hint)
    buffered = ref_in_src.buffer(dist_units)

    mask = rasterize(
        [(buffered, 1)],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    return np.logical_and(mask, ~cloud_mask)


def min_line_length_units(src: rasterio.io.DatasetReader, min_line_length_m: float) -> float:
    src_crs = CRS.from_user_input(src.crs) if src.crs is not None else CRS.from_epsg(4326)
    bounds = src.bounds
    lat_hint = float((bounds.bottom + bounds.top) * 0.5)
    return meters_to_crs_units(min_line_length_m, src_crs, lat_hint)


def estimate_pixel_area_m2(src: rasterio.io.DatasetReader) -> float:
    """
    Estimate pixel area in m^2 from affine transform and CRS.

    For projected CRS (e.g., UTM): transform units are meters, so area is direct.
    For geographic CRS (degrees): convert per-axis degree lengths to meters at scene latitude.
    """
    a = abs(float(src.transform.a))
    e = abs(float(src.transform.e))
    if a <= 0 or e <= 0:
        return 1.0

    src_crs = CRS.from_user_input(src.crs) if src.crs is not None else CRS.from_epsg(4326)
    if src_crs.is_geographic:
        lat = float((src.bounds.bottom + src.bounds.top) * 0.5)
        cos_lat = max(abs(math.cos(math.radians(lat))), 0.1)
        dx_m = a * 111_320.0 * cos_lat
        dy_m = e * 111_320.0
        area = dx_m * dy_m
    else:
        area = a * e

    if not np.isfinite(area) or area <= 0:
        return 1.0
    return float(area)


def primary_contour_from_candidates(
    contours: list[np.ndarray],
    transform,
    src_crs: CRS,
    min_len_units: float,
    image_height: int,
    image_width: int,
    ref_line_src_crs: LineString | None = None,
    metric_crs: CRS | None = None,
    transects_metric: list[LineString] | None = None,
    transect_length_m: float = 0.0,
    min_transect_hit_ratio: float = 0.0,
    ref_mask_edge_dist_px: np.ndarray | None = None,
    edge_proximity_px: float = 1.5,
) -> tuple[np.ndarray | None, LineString | None, dict[str, float]]:
    if not contours:
        return None, None, {}

    best_contour = None
    best_line = None
    best_score = -1.0
    best_meta: dict[str, float] = {}
    for contour in contours:
        coords = []
        for row, col in contour:
            x, y = rasterio.transform.xy(transform, row, col)
            coords.append((float(x), float(y)))
        if len(coords) < 2:
            continue
        line = LineString(coords)
        if not line.is_valid:
            continue
        if line.length < min_len_units:
            continue

        # Penalize contours attached to image borders (common false edges).
        rr = contour[:, 0]
        cc = contour[:, 1]
        rr_i = np.clip(np.round(rr).astype(int), 0, image_height - 1)
        cc_i = np.clip(np.round(cc).astype(int), 0, image_width - 1)
        border_hits = (
            (rr <= 2) | (rr >= image_height - 3) | (cc <= 2) | (cc >= image_width - 3)
        )
        border_frac = float(np.mean(border_hits)) if len(border_hits) > 0 else 0.0
        border_weight = max(0.05, 1.0 - border_frac)

        mask_edge_frac = 0.0
        mask_edge_weight = 1.0
        if ref_mask_edge_dist_px is not None:
            near_mask_edge = ref_mask_edge_dist_px[rr_i, cc_i] <= float(edge_proximity_px)
            mask_edge_frac = float(np.mean(near_mask_edge)) if len(near_mask_edge) > 0 else 0.0
            mask_edge_weight = max(0.05, 1.0 - mask_edge_frac)

        ref_dist = float("nan")
        ref_cover_ratio = float("nan")
        if ref_line_src_crs is not None:
            # Prefer long contours that remain close to reference line.
            ref_dist = float(line.distance(ref_line_src_crs))
            score = float((line.length / (1.0 + ref_dist)) * border_weight * mask_edge_weight)
            ref_len = float(ref_line_src_crs.length)
            if ref_len > 0:
                step = max(1, int(len(coords) // 200))
                proj_vals = [
                    float(ref_line_src_crs.project(Point(float(x), float(y))))
                    for x, y in coords[::step]
                ]
                if proj_vals:
                    span = max(proj_vals) - min(proj_vals)
                    ref_cover_ratio = float(np.clip(span / ref_len, 0.0, 1.0))
                    cover_weight = 0.6 + (0.8 * ref_cover_ratio)
                    score *= float(cover_weight)
        else:
            score = float(line.length * border_weight * mask_edge_weight)

        transect_hit_ratio = 0.0
        transect_hits = 0
        transect_offset_m = float("inf")
        if metric_crs is not None and transects_metric:
            line_metric = reproject_line(line, src_crs, metric_crs)
            transect_hit_ratio, transect_offset_m, transect_hits = contour_transect_stats(
                line_metric=line_metric,
                transects_metric=transects_metric,
            )
            if float(min_transect_hit_ratio) > 0.0 and transect_hit_ratio < float(min_transect_hit_ratio):
                continue
            transect_weight = 1.0 + (3.0 * transect_hit_ratio)
            offset_norm = max(float(transect_length_m) * 0.5, 1.0)
            if np.isfinite(transect_offset_m):
                offset_weight = 1.0 / (1.0 + (float(transect_offset_m) / offset_norm))
            else:
                offset_weight = 0.05
            score *= float(transect_weight * offset_weight)

        if score > best_score:
            best_score = score
            best_contour = contour
            best_line = line
            best_meta = {
                "score": float(score),
                "border_frac": float(border_frac),
                "mask_edge_frac": float(mask_edge_frac),
                "ref_dist": float(ref_dist) if np.isfinite(ref_dist) else float("nan"),
                "ref_cover_ratio": float(ref_cover_ratio) if np.isfinite(ref_cover_ratio) else float("nan"),
                "transect_hit_ratio": float(transect_hit_ratio),
                "transect_hits": float(transect_hits),
                "transect_offset_m": float(transect_offset_m) if np.isfinite(transect_offset_m) else float("nan"),
            }

    if best_line is None:
        return None, None, {}
    return best_contour, best_line, best_meta


def contour_to_profile_points(contour: np.ndarray, width: int, height: int) -> list[list[float]]:
    x = np.asarray(contour[:, 1], dtype=float)
    y = np.asarray(contour[:, 0], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.clip(x[valid], 0.0, float(width - 1))
    y = np.clip(y[valid], 0.0, float(height - 1))
    if len(x) < 3:
        return []

    cols = np.clip(np.round(x).astype(int), 0, width - 1)
    y_by_col = np.full((width,), np.nan, dtype=float)

    for col in np.unique(cols):
        y_by_col[col] = float(np.median(y[cols == col]))

    valid_cols = np.isfinite(y_by_col)
    if valid_cols.sum() < 3:
        return []

    # Keep only the contour span to avoid full-width straight tails.
    col_min = int(np.min(np.where(valid_cols)))
    col_max = int(np.max(np.where(valid_cols)))
    cols_span = np.arange(col_min, col_max + 1, dtype=float)
    y_interp = np.interp(cols_span, np.where(valid_cols)[0].astype(float), y_by_col[valid_cols])
    y_interp = np.clip(y_interp, 0.0, float(height - 1))

    return [[float(x), float(y)] for x, y in zip(cols_span, y_interp)]


def clip_index_vec(
    cloud_mask: np.ndarray,
    im_ndi: np.ndarray,
    im_labels: np.ndarray,
    im_ref_buffer: np.ndarray,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    COASTGUARD-style ClipIndexVec:
    - dilate ref buffer by 5 pixels
    - take veg/nonveg NDI values within buffered zone
    - balance class sample sizes before thresholding
    """
    nrows, ncols = cloud_mask.shape

    vec_ndi = im_ndi.reshape(nrows * ncols)
    vec_veg = im_labels[:, :, 0].reshape(nrows * ncols)
    vec_nonveg = im_labels[:, :, 1].reshape(nrows * ncols)

    se = morphology.disk(5)
    ref_extra = morphology.binary_dilation(im_ref_buffer, se)
    vec_buffer = ref_extra.reshape(nrows * ncols)

    int_veg = vec_ndi[np.logical_and(vec_buffer, vec_veg)]
    int_nonveg = vec_ndi[np.logical_and(vec_buffer, vec_nonveg)]
    int_veg = int_veg[np.isfinite(int_veg)]
    int_nonveg = int_nonveg[np.isfinite(int_nonveg)]

    if len(int_veg) == 0 or len(int_nonveg) == 0:
        return None, None

    rng = np.random.default_rng(seed)
    if len(int_veg) > len(int_nonveg):
        pick = rng.choice(len(int_veg), size=len(int_nonveg), replace=False)
        int_veg = int_veg[pick]
    elif len(int_nonveg) > len(int_veg):
        pick = rng.choice(len(int_nonveg), size=len(int_veg), replace=False)
        int_nonveg = int_nonveg[pick]
    return int_veg, int_nonveg


def find_wp_threshold(int_veg: np.ndarray, int_nonveg: np.ndarray) -> float:
    """
    COASTGUARD Toolbox.FindWPThresh-equivalent implementation.
    """
    bins = np.arange(-1.0, 1.0, 0.01)
    peaks: list[float] = []

    for idx, data in enumerate([int_veg, int_nonveg]):
        sample = np.asarray(data, dtype=float).reshape(-1, 1)
        sample = sample[np.isfinite(sample[:, 0])]
        if len(sample) < 10:
            return float((np.nanmedian(int_veg) * 0.2) + (np.nanmedian(int_nonveg) * 0.8))

        kde = KernelDensity(bandwidth=0.01, kernel="gaussian")
        kde.fit(sample)
        values = bins.reshape(-1, 1)
        probs = np.exp(kde.score_samples(values))

        if idx == 0:
            peaks.append(float(values[int(np.nanargmax(probs))][0]))
            continue

        prom = 0.5
        prom_idx, _ = signal.find_peaks(probs, prominence=prom)
        while len(prom_idx) == 0 and prom > 0.05:
            prom -= 0.05
            prom_idx, _ = signal.find_peaks(probs, prominence=prom)

        if len(prom_idx) == 0:
            peaks.append(float(values[int(np.nanargmax(probs))][0]))
        elif len(prom_idx) > 1:
            candidate_bins = bins[prom_idx]
            if np.any(candidate_bins < 0):
                peaks.append(float(np.max(candidate_bins)))
            else:
                peaks.append(float(candidate_bins[0]))
        else:
            peaks.append(float(bins[prom_idx[0]]))

    return float((0.2 * peaks[0]) + (0.8 * peaks[1]))


def process_contours(contours: list[np.ndarray]) -> list[np.ndarray]:
    """COASTGUARD-style contour cleanup: remove NaN vertices and tiny pieces."""
    cleaned: list[np.ndarray] = []
    for contour in contours:
        if contour is None:
            continue
        arr = np.asarray(contour)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        if np.isnan(arr).any():
            arr = arr[~np.isnan(arr).any(axis=1)]
        if len(arr) > 1:
            cleaned.append(arr)
    return cleaned


def contour_to_raw_points(contour: np.ndarray, width: int, height: int) -> list[list[float]]:
    points = []
    for row, col in contour:
        x = float(np.clip(col, 0.0, width - 1.0))
        y = float(np.clip(row, 0.0, height - 1.0))
        points.append([x, y])
    return points


def world_to_pixel_points(
    world_points: list[tuple[float, float]],
    transform,
    width: int,
    height: int,
) -> list[list[float]]:
    out: list[list[float]] = []
    inv = ~transform
    for x, y in world_points:
        col, row = inv * (float(x), float(y))
        col = float(np.clip(col, 0.0, float(width - 1)))
        row = float(np.clip(row, 0.0, float(height - 1)))
        if np.isfinite(col) and np.isfinite(row):
            out.append([col, row])
    return out


def reconstruct_line_from_transects(
    contours: list[np.ndarray],
    transform,
    transects_src: list[LineString],
    image_width: int,
    image_height: int,
    min_hit_ratio: float,
) -> tuple[list[list[float]] | None, dict[str, float]]:
    if (not contours) or (not transects_src):
        return None, {}

    contour_lines: list[LineString] = []
    for contour in contours:
        coords = []
        for row, col in contour:
            x, y = rasterio.transform.xy(transform, row, col)
            coords.append((float(x), float(y)))
        if len(coords) < 2:
            continue
        line = LineString(coords)
        if line.is_valid and line.length > 0:
            contour_lines.append(line)
    if not contour_lines:
        return None, {}

    hit_points_world: list[tuple[float, float] | None] = []
    for transect in transects_src:
        midpoint = transect.interpolate(0.5, normalized=True)
        best_point = None
        best_dist = float("inf")
        for line in contour_lines:
            inter = transect.intersection(line)
            points = intersection_points(inter)
            for p in points:
                d = float(p.distance(midpoint))
                if d < best_dist:
                    best_dist = d
                    best_point = p
        if best_point is None:
            hit_points_world.append(None)
        else:
            hit_points_world.append((float(best_point.x), float(best_point.y)))

    n_total = len(hit_points_world)
    n_hit = sum(1 for p in hit_points_world if p is not None)
    if n_total <= 0 or n_hit < 3:
        return None, {}

    hit_ratio = float(n_hit) / float(n_total)
    if hit_ratio < float(min_hit_ratio):
        return None, {"transect_reconstruct_hit_ratio": hit_ratio}

    xs = np.array([p[0] if p is not None else np.nan for p in hit_points_world], dtype=float)
    ys = np.array([p[1] if p is not None else np.nan for p in hit_points_world], dtype=float)
    idx = np.arange(n_total, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    if valid.sum() < 2:
        return None, {"transect_reconstruct_hit_ratio": hit_ratio}

    xs_interp = np.interp(idx, idx[valid], xs[valid])
    ys_interp = np.interp(idx, idx[valid], ys[valid])
    world_points = [(float(x), float(y)) for x, y in zip(xs_interp, ys_interp)]
    pixel_points = world_to_pixel_points(
        world_points=world_points,
        transform=transform,
        width=image_width,
        height=image_height,
    )
    if len(pixel_points) < 3:
        return None, {"transect_reconstruct_hit_ratio": hit_ratio}

    return pixel_points, {"transect_reconstruct_hit_ratio": hit_ratio, "transect_reconstruct_hits": float(n_hit)}


def dedupe_consecutive_points(points: list[list[float]], tol: float = 1e-6) -> list[list[float]]:
    if not points:
        return []
    out = [points[0]]
    for x, y in points[1:]:
        px, py = out[-1]
        if (abs(float(x) - float(px)) + abs(float(y) - float(py))) > float(tol):
            out.append([float(x), float(y)])
    return out


def smooth_polyline_points(
    points: list[list[float]],
    smooth_window: int,
    smooth_polyorder: int,
    smooth_passes: int,
) -> list[list[float]]:
    pts = dedupe_consecutive_points(points)
    if len(pts) < 5 or smooth_passes <= 0:
        return pts

    arr = np.asarray(pts, dtype=float)
    n = len(arr)
    win = max(5, int(smooth_window))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n if n % 2 == 1 else (n - 1)
    if win < 5:
        return pts

    poly = min(max(1, int(smooth_polyorder)), win - 1)
    original_start = arr[0].copy()
    original_end = arr[-1].copy()

    for _ in range(int(max(1, smooth_passes))):
        x_smooth = signal.savgol_filter(arr[:, 0], window_length=win, polyorder=poly, mode="interp")
        y_smooth = signal.savgol_filter(arr[:, 1], window_length=win, polyorder=poly, mode="interp")
        arr = np.column_stack([x_smooth, y_smooth])
        arr[0] = original_start
        arr[-1] = original_end

    return [[float(x), float(y)] for x, y in arr]


def resample_polyline_points(points: list[list[float]], target_points: int) -> list[list[float]]:
    pts = dedupe_consecutive_points(points)
    if len(pts) < 2 or int(target_points) <= 1:
        return pts

    line = LineString([(float(x), float(y)) for x, y in pts])
    if (not line.is_valid) or (line.length <= 0):
        return pts

    dists = np.linspace(0.0, float(line.length), int(target_points))
    out: list[list[float]] = []
    for d in dists:
        p = line.interpolate(float(d))
        out.append([float(p.x), float(p.y)])
    return out


def infer_target_points(
    points: list[list[float]],
    target_points: int,
    min_points: int,
    max_points: int,
    points_per_100px: float,
) -> int:
    pts = dedupe_consecutive_points(points)
    if len(pts) < 3:
        return max(3, len(pts))

    if int(target_points) > 1:
        return int(target_points)

    line = LineString([(float(x), float(y)) for x, y in pts])
    length_px = float(line.length) if line.is_valid else float(len(pts))
    density = max(float(points_per_100px) / 100.0, 0.01)
    inferred = int(round(length_px * density))

    lower = max(3, int(min_points))
    upper = int(max_points) if int(max_points) > 0 else inferred
    if upper < lower:
        upper = lower
    inferred = max(lower, inferred)
    inferred = min(upper, inferred)
    return int(inferred)


def regularize_label_points(
    points: list[list[float]],
    width: int,
    height: int,
    smooth_window: int,
    smooth_polyorder: int,
    smooth_passes: int,
    target_points: int,
    min_points: int,
    max_points: int,
    points_per_100px: float,
) -> tuple[list[list[float]], int]:
    pts = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x = float(np.clip(float(p[0]), 0.0, float(width - 1)))
            y = float(np.clip(float(p[1]), 0.0, float(height - 1)))
            if np.isfinite(x) and np.isfinite(y):
                pts.append([x, y])
    pts = dedupe_consecutive_points(pts)
    if len(pts) < 2:
        return pts, int(len(pts))

    pts = smooth_polyline_points(
        points=pts,
        smooth_window=int(smooth_window),
        smooth_polyorder=int(smooth_polyorder),
        smooth_passes=int(smooth_passes),
    )
    use_points = infer_target_points(
        points=pts,
        target_points=int(target_points),
        min_points=int(min_points),
        max_points=int(max_points),
        points_per_100px=float(points_per_100px),
    )
    if int(use_points) > 1:
        pts = resample_polyline_points(points=pts, target_points=int(use_points))

    out = []
    for x, y in pts:
        out.append(
            [
                float(np.clip(float(x), 0.0, float(width - 1))),
                float(np.clip(float(y), 0.0, float(height - 1))),
            ]
        )
    return out, int(use_points)


def write_labelme_json(
    out_json: Path,
    image_name: str,
    image_width: int,
    image_height: int,
    points: list[list[float]],
    threshold: float,
    method_desc: str,
) -> None:
    payload = {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": [
            {
                "label": "ve",
                "points": points,
                "group_id": None,
                "description": (
                    f"auto-labeled vegetation edge; method={method_desc}; "
                    f"ndvi_threshold={threshold:.6f}"
                ),
                "shape_type": "linestrip",
                "flags": {},
                "mask": None,
            }
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_overlay(
    image_path: Path,
    points: list[list[float]],
    out_path: Path,
    threshold: float,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    if len(points) >= 2:
        draw.line([tuple(p) for p in points], fill=(40, 240, 90, 255), width=2)

    draw.rectangle((6, 6, min(430, image.width - 6), 34), fill=(0, 0, 0, 180))
    draw.text((10, 12), f"ve threshold={threshold:.4f}", fill=(255, 255, 255, 255))

    merged = Image.alpha_composite(image, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path)


def maybe_scale_points(
    points: list[list[float]],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[list[float]]:
    if src_w == dst_w and src_h == dst_h:
        return points
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out = []
    for x, y in points:
        out.append([float(x * sx), float(y * sy)])
    return out


def process_single_image(
    png_path: Path,
    out_png_path: Path,
    out_json_path: Path,
    image_path_for_json: str,
    runs_dir: Path,
    model,
    max_dist_ref_m: float,
    min_line_length_m: float,
    min_patch_size_px: int,
    min_beach_area_m2: float,
    shape_mode: str,
    target_points: int,
    min_points: int,
    max_points: int,
    points_per_100px: float,
    smooth_window: int,
    smooth_polyorder: int,
    smooth_passes: int,
    transect_spacing_m: float,
    transect_length_m: float,
    transect_offshore_ratio: float,
    transect_corridor_m: float,
    min_transect_hit_ratio: float,
    transect_reconstruct_min_hit_ratio: float,
    ref_cache: dict[str, RefLineState],
    overlay_dir: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    parsed = parse_png_name(png_path)
    if parsed is None:
        return {"status": "skip", "reason": "bad_name", "png": png_path.name}

    run_id, scene_id = parsed
    aoi_key = parse_aoi_key(run_id)
    tif_path = find_tif_path(runs_dir, run_id, scene_id)
    if tif_path is None:
        return {
            "status": "skip",
            "reason": "missing_tif",
            "png": png_path.name,
            "run_id": run_id,
            "scene_id": scene_id,
        }

    with rasterio.open(tif_path) as src:
        stack = src.read([1, 2, 3, 4, 5]).astype(float)
        im_ms = np.moveaxis(stack, 0, -1)  # (H, W, bands)
        h_tif, w_tif = im_ms.shape[:2]
        src_crs = CRS.from_user_input(src.crs) if src.crs is not None else CRS.from_epsg(4326)

        cloud_mask = make_cloud_mask(im_ms)
        valid_ratio = float((~cloud_mask).mean())
        if valid_ratio < 0.2:
            return {
                "status": "skip",
                "reason": "mostly_cloud_or_nodata",
                "png": png_path.name,
                "run_id": run_id,
                "scene_id": scene_id,
                "valid_ratio": valid_ratio,
            }

        if min_patch_size_px > 0:
            min_patch_size = int(min_patch_size_px)
        else:
            # Fallback path for explicit area-based control.
            pixel_area_m2 = estimate_pixel_area_m2(src)
            min_patch_size = max(1, int(math.ceil(min_beach_area_m2 / pixel_area_m2)))

        _, veg_labels = classify_image_nn(
            im_ms=im_ms,
            cloud_mask=cloud_mask,
            model=model,
            min_patch_size=min_patch_size,
        )
        veg_class = veg_labels[:, :, 0]
        nonveg_class = veg_labels[:, :, 1]

        veg_ratio = float(veg_class.mean())
        nonveg_ratio = float(nonveg_class.mean())
        if veg_ratio < 0.01 or nonveg_ratio < 0.01:
            return {
                "status": "skip",
                "reason": "insufficient_class_separation",
                "png": png_path.name,
                "run_id": run_id,
                "scene_id": scene_id,
                "veg_ratio": veg_ratio,
                "nonveg_ratio": nonveg_ratio,
            }

        ndvi = nd_index(im_ms[:, :, 3], im_ms[:, :, 2], cloud_mask)
        ref_state = ref_cache.get(aoi_key)
        is_manual_ref = ref_state is not None and ref_state.source == "manual"
        ref_mask = build_ref_buffer_mask(src, cloud_mask, ref_state, max_dist_ref_m)
        if float(ref_mask.mean()) < 0.01:
            if is_manual_ref:
                ref_mask = build_ref_buffer_mask(src, cloud_mask, ref_state, max_dist_ref_m * 2.0)
            else:
                # Fallback to full valid area only for auto-ref mode.
                ref_mask = ~cloud_mask

        ref_line_src = None
        transect_metric_crs = None
        transects_metric: list[LineString] = []
        transects_src: list[LineString] = []
        transect_corridor_mask = None
        if ref_state is not None:
            ref_line_src = reproject_line(ref_state.line, ref_state.crs, src_crs)
            if is_manual_ref and transect_spacing_m > 0 and transect_length_m > 0:
                transect_metric_crs, transects_metric, transects_src = generate_transects_from_refline(
                    ref_line_src=ref_line_src,
                    src_crs=src_crs,
                    spacing_m=transect_spacing_m,
                    length_m=transect_length_m,
                    offshore_ratio=transect_offshore_ratio,
                )
                transect_corridor_mask = build_transect_corridor_mask(
                    src=src,
                    cloud_mask=cloud_mask,
                    transects_src=transects_src,
                    src_crs=src_crs,
                    corridor_m=transect_corridor_m,
                )

        sampling_mask = ref_mask
        if transect_corridor_mask is not None:
            sample_mask_candidate = np.logical_and(ref_mask, transect_corridor_mask)
            if float(sample_mask_candidate.mean()) >= 0.002:
                sampling_mask = sample_mask_candidate

        seed = zlib.crc32(f"{run_id}__{scene_id}".encode("utf-8")) & 0xFFFFFFFF
        int_veg, int_nonveg = clip_index_vec(
            cloud_mask=cloud_mask,
            im_ndi=ndvi,
            im_labels=veg_labels,
            im_ref_buffer=sampling_mask,
            seed=seed,
        )
        if int_veg is None or int_nonveg is None:
            if is_manual_ref:
                # Keep manual ref as hard prior; only expand buffer, no full-image fallback.
                ref_mask = build_ref_buffer_mask(src, cloud_mask, ref_state, max_dist_ref_m * 3.0)
                sampling_mask = ref_mask
                if transect_corridor_mask is not None:
                    sample_mask_candidate = np.logical_and(ref_mask, transect_corridor_mask)
                    if float(sample_mask_candidate.mean()) >= 0.002:
                        sampling_mask = sample_mask_candidate
                int_veg, int_nonveg = clip_index_vec(
                    cloud_mask=cloud_mask,
                    im_ndi=ndvi,
                    im_labels=veg_labels,
                    im_ref_buffer=sampling_mask,
                    seed=seed,
                )
            else:
                # Auto-ref mode: allow full-image fallback.
                int_veg, int_nonveg = clip_index_vec(
                    cloud_mask=cloud_mask,
                    im_ndi=ndvi,
                    im_labels=veg_labels,
                    im_ref_buffer=~cloud_mask,
                    seed=seed,
                )
                ref_mask = ~cloud_mask
        if int_veg is None or int_nonveg is None:
            return {
                "status": "skip",
                "reason": "empty_class_vectors",
                "png": png_path.name,
                "run_id": run_id,
                "scene_id": scene_id,
            }

        threshold = find_wp_threshold(int_veg, int_nonveg)
        ref_edge_dist_px = build_ref_mask_edge_distance_px(ref_mask)
        ndvi_buffer = np.copy(ndvi)
        ndvi_buffer[~ref_mask] = np.nan
        contours = measure.find_contours(ndvi_buffer, threshold)
        contours = process_contours(contours)
        if (not contours) and is_manual_ref:
            # One more retry with widened manual buffer.
            ref_mask = build_ref_buffer_mask(src, cloud_mask, ref_state, max_dist_ref_m * 3.0)
            ref_edge_dist_px = build_ref_mask_edge_distance_px(ref_mask)
            ndvi_buffer = np.copy(ndvi)
            ndvi_buffer[~ref_mask] = np.nan
            contours = process_contours(measure.find_contours(ndvi_buffer, threshold))
        if not contours:
            return {
                "status": "skip",
                "reason": "no_contours",
                "png": png_path.name,
                "run_id": run_id,
                "scene_id": scene_id,
                "threshold": threshold,
            }

        points: list[list[float]] = []
        primary_meta: dict[str, float] = {}
        primary_line: LineString | None = None
        recon_meta: dict[str, float] = {}

        # Prefer full-line reconstruction from transect intersections when manual ref exists.
        if is_manual_ref and transects_src:
            recon_points, recon_meta = reconstruct_line_from_transects(
                contours=contours,
                transform=src.transform,
                transects_src=transects_src,
                image_width=w_tif,
                image_height=h_tif,
                min_hit_ratio=float(transect_reconstruct_min_hit_ratio),
            )
            if recon_points is not None and len(recon_points) >= 3:
                world_coords = []
                for col, row in recon_points:
                    xw, yw = rasterio.transform.xy(src.transform, float(row), float(col))
                    world_coords.append((float(xw), float(yw)))
                recon_line = LineString(world_coords)
                if recon_line.is_valid and recon_line.length > 0:
                    points = recon_points
                    primary_line = recon_line
                    primary_meta.update(recon_meta)
                    primary_meta["used_transect_reconstruct"] = 1.0

        if primary_line is None:
            min_len_units = min_line_length_units(src, min_line_length_m)
            primary_contour, primary_line, primary_meta = primary_contour_from_candidates(
                contours=contours,
                transform=src.transform,
                src_crs=src_crs,
                min_len_units=min_len_units,
                image_height=h_tif,
                image_width=w_tif,
                ref_line_src_crs=ref_line_src,
                metric_crs=transect_metric_crs,
                transects_metric=transects_metric,
                transect_length_m=float(transect_length_m),
                min_transect_hit_ratio=float(min_transect_hit_ratio) if is_manual_ref else 0.0,
                ref_mask_edge_dist_px=ref_edge_dist_px,
                edge_proximity_px=1.5,
            )
            if primary_contour is None or primary_line is None:
                return {
                    "status": "skip",
                    "reason": "no_primary_line",
                    "png": png_path.name,
                    "run_id": run_id,
                    "scene_id": scene_id,
                    "threshold": threshold,
                    "transects": len(transects_metric),
                    "transect_reconstruct_hit_ratio": float(
                        recon_meta.get("transect_reconstruct_hit_ratio", float("nan"))
                    ),
                }

            if shape_mode == "contour":
                points = contour_to_raw_points(primary_contour, width=w_tif, height=h_tif)
            else:
                points = contour_to_profile_points(primary_contour, width=w_tif, height=h_tif)
            if len(points) < 3:
                return {
                    "status": "skip",
                    "reason": "too_few_points",
                    "png": png_path.name,
                    "run_id": run_id,
                    "scene_id": scene_id,
                    "threshold": threshold,
                }

        # Match VedgeSat behavior: keep one fixed reference line, do not drift every scene.
        if aoi_key not in ref_cache:
            ref_cache[aoi_key] = RefLineState(line=primary_line, crs=src_crs, source="auto")

    with Image.open(png_path) as img:
        w_png, h_png = img.size
    points_scaled = maybe_scale_points(points, src_w=w_tif, src_h=h_tif, dst_w=w_png, dst_h=h_png)
    points_scaled, used_points = regularize_label_points(
        points=points_scaled,
        width=w_png,
        height=h_png,
        smooth_window=int(smooth_window),
        smooth_polyorder=int(smooth_polyorder),
        smooth_passes=int(smooth_passes),
        target_points=int(target_points),
        min_points=int(min_points),
        max_points=int(max_points),
        points_per_100px=float(points_per_100px),
    )
    if len(points_scaled) < 3:
        return {
            "status": "skip",
            "reason": "too_few_points_after_regularize",
            "png": png_path.name,
            "run_id": run_id,
            "scene_id": scene_id,
            "threshold": threshold,
        }

    if not dry_run:
        method_desc = "coastguard_weighted_peaks_ndvi"
        if is_manual_ref and len(transects_metric) > 0:
            method_desc = "coastguard_weighted_peaks_ndvi_manual_ref_transects"
        if float(primary_meta.get("used_transect_reconstruct", 0.0)) > 0.5:
            method_desc = f"{method_desc}_reconstruct"
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        write_labelme_json(
            out_json=out_json_path,
            image_name=image_path_for_json,
            image_width=w_png,
            image_height=h_png,
            points=points_scaled,
            threshold=float(threshold),
            method_desc=method_desc,
        )
        if overlay_dir is not None:
            overlay_path = overlay_dir / f"{png_path.stem}_overlay.png"
            draw_overlay(
                image_path=out_png_path,
                points=points_scaled,
                out_path=overlay_path,
                threshold=float(threshold),
            )

    return {
        "status": "ok",
        "reason": "",
        "png": png_path.name,
        "run_id": run_id,
        "scene_id": scene_id,
        "tif": str(tif_path),
        "threshold": float(threshold),
        "line_length": float(primary_line.length),
        "points": len(points_scaled),
        "transects": len(transects_metric),
        "transect_hit_ratio": float(primary_meta.get("transect_hit_ratio", float("nan"))),
        "transect_hits": float(primary_meta.get("transect_hits", float("nan"))),
        "ref_dist": float(primary_meta.get("ref_dist", float("nan"))),
        "ref_cover_ratio": float(primary_meta.get("ref_cover_ratio", float("nan"))),
        "score": float(primary_meta.get("score", float("nan"))),
        "used_transect_reconstruct": float(primary_meta.get("used_transect_reconstruct", 0.0)),
        "transect_reconstruct_hit_ratio": float(primary_meta.get("transect_reconstruct_hit_ratio", float("nan"))),
        "target_points": int(used_points),
    }


def main() -> int:
    args = parse_args()
    png_dir = args.png_dir.resolve()
    runs_dir = args.runs_dir.resolve()
    out_dir = (args.output_dir or args.png_dir).resolve()
    manual_ref_dir = args.manual_ref_dir.resolve() if args.manual_ref_dir else None
    if manual_ref_dir is None:
        auto_ref_dir = PROJECT_ROOT / "data" / "labelme_work" / "refline_see"
        if auto_ref_dir.exists():
            manual_ref_dir = auto_ref_dir.resolve()
    overlay_dir = args.overlay_dir.resolve() if args.overlay_dir else None
    report_csv = args.report_csv.resolve() if args.report_csv else None

    if not png_dir.exists():
        print(f"[ERROR] png dir not found: {png_dir}")
        return 1
    if not runs_dir.exists():
        print(f"[ERROR] runs dir not found: {runs_dir}")
        return 1

    model_path = resolve_model_path(args.model_path)
    model = load_classifier(model_path) if model_path is not None else None

    png_files = sorted([p for p in png_dir.glob("*.png") if p.is_file()])
    if not png_files:
        print(f"[ERROR] no png files found: {png_dir}")
        return 1

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if overlay_dir is not None:
            overlay_dir.mkdir(parents=True, exist_ok=True)

    print("COASTGUARD VE auto-label")
    print(f"Input PNG dir : {png_dir}")
    print(f"Runs dir      : {runs_dir}")
    print(f"Output dir    : {out_dir if not args.dry_run else '[dry-run]'}")
    print(f"Manual ref dir: {manual_ref_dir if manual_ref_dir else '[disabled]'}")
    print(f"Overlay dir   : {overlay_dir if overlay_dir else '[disabled]'}")
    print(f"Model         : {model_path if model_path else '[fallback NDVI classifier]'}")
    print(f"Min patch px  : {int(args.min_patch_size_px)}")
    if int(args.target_points) > 1:
        line_points_mode = f"fixed={int(args.target_points)}"
    else:
        line_points_mode = (
            f"adaptive(min={int(args.min_points)},max={int(args.max_points)},"
            f"density={float(args.points_per_100px):.1f}/100px)"
        )
    print(
        "Line regular. : "
        f"{line_points_mode}, "
        f"smooth_window={int(args.smooth_window)}, "
        f"polyorder={int(args.smooth_polyorder)}, "
        f"passes={int(args.smooth_passes)}"
    )
    print(
        "Transects     : "
        f"spacing={float(args.transect_spacing_m):.1f}m, "
        f"length={float(args.transect_length_m):.1f}m, "
        f"offshore_ratio={float(args.transect_offshore_ratio):.2f}, "
        f"corridor={float(args.transect_corridor_m):.1f}m, "
        f"min_hit={float(args.min_transect_hit_ratio):.2f}, "
        f"reconstruct_min_hit={float(args.transect_reconstruct_min_hit_ratio):.2f}"
    )
    print(f"Images        : {len(png_files)}")
    print("-" * 72)

    ref_cache: dict[str, RefLineState] = {}
    if manual_ref_dir is not None:
        manual_refs = load_manual_reference_lines(
            manual_ref_dir=manual_ref_dir,
            png_dir=png_dir,
            runs_dir=runs_dir,
            manual_ref_label=args.manual_ref_label,
        )
        ref_cache.update(manual_refs)
        print(f"Loaded manual AOI ref lines: {len(manual_refs)}")
        if manual_refs:
            print("AOIs with manual ref:", ", ".join(sorted(manual_refs.keys())))
        print("-" * 72)

    records: list[dict[str, Any]] = []

    for idx, png_path in enumerate(png_files, start=1):
        out_png_path = out_dir / png_path.name
        out_json_path = out_dir / f"{png_path.stem}.json"
        image_path_for_json = out_png_path.name

        if not args.dry_run and out_dir != png_dir and args.copy_images:
            shutil.copy2(png_path, out_png_path)
        else:
            out_png_path = png_path
            if out_dir != png_dir:
                image_path_for_json = os.path.relpath(png_path, out_dir).replace("\\", "/")

        result = process_single_image(
            png_path=png_path,
            out_png_path=out_png_path,
            out_json_path=out_json_path,
            image_path_for_json=image_path_for_json,
            runs_dir=runs_dir,
            model=model,
            max_dist_ref_m=float(args.max_dist_ref_m),
            min_line_length_m=float(args.min_line_length_m),
            min_patch_size_px=int(args.min_patch_size_px),
            min_beach_area_m2=float(args.min_beach_area_m2),
            shape_mode=args.shape_mode,
            target_points=int(args.target_points),
            min_points=int(args.min_points),
            max_points=int(args.max_points),
            points_per_100px=float(args.points_per_100px),
            smooth_window=int(args.smooth_window),
            smooth_polyorder=int(args.smooth_polyorder),
            smooth_passes=int(args.smooth_passes),
            transect_spacing_m=float(args.transect_spacing_m),
            transect_length_m=float(args.transect_length_m),
            transect_offshore_ratio=float(args.transect_offshore_ratio),
            transect_corridor_m=float(args.transect_corridor_m),
            min_transect_hit_ratio=float(args.min_transect_hit_ratio),
            transect_reconstruct_min_hit_ratio=float(args.transect_reconstruct_min_hit_ratio),
            ref_cache=ref_cache,
            overlay_dir=overlay_dir,
            dry_run=bool(args.dry_run),
        )
        records.append(result)

        if result["status"] == "ok":
            msg = (
                f"[{idx:4d}/{len(png_files)}] OK   {png_path.name} | "
                f"thr={result.get('threshold', float('nan')):.4f} | "
                f"pts={result.get('points', 0)}"
            )
        else:
            msg = f"[{idx:4d}/{len(png_files)}] SKIP {png_path.name} | {result.get('reason', 'unknown')}"
        print(msg)

    ok_count = sum(1 for r in records if r.get("status") == "ok")
    skip_count = len(records) - ok_count

    reason_counts: dict[str, int] = {}
    for r in records:
        if r.get("status") != "ok":
            reason = str(r.get("reason", "unknown"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    if report_csv is not None and not args.dry_run:
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for rec in records for k in rec.keys()})
        with report_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    print("-" * 72)
    print(f"Labeled : {ok_count}")
    print(f"Skipped : {skip_count}")
    if reason_counts:
        print("Skip breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {reason}: {count}")
    if report_csv is not None and not args.dry_run:
        print(f"Report  : {report_csv}")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
