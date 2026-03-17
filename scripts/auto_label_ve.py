#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_label_ve.py
================
Automatically label vegetation edges (ve) from Sentinel-2 run directories
using a per-column NDVI gradient approach.

For each run directory the script:
  1. Loads the 5-band TIF (B02, B03, B04, B08, B11).
  2. Applies quality gates (cloud %, NDVI variance).
     → Silently skips images that are too noisy or uniform to be useful.
  3. Computes NDVI and smooths it vertically with a Gaussian filter.
  4. For every column x, finds the row y with the steepest NDVI gradient
     — that is the sharpest vegetation / non-vegetation transition.
  5. Interpolates over cloud-masked columns, then smooths the full-width
     row profile with Savitzky-Golay.
  6. Writes a LabelMe-compatible JSON containing:
       • "ve" linestrip – consumed by LabelMeCoastlineLoader in main.py
  7. Copies the preview PNG alongside the JSON.

Usage
-----
  python scripts/auto_label_ve.py                          # all AOIs
  python scripts/auto_label_ve.py --aoi aoi_01 aoi_02     # subset
  python scripts/auto_label_ve.py --dry-run                # stats only
  python scripts/auto_label_ve.py --output-dir data/labelme_work/auto_labeled
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# ── make src/ importable ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from terra_ugla.coastguard_port.indices import nd_index

# ── default paths ─────────────────────────────────────────────────────────────
_RUNS_DIR   = _PROJECT_ROOT / "data" / "runs"
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "labelme_work" / "auto_labeled"

# ── quality-gate thresholds ───────────────────────────────────────────────────
MAX_CLOUD_PCT  = 70.0   # skip if manifest cloud_pct > this
MIN_NDVI_RANGE = 0.12   # skip if (p90 − p10) NDVI < this → flat/uniform image
MIN_VALID_COLS = 0.40   # skip if < this fraction of columns have a reliable ve

# ── gradient / smoothing parameters ──────────────────────────────────────────
NDVI_SMOOTH_SIGMA = 2.0   # Gaussian sigma (rows) applied to NDVI before gradient
SMOOTH_WINDOW     = 21    # Savitzky-Golay window (must be odd)
SMOOTH_POLY       = 3     # Savitzky-Golay polynomial order

LABELME_VERSION = "5.11.4"


# ─────────────────────────────────────────────────────────────────────────────
# TIF loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tif(tif_path: Path) -> np.ndarray:
    """
    Load a 5-band Sentinel-2 L2A GeoTIFF.

    Returns
    -------
    im_ms : np.ndarray, shape (H, W, 5), float32
        Bands in order: B02(Blue), B03(Green), B04(Red), B08(NIR), B11(SWIR).
        Values are in [0, 1] reflectance range.
    """
    import rasterio
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)  # (5, H, W)
    return np.transpose(data, (1, 2, 0))      # → (H, W, 5)


def build_cloud_mask(im_ms: np.ndarray) -> np.ndarray:
    """
    Boolean mask of pixels likely to be no-data or cloud-obscured.

    Heuristic: green band (B03) near zero indicates fill / no-data.
    Also flags any pixel with a NaN in any band.
    """
    near_zero = im_ms[:, :, 1] < 1e-4          # green ≈ 0
    has_nan   = np.any(np.isnan(im_ms), axis=2)
    return near_zero | has_nan


# ─────────────────────────────────────────────────────────────────────────────
# Per-column NDVI gradient  →  vegetation edge row
# ─────────────────────────────────────────────────────────────────────────────

def find_ve_by_column_gradient(
    ndvi: np.ndarray,
    cloud_mask: np.ndarray,
    smooth_sigma: float = NDVI_SMOOTH_SIGMA,
    min_col_valid: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    Find the vegetation-edge row for every image column.

    Algorithm
    ---------
    1. Fill cloud / NaN pixels per column via linear interpolation.
    2. Smooth NDVI vertically with a Gaussian (reduces pixel-level noise
       without blurring the horizontal edge position).
    3. Compute |∂NDVI/∂row| for each column.
    4. The row with the peak gradient = the sharpest veg/non-veg transition.

    Returns
    -------
    ve_rows : np.ndarray, shape (W,)
        Fractional row index of the vegetation edge per column.
        NaN for columns where > (1 − min_col_valid) pixels are masked.
    valid_col_frac : float
        Fraction of columns that produced a reliable estimate.
    """
    H, W = ndvi.shape

    # ── per-column validity fraction ──────────────────────────────────────────
    col_valid_frac = (~cloud_mask).mean(axis=0)   # shape (W,)

    # ── fill masked / NaN pixels via column-wise linear interpolation ─────────
    ndvi_filled = ndvi.copy().astype(float)
    ys = np.arange(H, dtype=float)

    for x in range(W):
        col = ndvi_filled[:, x]
        bad = cloud_mask[:, x] | ~np.isfinite(col)
        if not bad.any():
            continue
        good = ~bad
        if good.sum() >= 2:
            col[bad] = np.interp(ys[bad], ys[good], col[good])
        elif good.sum() == 1:
            col[bad] = col[good][0]
        else:
            col[:] = 0.0   # fully masked — gradient will be zero everywhere

    # ── smooth vertically to suppress pixel noise ─────────────────────────────
    ndvi_smooth = gaussian_filter1d(ndvi_filled, sigma=smooth_sigma, axis=0)

    # ── gradient magnitude along rows: shape (H-1, W) ────────────────────────
    grad = np.abs(np.diff(ndvi_smooth, axis=0))

    # ── peak-gradient row per column ──────────────────────────────────────────
    # Add 0.5 because np.diff gives the gradient *between* rows i and i+1.
    ve_rows = np.argmax(grad, axis=0).astype(float) + 0.5   # shape (W,)

    # ── mark unreliable columns as NaN ───────────────────────────────────────
    ve_rows[col_valid_frac < min_col_valid] = np.nan

    valid_col_frac = float((col_valid_frac >= min_col_valid).mean())
    return ve_rows, valid_col_frac


# ─────────────────────────────────────────────────────────────────────────────
# Profile smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smooth_row_profile(
    rows: np.ndarray,
    window: int = SMOOTH_WINDOW,
    poly:   int = SMOOTH_POLY,
) -> np.ndarray:
    """
    Smooth a 1-D row-value profile with Savitzky-Golay.

    Falls back to a Gaussian when there are too few points for the requested
    S-G window.
    """
    n = len(rows)
    if n < 5:
        return rows.astype(float)

    w = min(window, n - (0 if n % 2 == 1 else 1))
    w = w if w % 2 == 1 else w - 1
    w = max(w, 3)
    p = min(poly, w - 1)

    if n < w:
        return gaussian_filter1d(rows.astype(float), sigma=max(1.0, n / 8.0))

    return savgol_filter(rows.astype(float), w, p)


# ─────────────────────────────────────────────────────────────────────────────
# LabelMe JSON writer
# ─────────────────────────────────────────────────────────────────────────────

def write_labelme_json(
    out_path: Path,
    image_name: str,
    W: int,
    H: int,
    ve_points: list,
) -> None:
    """
    Write a LabelMe-format JSON with a single "ve" linestrip shape.
    """
    data = {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": [
            {
                "label": "ve",
                "points": ve_points,
                "group_id": None,
                "description": "auto-labeled vegetation edge (per-column NDVI gradient, S-G smoothed)",
                "shape_type": "linestrip",
                "flags": {},
                "mask": None,
            },
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Per-run pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_run(
    run_dir: Path,
    output_dir: Optional[Path],
    max_cloud: float = MAX_CLOUD_PCT,
) -> str:
    """
    Full labeling pipeline for one run directory.

    Returns a short status string: "ok: ..." or "skip:<reason>=<value>".
    When output_dir is None (dry-run mode) nothing is written to disk.
    """
    # ── manifest ──────────────────────────────────────────────────────────────
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return "skip:no_manifest"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    scenes = manifest.get("scenes", [])
    if not scenes:
        return "skip:no_scenes"

    scene        = scenes[0]
    download     = scene.get("download", {})
    preview_info = scene.get("preview", {})

    # ── cloud gate ────────────────────────────────────────────────────────────
    cloud_pct = float(download.get("cloud_pct", 0.0))
    if cloud_pct > max_cloud:
        return f"skip:cloud={cloud_pct:.0f}%"

    # ── find TIF ──────────────────────────────────────────────────────────────
    tif_rel  = download.get("filepath", "").replace("\\", "/")
    tif_path = (_PROJECT_ROOT / tif_rel) if tif_rel else None
    if not tif_path or not tif_path.exists():
        candidates = (
            list((run_dir / "imagery").glob("*.tif")) +
            list(run_dir.glob("*.tif"))
        )
        if not candidates:
            return "skip:no_tif"
        tif_path = candidates[0]

    # ── find preview PNG ──────────────────────────────────────────────────────
    prev_rel  = preview_info.get("preview_path", "").replace("\\", "/")
    prev_path = (_PROJECT_ROOT / prev_rel) if prev_rel else None
    if not prev_path or not prev_path.exists():
        candidates = (
            list((run_dir / "preview").glob("*.png")) +
            list(run_dir.glob("*.png"))
        )
        if not candidates:
            return "skip:no_preview"
        prev_path = candidates[0]

    # ── load imagery ──────────────────────────────────────────────────────────
    try:
        im_ms = load_tif(tif_path)
    except Exception as exc:
        return f"skip:tif_error={exc}"

    H, W       = im_ms.shape[:2]
    cloud_mask = build_cloud_mask(im_ms)
    if (~cloud_mask).mean() < 0.30:
        return f"skip:mostly_masked={( ~cloud_mask).mean():.1%}"

    # ── NDVI range gate: skip featureless / fully cloudy images ──────────────
    ndvi = nd_index(im_ms[:, :, 3], im_ms[:, :, 2], cloud_mask)
    valid_ndvi = ndvi[np.isfinite(ndvi)]
    if len(valid_ndvi) < 100:
        return "skip:no_valid_ndvi"
    ndvi_range = float(np.percentile(valid_ndvi, 90) - np.percentile(valid_ndvi, 10))
    if ndvi_range < MIN_NDVI_RANGE:
        return f"skip:ndvi_range={ndvi_range:.3f}"

    # ── per-column gradient → raw ve row per column ───────────────────────────
    ve_rows_raw, valid_col_frac = find_ve_by_column_gradient(ndvi, cloud_mask)

    if valid_col_frac < MIN_VALID_COLS:
        return f"skip:sparse_cols={valid_col_frac:.1%}"

    # ── interpolate any remaining NaN columns ─────────────────────────────────
    xs    = np.arange(W, dtype=float)
    valid = np.isfinite(ve_rows_raw)
    if valid.sum() < 5:
        return "skip:too_few_valid_cols"

    ve_rows = ve_rows_raw.copy()
    if (~valid).any():
        ve_rows[~valid] = np.interp(xs[~valid], xs[valid], ve_rows_raw[valid])

    # ── smooth the full-width row profile ─────────────────────────────────────
    ve_rows = smooth_row_profile(ve_rows)
    ve_rows = np.clip(ve_rows, 0.0, H - 1.0)

    # ── build ve linestrip points (one per column, full image width) ──────────
    ve_points = [[float(x), float(y)] for x, y in enumerate(ve_rows)]

    # ── write outputs ─────────────────────────────────────────────────────────
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        run_id   = manifest.get("run_id", run_dir.name)
        scene_id = download.get("scene_id", tif_path.stem)
        out_stem = f"{run_id}__{scene_id}"

        out_png  = output_dir / f"{out_stem}.png"
        out_json = output_dir / f"{out_stem}.json"

        shutil.copy2(prev_path, out_png)
        write_labelme_json(out_json, out_png.name, W, H, ve_points)

    return (
        f"ok: ndvi_range={ndvi_range:.3f}, valid_cols={valid_col_frac:.1%}, "
        f"y=[{ve_rows.min():.0f},{ve_rows.max():.0f}], pts={len(ve_points)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=_RUNS_DIR,
        help="Root directory containing per-run sub-folders",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_OUTPUT_DIR,
        help="Destination for labeled JSON + PNG pairs",
    )
    parser.add_argument(
        "--max-cloud", type=float, default=MAX_CLOUD_PCT,
        help="Skip runs with cloud_pct above this value",
    )
    parser.add_argument(
        "--aoi", nargs="*", metavar="PREFIX",
        help="Only process run dirs starting with these prefixes, e.g. aoi_01 aoi_02",
    )
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="Only process runs with YYYYMM >= start year (e.g. 2020).",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="Only process runs with YYYYMM <= end year (e.g. 2026).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run the full pipeline but write no files",
    )
    args = parser.parse_args()

    if args.start_year is not None and args.end_year is not None and args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")

    out_dir  = None if args.dry_run else args.output_dir
    run_dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir())
    if args.aoi:
        run_dirs = [d for d in run_dirs if any(d.name.startswith(a) for a in args.aoi)]
    if args.start_year is not None or args.end_year is not None:
        ym_pattern = re.compile(r"^aoi_\d+_(\d{4})(\d{2})$")
        filtered = []
        for d in run_dirs:
            match = ym_pattern.match(d.name)
            if match is None:
                continue
            year = int(match.group(1))
            if args.start_year is not None and year < args.start_year:
                continue
            if args.end_year is not None and year > args.end_year:
                continue
            filtered.append(d)
        run_dirs = filtered

    print(f"Found {len(run_dirs)} run directories")
    print(f"Output : {out_dir or '[dry-run, no writes]'}")
    print("-" * 60)

    n_ok   = 0
    n_skip = 0
    skip_reasons: dict[str, int] = {}

    for i, run_dir in enumerate(run_dirs, 1):
        status = process_run(run_dir, out_dir, max_cloud=args.max_cloud)
        tag    = status.split(":")[0]

        if tag == "ok":
            n_ok += 1
            if n_ok <= 5 or i % 100 == 0:
                print(f"[{i:5d}/{len(run_dirs)}] {run_dir.name}: {status}")
        else:
            n_skip += 1
            reason = status.split(":")[1].split("=")[0] if ":" in status else "unknown"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    print()
    print("=" * 60)
    print(f"  Labeled : {n_ok}")
    print(f"  Skipped : {n_skip}")
    if skip_reasons:
        print("  Skip breakdown:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:35s}: {count}")
    if out_dir:
        print(f"  Output  : {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
