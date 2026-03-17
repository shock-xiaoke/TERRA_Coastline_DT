"""Download Sentinel-2 scenes for one AOI and optionally build fixed-size chips.

This script reuses project services:
- scene search: src.terra_ugla.services.imagery.search_scenes
- scene download: src.terra_ugla.services.imagery.download_scene_multiband_tiff

Outputs:
- data/runs/<run_id>/imagery/*.tif
- data/runs/<run_id>/preview/*.png
- data/runs/<run_id>/labelme_chips/<scene_id>/*.png (optional)
- data/runs/<run_id>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window, bounds, transform

# Allow running as: python scripts/pull_sentinelhub_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.terra_ugla.config import initialize_sentinel_hub_config
from src.terra_ugla.services.imagery import download_scene_multiband_tiff, search_scenes


def _safe_scene_id(scene_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", scene_id).strip("_")
    return cleaned or "scene"


def _stretch_to_uint8(rgb: np.ndarray, lower_pct: float, upper_pct: float) -> np.ndarray:
    """Convert 3xHxW float array into HxWx3 uint8 image with percentile stretch."""
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected RGB array shape (3,H,W), got {rgb.shape}")

    out = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype=np.uint8)
    for c in range(3):
        channel = rgb[c, :, :]
        valid = np.isfinite(channel)
        if not np.any(valid):
            continue

        low = float(np.percentile(channel[valid], lower_pct))
        high = float(np.percentile(channel[valid], upper_pct))
        if high <= low:
            high = low + 1e-6

        scaled = (channel - low) / (high - low)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled[~valid] = 0.0
        out[:, :, c] = (scaled * 255.0).astype(np.uint8)

    return out


def _save_preview_png(
    tif_path: Path,
    out_png: Path,
    lower_pct: float,
    upper_pct: float,
) -> dict[str, Any]:
    """Build quick-look true-color PNG preview from B04/B03/B02 bands."""
    with rasterio.open(tif_path) as src:
        rgb = src.read([3, 2, 1]).astype(np.float32)
        if src.nodata is not None:
            rgb[rgb == src.nodata] = np.nan
        preview = _stretch_to_uint8(rgb, lower_pct=lower_pct, upper_pct=upper_pct)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(out_png)
    return {"preview_path": str(out_png), "size": [int(preview.shape[1]), int(preview.shape[0])]}


def _window_grid(width: int, height: int, chip_size: int, stride: int) -> list[Window]:
    windows: list[Window] = []
    for row_off in range(0, height, stride):
        for col_off in range(0, width, stride):
            w = min(chip_size, width - col_off)
            h = min(chip_size, height - row_off)
            windows.append(Window(col_off=col_off, row_off=row_off, width=w, height=h))
    return windows


def _save_chips_from_scene(
    tif_path: Path,
    scene_id: str,
    chips_dir: Path,
    chip_size: int,
    stride: int,
    lower_pct: float,
    upper_pct: float,
) -> list[dict[str, Any]]:
    chips_dir.mkdir(parents=True, exist_ok=True)
    scene_safe = _safe_scene_id(scene_id)
    chip_records: list[dict[str, Any]] = []

    with rasterio.open(tif_path) as src:
        windows = _window_grid(src.width, src.height, chip_size=chip_size, stride=stride)
        for idx, win in enumerate(windows):
            rgb = src.read([3, 2, 1], window=win).astype(np.float32)
            if src.nodata is not None:
                rgb[rgb == src.nodata] = np.nan

            padded = np.full((3, chip_size, chip_size), np.nan, dtype=np.float32)
            h = int(win.height)
            w = int(win.width)
            padded[:, :h, :w] = rgb

            image = _stretch_to_uint8(padded, lower_pct=lower_pct, upper_pct=upper_pct)
            chip_name = f"{scene_safe}_chip_{idx:04d}.png"
            chip_path = chips_dir / chip_name
            Image.fromarray(image).save(chip_path)

            geo_t = transform(win, src.transform)
            left, bottom, right, top = bounds(win, src.transform)
            chip_records.append(
                {
                    "chip_path": str(chip_path),
                    "source_tif": str(tif_path),
                    "scene_id": scene_id,
                    "window": {
                        "col_off": int(win.col_off),
                        "row_off": int(win.row_off),
                        "width": int(win.width),
                        "height": int(win.height),
                    },
                    "chip_size": chip_size,
                    "actual_pixels": [int(w), int(h)],
                    "crs": str(src.crs) if src.crs else None,
                    "bounds_wgs84_or_native": [float(left), float(bottom), float(right), float(top)],
                    "transform": [float(v) for v in geo_t.to_gdal()],
                }
            )

    return chip_records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 scenes from Sentinel Hub/CDSE and prepare fixed-size chips."
    )
    parser.add_argument("--aoi-id", required=True, help="AOI identifier used in cache/manifests.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        required=True,
        help="AOI bounding box in WGS84 coordinates.",
    )
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--max-cloud-pct", type=int, default=30, help="Maximum scene cloud percentage.")
    parser.add_argument("--max-images", type=int, default=5, help="Maximum number of scenes to download.")
    parser.add_argument("--run-id", default=None, help="Optional run id. Defaults to utc timestamp.")
    parser.add_argument(
        "--build-chips",
        action="store_true",
        help="If set, build fixed-size true-color PNG chips for annotation/training.",
    )
    parser.add_argument("--chip-size", type=int, default=512, help="Output chip size in pixels.")
    parser.add_argument("--stride", type=int, default=512, help="Sliding stride for chip extraction.")
    parser.add_argument(
        "--lower-pct",
        type=float,
        default=2.0,
        help="Lower percentile for reflectance stretch when creating PNGs.",
    )
    parser.add_argument(
        "--upper-pct",
        type=float,
        default=98.0,
        help="Upper percentile for reflectance stretch when creating PNGs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.chip_size <= 0:
        raise ValueError("--chip-size must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if not (0.0 <= args.lower_pct < args.upper_pct <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

    run_id = args.run_id or f"{args.aoi_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bbox = [float(v) for v in args.bbox]

    _, sentinel_hub_available = initialize_sentinel_hub_config()
    scenes = search_scenes(
        aoi_id=args.aoi_id,
        bbox_wgs84=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        max_cloud_pct=int(args.max_cloud_pct),
        max_images=int(args.max_images),
        sentinel_hub_available=sentinel_hub_available,
    )
    if not scenes:
        print("No scenes found for the query.")
        return 1

    run_root = Path("data") / "runs" / run_id
    preview_dir = run_root / "preview"
    chips_root = run_root / "labelme_chips"

    scene_records: list[dict[str, Any]] = []
    for scene in scenes:
        scene_meta = download_scene_multiband_tiff(
            run_id=run_id,
            aoi_bbox_wgs84=bbox,
            scene=scene,
            sentinel_hub_available=sentinel_hub_available,
        )

        tif_path = Path(scene_meta["filepath"])
        scene_id = scene_meta.get("scene_id", tif_path.stem)
        scene_safe = _safe_scene_id(scene_id)

        preview_path = preview_dir / f"{scene_safe}.png"
        preview_meta = _save_preview_png(
            tif_path=tif_path,
            out_png=preview_path,
            lower_pct=float(args.lower_pct),
            upper_pct=float(args.upper_pct),
        )

        chips_meta: list[dict[str, Any]] = []
        if args.build_chips:
            chips_meta = _save_chips_from_scene(
                tif_path=tif_path,
                scene_id=scene_id,
                chips_dir=chips_root / scene_safe,
                chip_size=int(args.chip_size),
                stride=int(args.stride),
                lower_pct=float(args.lower_pct),
                upper_pct=float(args.upper_pct),
            )

        scene_records.append(
            {
                "scene": scene,
                "download": scene_meta,
                "preview": preview_meta,
                "chips_count": len(chips_meta),
                "chips": chips_meta,
            }
        )

    manifest = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "aoi_id": args.aoi_id,
        "bbox_wgs84": bbox,
        "query": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "max_cloud_pct": args.max_cloud_pct,
            "max_images": args.max_images,
            "build_chips": args.build_chips,
            "chip_size": args.chip_size,
            "stride": args.stride,
        },
        "sentinel_hub_available": bool(sentinel_hub_available),
        "scene_count": len(scene_records),
        "scenes": scene_records,
    }

    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Run complete: {run_id}")
    print(f"Scenes downloaded: {len(scene_records)}")
    print(f"Manifest: {manifest_path}")
    if args.build_chips:
        total_chips = sum(s["chips_count"] for s in scene_records)
        print(f"Fixed-size chips generated: {total_chips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
