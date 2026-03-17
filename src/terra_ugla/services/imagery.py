"""Sentinel Hub scene search and raw multiband download services."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.transform import from_bounds

from ..api.sentinel_hub import get_cdse_access_token
from ..config import load_sentinel_hub_config

SCENE_CACHE_DIR = Path("data") / "aoi"
RUNS_DIR = Path("data") / "runs"


class SceneLookupError(Exception):
    pass


def _scene_cache_path(aoi_id: str) -> Path:
    SCENE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SCENE_CACHE_DIR / f"{aoi_id}_scenes.json"


def _save_scene_cache(aoi_id: str, payload: dict[str, Any]) -> None:
    path = _scene_cache_path(aoi_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_scene_cache(aoi_id: str) -> dict[str, Any]:
    path = _scene_cache_path(aoi_id)
    if not path.exists():
        return {"scenes": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_parse_iso(value: str) -> datetime:
    """Parse ISO-like datetime strings from CDSE robustly across Python versions."""
    raw = (value or "").strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    # Normalise fractional seconds length so fromisoformat can parse reliably.
    match = re.match(r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?P<frac>\.\d+)?(?P<tz>[+-]\d{2}:\d{2})?$", raw)
    if match:
        head = match.group("head")
        frac = match.group("frac") or ""
        tz = match.group("tz") or "+00:00"
        if frac:
            digits = frac[1:]
            digits = (digits + "000000")[:6]
            frac = "." + digits
        raw = f"{head}{frac}{tz}"

    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _search_scenes_cdse(
    bbox_wgs84: list[float],
    start_date: str,
    end_date: str,
    max_cloud_pct: int,
    max_images: int,
) -> list[dict[str, Any]]:
    conf = load_sentinel_hub_config()
    try:
        token = get_cdse_access_token(conf["client_id"], conf["client_secret"])
    except Exception as exc:
        raise RuntimeError(f"Unable to acquire CDSE token for catalog search: {exc}") from exc

    url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    payload = {
        "bbox": bbox_wgs84,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "collections": ["sentinel-2-l2a"],
        # CDSE catalog endpoint rejects STAC "query"/"sortby" keys in this mode.
        # Request a broader set, then filter/sort client-side.
        "limit": max(max_images * 8, 40),
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    scenes: list[dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties", {})
        dt = props.get("datetime")
        cloud = float(props.get("eo:cloud_cover", 100.0))
        if not dt:
            continue
        if cloud > float(max_cloud_pct):
            continue
        scenes.append(
            {
                "scene_id": feature.get("id", f"scene_{len(scenes)}"),
                "datetime": dt,
                "cloud_pct": cloud,
                "collection": feature.get("collection", "sentinel-2-l2a"),
                "bbox": feature.get("bbox", bbox_wgs84),
                "properties": {
                    "platform": props.get("platform"),
                    "constellation": props.get("constellation"),
                },
            }
        )

    scenes.sort(key=lambda s: _try_parse_iso(s["datetime"]), reverse=True)
    return scenes[:max_images]


def _search_scenes_demo(start_date: str, end_date: str, max_images: int) -> list[dict[str, Any]]:
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    if end <= start:
        end = start + timedelta(days=1)

    span_days = max((end - start).days, 1)
    step = max(span_days // max(max_images, 1), 1)

    scenes: list[dict[str, Any]] = []
    cur = end
    for i in range(max_images):
        if cur < start:
            break
        scenes.append(
            {
                "scene_id": f"demo_scene_{cur.strftime('%Y%m%d')}_{i}",
                "datetime": cur.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
                "cloud_pct": float(min(90, 5 + (i * 7))),
                "collection": "demo",
                "bbox": None,
                "properties": {"platform": "Sentinel-2", "constellation": "Demo"},
            }
        )
        cur -= timedelta(days=step)
    return scenes


def search_scenes(
    aoi_id: str,
    bbox_wgs84: list[float],
    start_date: str,
    end_date: str,
    max_cloud_pct: int = 30,
    max_images: int = 10,
    sentinel_hub_available: bool = True,
) -> list[dict[str, Any]]:
    """Search Sentinel-2 scenes and cache scene metadata for AOI."""
    max_images = max(1, int(max_images))
    search_mode = "cdse"
    search_error = None
    try:
        if sentinel_hub_available:
            scenes = _search_scenes_cdse(bbox_wgs84, start_date, end_date, int(max_cloud_pct), max_images)
        else:
            search_mode = "demo_fallback_no_credentials"
            scenes = _search_scenes_demo(start_date, end_date, max_images)
    except Exception as exc:
        search_mode = "demo_fallback_error"
        search_error = str(exc)
        scenes = _search_scenes_demo(start_date, end_date, max_images)

    cache_payload = {
        "aoi_id": aoi_id,
        "bbox_wgs84": bbox_wgs84,
        "start_date": start_date,
        "end_date": end_date,
        "max_cloud_pct": max_cloud_pct,
        "cached_at": datetime.utcnow().isoformat() + "Z",
        "search_mode": search_mode,
        "search_error": search_error,
        "scenes": scenes,
    }
    _save_scene_cache(aoi_id, cache_payload)
    return scenes


def resolve_scenes(aoi_id: str, scene_ids: list[str]) -> list[dict[str, Any]]:
    cache = load_scene_cache(aoi_id)
    scenes = cache.get("scenes", [])
    by_id = {scene["scene_id"]: scene for scene in scenes}

    resolved = [by_id[sid] for sid in scene_ids if sid in by_id]
    if len(resolved) != len(scene_ids):
        missing = sorted(set(scene_ids) - set(by_id))
        raise SceneLookupError(f"Scene IDs not found in cache: {', '.join(missing)}")
    return resolved


def _evalscript_raw_bands() -> str:
    return """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08", "B11"], units: "REFLECTANCE" }],
    output: { bands: 5, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11];
}
"""


def _scene_time_window(scene_datetime: str) -> tuple[str, str]:
    dt = _try_parse_iso(scene_datetime)
    # Use a wider temporal window around the catalog timestamp to avoid
    # CDSE process API misses caused by strict minute-level filtering.
    start = (dt - timedelta(hours=12)).astimezone(timezone.utc)
    end = (dt + timedelta(hours=12)).astimezone(timezone.utc)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _bbox_dimensions_for_10m(bbox_wgs84: list[float]) -> tuple[int, int]:
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    lon_span = max(1e-8, float(max_lon - min_lon))
    lat_span = max(1e-8, float(max_lat - min_lat))

    lat_mid = (float(min_lat) + float(max_lat)) / 2.0
    cos_lat = max(float(np.cos(np.radians(lat_mid))), 0.1)

    meters_x = lon_span * 111_320.0 * cos_lat
    meters_y = lat_span * 110_540.0

    width = int(np.ceil(meters_x / 10.0))
    height = int(np.ceil(meters_y / 10.0))

    # Keep within practical process API limits.
    width = int(min(max(width, 64), 2500))
    height = int(min(max(height, 64), 2500))
    return width, height


def _download_scene_cdse(
    bbox_wgs84: list[float],
    scene: dict[str, Any],
    out_tif_path: Path,
) -> None:
    conf = load_sentinel_hub_config()
    try:
        token = get_cdse_access_token(conf["client_id"], conf["client_secret"])
    except Exception as exc:
        raise RuntimeError(f"Unable to acquire CDSE token for process download: {exc}") from exc

    t_from, t_to = _scene_time_window(scene["datetime"])
    width, height = _bbox_dimensions_for_10m(bbox_wgs84)

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox_wgs84,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": t_from, "to": t_to},
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
            "width": width,
            "height": height,
        },
        "evalscript": _evalscript_raw_bands(),
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff",
    }

    resp = requests.post(
        "https://sh.dataspace.copernicus.eu/api/v1/process",
        json=payload,
        headers=headers,
        timeout=180,
    )
    if resp.status_code != 200:
        snippet = (resp.text or "")[:500]
        raise RuntimeError(f"CDSE process error {resp.status_code}: {snippet}")

    out_tif_path.parent.mkdir(parents=True, exist_ok=True)
    with out_tif_path.open("wb") as f:
        f.write(resp.content)


def _geotiff_has_signal(tif_path: Path) -> bool:
    """Return False when TIFF appears to contain only nodata/zero values."""
    try:
        with rasterio.open(tif_path) as src:
            data = src.read().astype(np.float32)
            if data.size == 0:
                return False
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            finite = np.isfinite(data)
            if not np.any(finite):
                return False
            # Empty Process API responses are commonly all zeros.
            return bool(np.nanmax(np.abs(data[finite])) > 1e-8)
    except Exception:
        return False


def _write_demo_geotiff(out_tif_path: Path, bbox_wgs84: list[float]) -> None:
    out_tif_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 256, 256
    transform = from_bounds(*bbox_wgs84, width, height)

    y = np.linspace(0, 1, height, dtype=np.float32).reshape(-1, 1)
    x = np.linspace(0, 1, width, dtype=np.float32).reshape(1, -1)

    # Build full-sized synthetic bands so stacking always succeeds.
    x2d = np.broadcast_to(x, (height, width))
    y2d = np.broadcast_to(y, (height, width))

    blue = 0.1 + 0.05 * x2d
    green = 0.2 + 0.3 * y2d
    red = 0.15 + 0.2 * y2d
    nir = 0.1 + 0.6 * y2d
    swir = 0.4 - (0.25 * y2d)

    stack = np.stack([blue, green, red, nir, swir]).astype(np.float32)

    with rasterio.open(
        out_tif_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=5,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        for band_index in range(5):
            dst.write(stack[band_index, :, :], band_index + 1)


def download_scene_multiband_tiff(
    run_id: str,
    aoi_bbox_wgs84: list[float],
    scene: dict[str, Any],
    sentinel_hub_available: bool = True,
) -> dict[str, Any]:
    """Download 5-band (B02/B03/B04/B08/B11) GeoTIFF for one scene."""
    scene_id_safe = scene["scene_id"].replace("/", "_")
    run_imagery_dir = RUNS_DIR / run_id / "imagery"
    out_tif_path = run_imagery_dir / f"{scene_id_safe}.tif"
    out_meta_path = run_imagery_dir / f"{scene_id_safe}.json"
    search_fallback_demo = str(scene.get("collection", "")).lower() == "demo"

    download_error = None
    if sentinel_hub_available and not search_fallback_demo:
        try:
            _download_scene_cdse(aoi_bbox_wgs84, scene, out_tif_path)
            if not _geotiff_has_signal(out_tif_path):
                download_error = "CDSE returned empty raster (all nodata/zero values)"
                out_tif_path.unlink(missing_ok=True)
        except Exception as exc:
            download_error = str(exc)
    elif search_fallback_demo:
        download_error = "Scene list is demo fallback; generating synthetic demo raster."

    if not out_tif_path.exists():
        _write_demo_geotiff(out_tif_path, aoi_bbox_wgs84)

    metadata = {
        "scene_id": scene["scene_id"],
        "datetime": scene["datetime"],
        "cloud_pct": scene.get("cloud_pct"),
        "collection": scene.get("collection", "sentinel-2-l2a"),
        "search_fallback_demo": search_fallback_demo,
        "bbox_wgs84": aoi_bbox_wgs84,
        "bands": ["B02", "B03", "B04", "B08", "B11"],
        "filepath": str(out_tif_path),
        "download_error": download_error,
        "is_demo": search_fallback_demo or download_error is not None or not sentinel_hub_available,
    }

    with out_meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
