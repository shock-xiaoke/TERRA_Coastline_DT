"""AOI-scene extraction orchestration: waterline + vegetation edge + intersections."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import LineString, mapping

from ..coastguard_port import (
    classify_image_nn,
    classify_image_nn_shore,
    contour_pixels_to_lines,
    find_contours_weighted_peaks,
    nd_index,
    pick_primary_line,
)
from ..coastguard_port.classification import load_classifier
from .aoi import load_aoi
from .imagery import download_scene_multiband_tiff, resolve_scenes
from .intersections import (
    build_intersection_timeseries,
    generate_transects_from_baseline,
    transform_geometry,
    transects_to_geojson,
)
from .unet_segmentation import MODEL_INPUT_SIZE, segment_waterline_from_multiband

RUNS_DIR = Path("data") / "runs"
MODEL_CACHE_DIR = Path("data") / "models" / "coastguard"

DEFAULT_PARAMS = {
    "transect_spacing_m": 100,
    "transect_length_m": 500,
    "offshore_ratio": 0.7,
    "max_dist_ref_m": 150,
}

REQUIRED_MODELS = {
    "veg": "MLPClassifier_Veg_L5L8S2.pkl",
    "shore": "NN_4classes_S2.pkl",
}


def _ensure_model_artifacts() -> dict[str, Path]:
    """Copy required COASTGUARD model files into TERRA-managed model cache."""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    source_dir = Path("COASTGUARD") / "Classification" / "models"

    model_paths: dict[str, Path] = {}
    for key, filename in REQUIRED_MODELS.items():
        src = source_dir / filename
        dst = MODEL_CACHE_DIR / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        model_paths[key] = dst if dst.exists() else src
    return model_paths


def _load_models() -> dict[str, Any]:
    model_paths = _ensure_model_artifacts()
    return {
        "veg": load_classifier(model_paths["veg"]),
        "shore": load_classifier(model_paths["shore"]),
        "paths": {k: str(v) for k, v in model_paths.items()},
    }


def _buffer_distance_in_raster_units(src: rasterio.io.DatasetReader, max_dist_ref_m: float, lat_hint: float) -> float:
    if src.crs is None:
        return max_dist_ref_m / 111_320.0

    try:
        if src.crs.is_geographic:
            cos_lat = max(np.cos(np.radians(lat_hint)), 0.1)
            return max_dist_ref_m / (111_320.0 * cos_lat)
    except Exception:
        pass

    return max_dist_ref_m


def _min_line_length_in_raster_units(src: rasterio.io.DatasetReader, min_length_m: float) -> float:
    bounds = src.bounds
    lat_hint = float((bounds.bottom + bounds.top) / 2.0) if bounds is not None else 0.0
    return _buffer_distance_in_raster_units(src, min_length_m, lat_hint)


def _reference_buffer_mask(
    src: rasterio.io.DatasetReader,
    cloud_mask: np.ndarray,
    baseline_line_utm: LineString | None,
    utm_epsg: int,
    max_dist_ref_m: float,
) -> np.ndarray:
    if baseline_line_utm is None:
        return ~cloud_mask

    src_epsg = src.crs.to_epsg() if src.crs is not None else 4326
    baseline_in_raster = transform_geometry(baseline_line_utm, utm_epsg, src_epsg)

    lat_hint = baseline_in_raster.centroid.y if baseline_in_raster is not None else 0.0
    buff_units = _buffer_distance_in_raster_units(src, max_dist_ref_m, lat_hint)
    buffer_geom = baseline_in_raster.buffer(buff_units)

    mask = rasterize(
        [(buffer_geom, 1)],
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def _extract_scene_lines(
    scene_meta: dict[str, Any],
    aoi_polygon_wgs84,
    baseline_line_utm: LineString | None,
    utm_epsg: int,
    max_dist_ref_m: float,
    models: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    tif_path = Path(scene_meta["filepath"])
    with rasterio.open(tif_path) as src:
        src_epsg = src.crs.to_epsg() if src.crs is not None else 4326
        stack = src.read([1, 2, 3, 4, 5]).astype(float)
        im_ms = np.moveaxis(stack, 0, -1)  # (rows, cols, bands)

        cloud_mask = np.any(~np.isfinite(im_ms), axis=2)
        cloud_mask = np.logical_or(cloud_mask, np.all(im_ms == 0, axis=2))

        # Restrict processing to the actual AOI polygon, not the entire bbox tile.
        if aoi_polygon_wgs84 is not None:
            if src_epsg == 4326:
                aoi_in_raster = aoi_polygon_wgs84
            else:
                aoi_in_raster = transform_geometry(aoi_polygon_wgs84, 4326, src_epsg)
            aoi_mask = rasterize(
                [(aoi_in_raster, 1)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            cloud_mask = np.logical_or(cloud_mask, ~aoi_mask)
        else:
            aoi_mask = np.ones((src.height, src.width), dtype=bool)

        _, veg_labels = classify_image_nn(im_ms, cloud_mask, model=models.get("veg"), min_patch_size=30)
        ref_mask = _reference_buffer_mask(src, cloud_mask, baseline_line_utm, utm_epsg, max_dist_ref_m)
        ref_mask = np.logical_and(ref_mask, aoi_mask)
        min_line_length = _min_line_length_in_raster_units(src, 20.0)

        segmentation_result = segment_waterline_from_multiband(
            im_ms=im_ms,
            affine_transform=src.transform,
            src_epsg=src_epsg,
            utm_epsg=utm_epsg,
            cloud_mask=cloud_mask,
            aoi_mask=aoi_mask,
            threshold=0.5,
        )

        wl_primary_raster = segmentation_result.waterline_src
        wl_threshold = segmentation_result.threshold
        wl_lines_raster: list[LineString] = [wl_primary_raster] if wl_primary_raster is not None else []

        if wl_primary_raster is None:
            mndwi = nd_index(im_ms[:, :, 1], im_ms[:, :, 4], cloud_mask)
            _, shore_labels = classify_image_nn_shore(im_ms, cloud_mask, model=models.get("shore"), min_patch_size=30)
            water_class = shore_labels[:, :, 2]
            non_water_class = np.logical_or(shore_labels[:, :, 0], shore_labels[:, :, 1])
            wl_contours, wl_threshold = find_contours_weighted_peaks(
                mndwi,
                water_class,
                non_water_class,
                ref_mask,
                cloud_mask,
            )
            wl_lines_raster = contour_pixels_to_lines(wl_contours, src.transform, min_length=min_line_length)
            wl_primary_raster = pick_primary_line(wl_lines_raster)
            waterline_method = "coastguard_weighted_peaks"
            segmentation_error = segmentation_result.error
        else:
            waterline_method = segmentation_result.method
            segmentation_error = None

        ndvi = nd_index(im_ms[:, :, 3], im_ms[:, :, 2], cloud_mask)
        veg_class = veg_labels[:, :, 0]
        non_veg_class = veg_labels[:, :, 1]
        ve_contours, ve_threshold = find_contours_weighted_peaks(
            ndvi,
            veg_class,
            non_veg_class,
            ref_mask,
            cloud_mask,
        )
        ve_lines_raster = contour_pixels_to_lines(ve_contours, src.transform, min_length=min_line_length)
        ve_primary_raster = pick_primary_line(ve_lines_raster)

    def _to_crs(line: LineString | None, dst_epsg: int):
        if line is None:
            return None
        if src_epsg == dst_epsg:
            return line
        return transform_geometry(line, src_epsg, dst_epsg)

    def _list_to_crs(lines: list[LineString], dst_epsg: int) -> list[LineString]:
        out = []
        for line in lines:
            if src_epsg == dst_epsg:
                out.append(line)
            else:
                out.append(transform_geometry(line, src_epsg, dst_epsg))
        return out

    wl_primary_wgs84 = _to_crs(wl_primary_raster, 4326)
    wl_primary_utm = _to_crs(wl_primary_raster, utm_epsg)
    ve_primary_wgs84 = _to_crs(ve_primary_raster, 4326)
    ve_primary_utm = _to_crs(ve_primary_raster, utm_epsg)

    return {
        "run_id": run_id,
        "scene_id": scene_meta["scene_id"],
        "datetime": scene_meta["datetime"],
        "cloud_pct": scene_meta.get("cloud_pct"),
        "wl_threshold": wl_threshold,
        "ve_threshold": ve_threshold,
        "waterline_wgs84": wl_primary_wgs84,
        "waterline_utm": wl_primary_utm,
        "vegline_wgs84": ve_primary_wgs84,
        "vegline_utm": ve_primary_utm,
        "waterline_segments_wgs84": _list_to_crs(wl_lines_raster, 4326),
        "vegline_segments_wgs84": _list_to_crs(ve_lines_raster, 4326),
        "waterline_method": waterline_method,
        "segmentation_model": segmentation_result.model_path,
        "segmentation_error": segmentation_error,
    }


def _lines_geojson(per_scene_results: list[dict[str, Any]], key: str, type_label: str) -> dict[str, Any]:
    features = []
    for scene in per_scene_results:
        line = scene.get(key)
        if line is None:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "run_id": scene["run_id"],
                    "scene_id": scene["scene_id"],
                    "datetime": scene["datetime"],
                    "boundary_type": type_label,
                    "threshold": scene.get("wl_threshold" if type_label == "waterline" else "ve_threshold"),
                },
                "geometry": mapping(line),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _intersections_geojson(df: pd.DataFrame) -> dict[str, Any]:
    features = []
    for _, row in df.iterrows():
        if pd.notna(row.get("wl_lon")) and pd.notna(row.get("wl_lat")):
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "run_id": row["run_id"],
                        "scene_id": row["scene_id"],
                        "datetime": row["datetime"].isoformat() if hasattr(row["datetime"], "isoformat") else str(row["datetime"]),
                        "transect_id": int(row["transect_id"]),
                        "type": "waterline",
                        "distance_m": float(row["WL_distance_m"]),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["wl_lon"]), float(row["wl_lat"])],
                    },
                }
            )
        if pd.notna(row.get("ve_lon")) and pd.notna(row.get("ve_lat")):
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "run_id": row["run_id"],
                        "scene_id": row["scene_id"],
                        "datetime": row["datetime"].isoformat() if hasattr(row["datetime"], "isoformat") else str(row["datetime"]),
                        "transect_id": int(row["transect_id"]),
                        "type": "vegetation_edge",
                        "distance_m": float(row["VE_distance_m"]),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["ve_lon"]), float(row["ve_lat"])],
                    },
                }
            )

    return {"type": "FeatureCollection", "features": features}


def execute_extraction_job(
    job_id: str,
    phase_callback,
    aoi_id: str,
    scene_ids: list[str],
    sentinel_hub_available: bool,
    transect_spacing_m: float = DEFAULT_PARAMS["transect_spacing_m"],
    transect_length_m: float = DEFAULT_PARAMS["transect_length_m"],
    offshore_ratio: float = DEFAULT_PARAMS["offshore_ratio"],
    max_dist_ref_m: float = DEFAULT_PARAMS["max_dist_ref_m"],
) -> dict[str, Any]:
    """Full extraction workflow used by async jobs endpoint."""
    run_id = f"{aoi_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id
    imagery_dir = run_dir / "imagery"
    results_dir = run_dir / "results"
    exports_dir = run_dir / "exports"
    for path in [imagery_dir, results_dir, exports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    aoi = load_aoi(aoi_id)
    scenes = resolve_scenes(aoi_id, scene_ids)
    if not scenes:
        raise ValueError("No scenes selected for extraction")

    models = _load_models()

    phase_callback(job_id, "imagery_download", "running", "Downloading scene rasters")
    downloaded_scenes = []
    for scene in scenes:
        downloaded_scenes.append(
            download_scene_multiband_tiff(
                run_id=run_id,
                aoi_bbox_wgs84=aoi["bbox_wgs84"],
                scene=scene,
                sentinel_hub_available=sentinel_hub_available,
            )
        )
    phase_callback(job_id, "imagery_download", "completed", f"Downloaded {len(downloaded_scenes)} scenes")

    demo_scene_count = sum(1 for scene in downloaded_scenes if bool(scene.get("is_demo")))
    real_scene_count = len(downloaded_scenes) - demo_scene_count
    search_fallback_scene_count = sum(1 for scene in downloaded_scenes if bool(scene.get("search_fallback_demo")))
    scene_errors = [
        {
            "scene_id": scene.get("scene_id"),
            "error": scene.get("download_error"),
        }
        for scene in downloaded_scenes
        if scene.get("download_error")
    ]

    phase_callback(job_id, "waterline_extract", "running", "Extracting waterlines")
    phase_callback(job_id, "ve_extract", "running", "Extracting vegetation edges")

    per_scene_results: list[dict[str, Any]] = []
    baseline_line_utm: LineString | None = None

    for scene_meta in downloaded_scenes:
        result = _extract_scene_lines(
            scene_meta=scene_meta,
            aoi_polygon_wgs84=aoi.get("polygon"),
            baseline_line_utm=baseline_line_utm,
            utm_epsg=aoi["utm_epsg"],
            max_dist_ref_m=max_dist_ref_m,
            models=models,
            run_id=run_id,
        )

        if baseline_line_utm is None and result.get("waterline_utm") is not None:
            baseline_line_utm = result["waterline_utm"]

        per_scene_results.append(result)

    unet_scene_count = sum(1 for scene in per_scene_results if scene.get("waterline_method") == "unet_robust")
    waterline_fallback_count = len(per_scene_results) - unet_scene_count
    segmentation_errors = [
        {
            "scene_id": scene.get("scene_id"),
            "error": scene.get("segmentation_error"),
        }
        for scene in per_scene_results
        if scene.get("segmentation_error")
    ]

    phase_callback(job_id, "waterline_extract", "completed", "Waterline extraction completed")
    phase_callback(job_id, "ve_extract", "completed", "Vegetation edge extraction completed")

    phase_callback(job_id, "transect_build", "running", "Generating fixed baseline transects")
    if baseline_line_utm is None:
        baseline_line_utm = next(
            (scene.get("vegline_utm") for scene in per_scene_results if scene.get("vegline_utm") is not None),
            None,
        )
    if baseline_line_utm is None:
        raise RuntimeError("Failed to derive baseline from extracted waterlines or vegetation edges")

    baseline_line_wgs84 = transform_geometry(baseline_line_utm, aoi["utm_epsg"], 4326)
    transects = generate_transects_from_baseline(
        baseline_line_wgs84,
        aoi["utm_epsg"],
        spacing_m=transect_spacing_m,
        length_m=transect_length_m,
        offshore_ratio=offshore_ratio,
    )
    if not transects:
        raise RuntimeError("Failed to generate transects from baseline")

    transects_geojson = transects_to_geojson(transects, run_id)
    with (results_dir / "transects.geojson").open("w", encoding="utf-8") as f:
        json.dump(transects_geojson, f, indent=2)
    phase_callback(job_id, "transect_build", "completed", f"Generated {len(transects)} transects")

    phase_callback(job_id, "intersections", "running", "Computing per-transect intersections")
    timeseries_df = build_intersection_timeseries(per_scene_results, transects, aoi["utm_epsg"])
    timeseries_df.to_csv(results_dir / "timeseries.csv", index=False)
    try:
        timeseries_df.to_parquet(results_dir / "timeseries.parquet", index=False)
    except Exception:
        pass
    phase_callback(job_id, "intersections", "completed", "Intersections computed")

    phase_callback(job_id, "export", "running", "Writing GeoJSON/manifest outputs")

    waterlines_geojson = _lines_geojson(per_scene_results, "waterline_wgs84", "waterline")
    veglines_geojson = _lines_geojson(per_scene_results, "vegline_wgs84", "vegetation_edge")
    intersections_geojson = _intersections_geojson(timeseries_df)

    with (exports_dir / "waterlines.geojson").open("w", encoding="utf-8") as f:
        json.dump(waterlines_geojson, f, indent=2)
    with (exports_dir / "veglines.geojson").open("w", encoding="utf-8") as f:
        json.dump(veglines_geojson, f, indent=2)
    with (results_dir / "intersections.geojson").open("w", encoding="utf-8") as f:
        json.dump(intersections_geojson, f, indent=2)

    valid_rows = timeseries_df.dropna(subset=["VE_distance_m", "WL_distance_m"], how="all") if not timeseries_df.empty else timeseries_df
    summary = {
        "run_id": run_id,
        "aoi_id": aoi_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "utm_epsg": aoi["utm_epsg"],
        "scene_count": len(per_scene_results),
        "real_scene_count": real_scene_count,
        "demo_scene_count": demo_scene_count,
        "search_fallback_scene_count": search_fallback_scene_count,
        "used_demo_imagery": bool(demo_scene_count > 0),
        "used_demo_scene_search": bool(search_fallback_scene_count > 0),
        "scene_errors": scene_errors,
        "waterline_model_input_size_px": [MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]],
        "waterline_unet_scene_count": unet_scene_count,
        "waterline_fallback_scene_count": waterline_fallback_count,
        "waterline_segmentation_errors": segmentation_errors,
        "transect_count": len(transects),
        "intersection_rows": int(len(timeseries_df)),
        "valid_intersection_rows": int(len(valid_rows)),
        "files": {
            "timeseries_csv": str(results_dir / "timeseries.csv"),
            "timeseries_parquet": str(results_dir / "timeseries.parquet"),
            "intersections_geojson": str(results_dir / "intersections.geojson"),
            "transects_geojson": str(results_dir / "transects.geojson"),
            "waterlines_geojson": str(exports_dir / "waterlines.geojson"),
            "veglines_geojson": str(exports_dir / "veglines.geojson"),
        },
        "parameters": {
            "transect_spacing_m": transect_spacing_m,
            "transect_length_m": transect_length_m,
            "offshore_ratio": offshore_ratio,
            "max_dist_ref_m": max_dist_ref_m,
        },
        "model_paths": models.get("paths", {}),
    }

    manifest = {
        "run_id": run_id,
        "aoi": {
            "aoi_id": aoi_id,
            "bbox_wgs84": aoi["bbox_wgs84"],
            "utm_epsg": aoi["utm_epsg"],
            "fixed_for_run": True,
        },
        "scenes": downloaded_scenes,
        "summary": summary,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    phase_callback(job_id, "export", "completed", "Exports and manifest written")

    return {
        "run_id": run_id,
        "summary": summary,
        "summary_path": str(run_dir / "summary.json"),
    }
