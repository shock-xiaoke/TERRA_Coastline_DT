"""Flask application for TERRA UGLA AOI-first coastal extraction workflows."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

# Keep compatibility with existing package execution paths.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.terra_ugla.config import create_data_directories, initialize_sentinel_hub_config
from src.terra_ugla.models.geojson import create_shoreline_geojson
from src.terra_ugla.services.aoi import create_aoi, create_or_get_fixed_aoi, list_aois, load_aoi
from src.terra_ugla.services.baseline import create_baseline, generate_transects, list_baselines
from src.terra_ugla.services.digital_twin import (
    get_digital_twin_bootstrap,
    predict_from_digital_twin_state,
    run_assimilation_cycle,
    start_digital_twin_scheduler,
)
from src.terra_ugla.services.extraction import DEFAULT_PARAMS, execute_extraction_job
from src.terra_ugla.services.imagery import load_scene_cache, search_scenes
from src.terra_ugla.services.jobs import job_manager
from src.terra_ugla.services.prediction import run_prediction
from src.terra_ugla.services.shoreline import delete_shoreline, list_shorelines, load_shoreline, save_shoreline

# Flask init.
app = Flask(__name__, template_folder="../../templates", static_folder="../../static")

create_data_directories()
_config, sentinel_hub_available = initialize_sentinel_hub_config()
start_digital_twin_scheduler(job_manager=job_manager, sentinel_hub_available=sentinel_hub_available)


def _run_prediction_job(
    job_id: str,
    phase_callback,
    run_id: str,
    train_split_date: str | None,
    sequence_len_days: int,
    forecast_days: int,
    model_preference: str,
) -> dict[str, Any]:
    phase_callback(job_id, "prepare", "running", "Preparing prediction dataset")
    phase_callback(job_id, "prepare", "completed", "Dataset loaded")

    phase_callback(job_id, "train", "running", "On-the-fly fine-tuning prediction model")
    artifacts = run_prediction(
        run_id=run_id,
        train_split_date=train_split_date,
        sequence_len_days=sequence_len_days,
        forecast_days=forecast_days,
        model_preference=model_preference,
    )
    phase_callback(job_id, "train", "completed", "Model fine-tuning finished")

    phase_callback(job_id, "forecast", "running", "Generating forecast")
    phase_callback(job_id, "forecast", "completed", "Forecast generated")

    phase_callback(job_id, "export", "running", "Writing prediction artifacts")
    phase_callback(job_id, "export", "completed", "Prediction artifacts exported")

    return {
        "run_id": run_id,
        "summary": artifacts.summary,
        "forecast_rows": int(len(artifacts.forecast_df)),
    }


def _run_extract_job(job_id: str, phase_callback, **kwargs) -> dict[str, Any]:
    return execute_extraction_job(job_id=job_id, phase_callback=phase_callback, **kwargs)


def _run_assimilate_job(job_id: str, phase_callback, **kwargs) -> dict[str, Any]:
    return run_assimilation_cycle(
        job_id=job_id,
        phase_callback=phase_callback,
        sentinel_hub_available=sentinel_hub_available,
        **kwargs,
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/baseline-workflow")
def baseline_workflow():
    return render_template("baseline_workflow.html")


@app.route("/aoi", methods=["POST"])
def create_aoi_route():
    try:
        data = request.get_json(force=True) or {}
        name = (data.get("name") or f"aoi_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}").strip()
        polygon_latlng = data.get("polygon_latlng", [])
        close_polygon = bool(data.get("close_polygon", True))

        record = create_aoi(name=name, polygon_latlng=polygon_latlng, close_polygon=close_polygon)
        return jsonify(
            {
                "success": True,
                "aoi_id": record["aoi_id"],
                "bbox_wgs84": record["bbox_wgs84"],
                "utm_epsg": record["utm_epsg"],
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/aoi", methods=["GET"])
def list_aois_route():
    return jsonify({"success": True, "aois": list_aois()})


@app.route("/aoi/fixed", methods=["GET", "POST"])
def fixed_aoi_route():
    try:
        data = request.get_json(silent=True) or {}
        force_recreate = bool(data.get("force_recreate", False))
        record = create_or_get_fixed_aoi(force_recreate=force_recreate)
        return jsonify(
            {
                "success": True,
                "aoi_id": record["aoi_id"],
                "name": record["name"],
                "bbox_wgs84": record["bbox_wgs84"],
                "utm_epsg": record["utm_epsg"],
                "is_fixed_model_aoi": bool(record.get("is_fixed_model_aoi", True)),
                "polygon_latlng": record.get("polygon_latlng"),
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/dt/bootstrap", methods=["GET"])
def digital_twin_bootstrap_route():
    try:
        aoi_id = request.args.get("aoi_id") or None
        payload = get_digital_twin_bootstrap(aoi_id=aoi_id)
        return jsonify({"success": True, **payload})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/dt/predict", methods=["POST"])
def digital_twin_predict_route():
    try:
        data = request.get_json(force=True) or {}
        forecast_years = float(data.get("forecast_years", 1.0))
        sequence_len_days = int(data.get("sequence_len_days", 30))
        lookback_days = int(data.get("lookback_days", 730))
        model_preference = str(data.get("model_preference", "mamba_lstm"))
        aoi_id = data.get("aoi_id")

        payload = predict_from_digital_twin_state(
            forecast_years=forecast_years,
            aoi_id=aoi_id,
            sequence_len_days=sequence_len_days,
            lookback_days=lookback_days,
            model_preference=model_preference,
        )
        return jsonify({"success": True, **payload})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/baseline", methods=["POST"])
def create_baseline_route():
    try:
        data = request.get_json(force=True) or {}
        name = (data.get("name") or f"baseline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}").strip()
        line_latlng = data.get("line_latlng", [])

        record = create_baseline(name=name, line_latlng=line_latlng)
        return jsonify(
            {
                "success": True,
                "baseline_id": record["baseline_id"],
                "bbox_wgs84": record["bbox_wgs84"],
                "utm_epsg": record["utm_epsg"],
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/baseline", methods=["GET"])
def list_baselines_route():
    return jsonify({"success": True, "baselines": list_baselines()})


@app.route("/baseline/<baseline_id>/transects", methods=["POST"])
def generate_transects_route(baseline_id: str):
    try:
        data = request.get_json(force=True) or {}
        spacing_m = float(data.get("spacing_m", 20.0))
        transect_length_m = float(data.get("transect_length_m", 500.0))
        offshore_ratio = float(data.get("offshore_ratio", 0.7))

        record = generate_transects(
            baseline_id=baseline_id,
            spacing_m=spacing_m,
            transect_length_m=transect_length_m,
            offshore_ratio=offshore_ratio,
        )
        return jsonify(
            {
                "success": True,
                "baseline_id": record["baseline_id"],
                "transect_count": record["transect_count"],
                "utm_epsg": record["utm_epsg"],
                "offshore_ratio": record["offshore_ratio"],
                "offshore_length_m": record["offshore_length_m"],
                "onshore_length_m": record["onshore_length_m"],
                "geojson": record["geojson"],
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/imagery/search", methods=["POST"])
def imagery_search_route():
    try:
        data = request.get_json(force=True) or {}
        aoi_id = data.get("aoi_id")
        if not aoi_id:
            return jsonify({"success": False, "error": "aoi_id is required"}), 400

        start_date = data.get("start_date", "2023-01-01")
        end_date = data.get("end_date", datetime.utcnow().strftime("%Y-%m-%d"))
        max_cloud_pct = int(data.get("max_cloud_pct", 30))
        max_images = int(data.get("max_images", 10))

        aoi = load_aoi(aoi_id)
        scenes = search_scenes(
            aoi_id=aoi_id,
            bbox_wgs84=aoi["bbox_wgs84"],
            start_date=start_date,
            end_date=end_date,
            max_cloud_pct=max_cloud_pct,
            max_images=max_images,
            sentinel_hub_available=sentinel_hub_available,
        )
        scene_cache = load_scene_cache(aoi_id)

        return jsonify(
            {
                "success": True,
                "scene_count": len(scenes),
                "search_mode": scene_cache.get("search_mode", "unknown"),
                "search_error": scene_cache.get("search_error"),
                "scenes": [
                    {
                        "scene_id": scene["scene_id"],
                        "datetime": scene["datetime"],
                        "cloud_pct": scene.get("cloud_pct"),
                    }
                    for scene in scenes
                ],
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/jobs/extract", methods=["POST"])
def create_extract_job_route():
    try:
        data = request.get_json(force=True) or {}
        aoi_id = data.get("aoi_id")
        scene_ids = data.get("scene_ids") or []
        if not aoi_id:
            return jsonify({"success": False, "error": "aoi_id is required"}), 400
        if not isinstance(scene_ids, list) or not scene_ids:
            return jsonify({"success": False, "error": "scene_ids must be a non-empty list"}), 400

        params = {
            "transect_spacing_m": float(data.get("transect_spacing_m", DEFAULT_PARAMS["transect_spacing_m"])),
            "transect_length_m": float(data.get("transect_length_m", DEFAULT_PARAMS["transect_length_m"])),
            "offshore_ratio": float(data.get("offshore_ratio", DEFAULT_PARAMS["offshore_ratio"])),
            "max_dist_ref_m": float(data.get("max_dist_ref_m", DEFAULT_PARAMS["max_dist_ref_m"])),
        }

        phases = [
            "imagery_download",
            "waterline_extract",
            "transect_build",
            "ve_extract",
            "intersections",
            "export",
        ]
        job_id = job_manager.submit_job(
            job_type="extract",
            phases=phases,
            fn=_run_extract_job,
            aoi_id=aoi_id,
            scene_ids=scene_ids,
            sentinel_hub_available=sentinel_hub_available,
            **params,
        )

        return jsonify({"success": True, "job_id": job_id, "status": "queued"})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/jobs/assimilate", methods=["POST"])
def create_assimilate_job_route():
    try:
        data = request.get_json(force=True) or {}
        aoi_id = data.get("aoi_id")
        phases = [
            "poll",
            "imagery_download",
            "waterline_extract",
            "transect_build",
            "ve_extract",
            "intersections",
            "export",
            "state_update",
            "retrain_check",
        ]
        job_id = job_manager.submit_job(
            job_type="assimilate",
            phases=phases,
            fn=_run_assimilate_job,
            aoi_id=aoi_id,
            max_cloud_pct=int(data.get("max_cloud_pct", 30)),
            max_images=int(data.get("max_images", 6)),
            lookback_days=int(data.get("lookback_days", 45)),
            sequence_len_days=int(data.get("sequence_len_days", 30)),
            model_preference=str(data.get("model_preference", "mamba_lstm")),
            retrain_interval_days=int(data.get("retrain_interval_days", 180)),
            min_samples_for_retrain=int(data.get("min_samples_for_retrain", 200)),
        )
        return jsonify({"success": True, "job_id": job_id, "status": "queued"})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/jobs/predict", methods=["POST"])
def create_predict_job_route():
    try:
        data = request.get_json(force=True) or {}
        run_id = data.get("run_id")
        if not run_id:
            return jsonify({"success": False, "error": "run_id is required"}), 400

        sequence_len_days = int(data.get("sequence_len_days", 10))
        if data.get("forecast_years") is not None:
            forecast_days = max(1, int(round(float(data.get("forecast_years")) * 365.25)))
        else:
            forecast_days = int(data.get("forecast_days", 30))
        train_split_date = data.get("train_split_date")
        model_preference = str(data.get("model_preference", "mamba_lstm"))

        phases = ["prepare", "train", "forecast", "export"]
        job_id = job_manager.submit_job(
            job_type="predict",
            phases=phases,
            fn=_run_prediction_job,
            run_id=run_id,
            train_split_date=train_split_date,
            sequence_len_days=sequence_len_days,
            forecast_days=forecast_days,
            model_preference=model_preference,
        )
        return jsonify({"success": True, "job_id": job_id, "status": "queued"})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job_route(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    return jsonify({"success": True, **job})


@app.route("/results/<run_id>/summary", methods=["GET"])
def get_result_summary_route(run_id: str):
    run_dir = Path("data") / "runs" / run_id
    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        return jsonify({"success": False, "error": "Run summary not found"}), 404

    with summary_file.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    prediction_summary_file = run_dir / "predictions" / "summary.json"
    if prediction_summary_file.exists():
        with prediction_summary_file.open("r", encoding="utf-8") as f:
            summary["prediction"] = json.load(f)

    return jsonify({"success": True, "summary": summary})


@app.route("/results/<run_id>/download", methods=["GET"])
def download_result_route(run_id: str):
    file_type = request.args.get("type", "geojson")
    artifact = request.args.get("artifact", "default")

    run_dir = Path("data") / "runs" / run_id
    results_dir = run_dir / "results"
    pred_dir = run_dir / "predictions"
    exports_dir = run_dir / "exports"

    if file_type == "geojson":
        if artifact in ("default", "intersections"):
            path = results_dir / "intersections.geojson"
        elif artifact == "transects":
            path = results_dir / "transects.geojson"
        elif artifact == "waterlines":
            path = exports_dir / "waterlines.geojson"
        elif artifact == "veglines":
            path = exports_dir / "veglines.geojson"
        elif artifact in ("forecast_shorelines", "predicted_shorelines"):
            path = pred_dir / "shoreline_forecast.geojson"
        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid geojson artifact. Use intersections|transects|waterlines|veglines|forecast_shorelines",
                    }
                ),
                400,
            )
    elif file_type == "csv":
        if artifact == "forecast":
            path = pred_dir / "forecast.csv"
        elif artifact == "metrics":
            path = pred_dir / "metrics.csv"
        else:
            path = results_dir / "timeseries.csv"
    elif file_type == "parquet":
        if artifact == "forecast":
            path = pred_dir / "forecast.parquet"
        else:
            path = results_dir / "timeseries.parquet"
    else:
        return jsonify({"success": False, "error": "Invalid type. Use geojson|csv|parquet"}), 400

    path = path.resolve()

    if not path.exists():
        return jsonify({"success": False, "error": f"Requested file not found: {path.name}"}), 404

    return send_file(path, as_attachment=True, download_name=path.name)


# -----------------------------------------------------------------------------
# Legacy shoreline-first routes kept for compatibility (deprecated).
# -----------------------------------------------------------------------------


@app.route("/save_shoreline", methods=["POST"])
def save_shoreline_route():
    try:
        data = request.get_json(force=True) or {}
        coordinates = data.get("coordinates", [])
        name = data.get("name", f"shoreline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

        if len(coordinates) < 2:
            return jsonify({"success": False, "error": "At least 2 points required"}), 400

        geojson_coords = [[coord[1], coord[0]] for coord in coordinates]
        geojson_data = create_shoreline_geojson(geojson_coords, name)
        filepath = save_shoreline(geojson_data, f"{name}.geojson")

        return jsonify(
            {
                "success": True,
                "deprecated": True,
                "message": "Legacy endpoint. Use /aoi for AOI-first workflow.",
                "filename": filepath,
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/generate_transects", methods=["POST"])
def deprecated_generate_transects_route():
    return (
        jsonify(
            {
                "success": False,
                "deprecated": True,
                "error": "Legacy endpoint deprecated. Use /jobs/extract with AOI-first workflow.",
            }
        ),
        410,
    )


@app.route("/get_satellite_data", methods=["POST"])
def deprecated_satellite_route():
    return (
        jsonify(
            {
                "success": False,
                "deprecated": True,
                "error": "Legacy endpoint deprecated. Use /imagery/search and /jobs/extract.",
            }
        ),
        410,
    )


@app.route("/analyze_vegetation_edge", methods=["POST"])
def deprecated_analyze_route():
    return (
        jsonify(
            {
                "success": False,
                "deprecated": True,
                "error": "Legacy endpoint deprecated. Use /jobs/extract.",
            }
        ),
        410,
    )


@app.route("/extract_vegetation_contours", methods=["POST"])
def deprecated_contours_route():
    return (
        jsonify(
            {
                "success": False,
                "deprecated": True,
                "error": "Legacy endpoint deprecated. Use /jobs/extract.",
            }
        ),
        410,
    )


@app.route("/list_shorelines", methods=["GET"])
def list_shorelines_route():
    try:
        return jsonify(list_shorelines())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/load_shoreline/<filename>", methods=["GET"])
def load_shoreline_route(filename: str):
    try:
        return jsonify(load_shoreline(filename))
    except FileNotFoundError:
        return jsonify({"error": "Shoreline file not found"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/delete_shoreline/<filename>", methods=["DELETE"])
def delete_shoreline_route(filename: str):
    try:
        delete_shoreline(filename)
        return jsonify({"success": True, "message": f"Deleted {filename}"})
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/check_sentinel_hub_status", methods=["GET"])
def check_sentinel_hub_status_route():
    return jsonify(
        {
            "status": "configured" if sentinel_hub_available else "not_configured",
            "available": bool(sentinel_hub_available),
            "message": (
                "Sentinel Hub/CDSE is configured"
                if sentinel_hub_available
                else "Sentinel Hub/CDSE is not configured; demo imagery fallback will be used"
            ),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
