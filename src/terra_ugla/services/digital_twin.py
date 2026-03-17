"""Digital Twin orchestration: synchronous inference + asynchronous assimilation."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .aoi import create_or_get_fixed_aoi, load_aoi
from .extraction import DEFAULT_PARAMS, execute_extraction_job
from .imagery import search_scenes
from .prediction import (
    resolve_mamba_checkpoint_path,
    run_mamba_coastline_prediction,
    run_prediction,
)

DT_ROOT = Path("data") / "dt"
RUNS_ROOT = Path("data") / "runs"


def _dt_dir(aoi_id: str) -> Path:
    return DT_ROOT / aoi_id


def _state_dir(aoi_id: str) -> Path:
    return _dt_dir(aoi_id) / "state"


def _meta_path(aoi_id: str) -> Path:
    return _state_dir(aoi_id) / "metadata.json"


def _timeseries_path(aoi_id: str) -> Path:
    return _state_dir(aoi_id) / "timeseries.csv"


def _transects_path(aoi_id: str) -> Path:
    return _state_dir(aoi_id) / "transects.geojson"


def _latest_coastline_path(aoi_id: str) -> Path:
    return _state_dir(aoi_id) / "latest_waterline.geojson"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _ensure_state_dirs(aoi_id: str) -> None:
    _state_dir(aoi_id).mkdir(parents=True, exist_ok=True)


def _load_metadata(aoi_id: str) -> dict[str, Any]:
    _ensure_state_dirs(aoi_id)
    path = _meta_path(aoi_id)
    if not path.exists():
        return {
            "aoi_id": aoi_id,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "last_assimilation_at": None,
            "last_scene_datetime": None,
            "last_run_id": None,
            "last_retrained_at": None,
            "active_model_run_id": None,
            "timeseries_rows": 0,
        }
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_metadata(aoi_id: str, metadata: dict[str, Any]) -> None:
    _ensure_state_dirs(aoi_id)
    metadata["updated_at"] = _now_iso()
    with _meta_path(aoi_id).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _load_state_timeseries(aoi_id: str) -> pd.DataFrame:
    path = _timeseries_path(aoi_id)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    return df


def _save_state_timeseries(aoi_id: str, df: pd.DataFrame) -> None:
    _ensure_state_dirs(aoi_id)
    if not df.empty and "datetime" in df.columns:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        sort_cols = [col for col in ["datetime", "transect_id"] if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    df.to_csv(_timeseries_path(aoi_id), index=False)


def _load_geojson(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_geojson(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _discover_latest_seed_run(aoi_id: str) -> Path | None:
    if not RUNS_ROOT.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for run_dir in RUNS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "summary.json"
        ts_path = run_dir / "results" / "timeseries.csv"
        transects_path = run_dir / "results" / "transects.geojson"
        waterline_path = run_dir / "exports" / "waterlines.geojson"

        if not (summary_path.exists() and ts_path.exists() and transects_path.exists()):
            continue

        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue

        summary_aoi_id = str(summary.get("aoi_id", "")).strip()
        if summary_aoi_id and summary_aoi_id != aoi_id:
            continue
        if not summary_aoi_id and not run_dir.name.startswith(f"{aoi_id}_"):
            continue

        if not waterline_path.exists():
            continue

        generated_at = _parse_iso(str(summary.get("generated_at")))
        timestamp = generated_at.timestamp() if generated_at is not None else run_dir.stat().st_mtime
        candidates.append((timestamp, run_dir))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _seed_state_from_latest_run_if_missing(aoi_id: str) -> bool:
    state_df = _load_state_timeseries(aoi_id)
    has_state = not state_df.empty
    has_transects = _transects_path(aoi_id).exists()
    has_latest_coastline = _latest_coastline_path(aoi_id).exists()
    if has_state and has_transects and has_latest_coastline:
        return False

    seed_run_dir = _discover_latest_seed_run(aoi_id)
    if seed_run_dir is None:
        return False

    run_id = seed_run_dir.name
    run_ts_path = seed_run_dir / "results" / "timeseries.csv"
    run_transects_path = seed_run_dir / "results" / "transects.geojson"

    if not has_state and run_ts_path.exists():
        try:
            seed_df = pd.read_csv(run_ts_path)
            _save_state_timeseries(aoi_id, seed_df)
        except Exception:
            pass

    if not has_transects and run_transects_path.exists():
        _ensure_state_dirs(aoi_id)
        _transects_path(aoi_id).write_text(run_transects_path.read_text(encoding="utf-8"), encoding="utf-8")

    if not has_latest_coastline:
        _update_latest_waterline(aoi_id, run_id)

    refreshed_state = _load_state_timeseries(aoi_id)
    metadata = _load_metadata(aoi_id)
    metadata["last_run_id"] = run_id
    metadata["timeseries_rows"] = int(len(refreshed_state))
    if not refreshed_state.empty and "datetime" in refreshed_state.columns:
        max_dt = pd.to_datetime(refreshed_state["datetime"], utc=True, errors="coerce").max()
        if pd.notna(max_dt):
            metadata["last_scene_datetime"] = max_dt.isoformat()
    if metadata.get("last_assimilation_at") is None:
        metadata["last_assimilation_at"] = _now_iso()
    _save_metadata(aoi_id, metadata)
    return True


def get_digital_twin_bootstrap(aoi_id: str | None = None) -> dict[str, Any]:
    """Return fixed AOI + latest state for fast front-end initialization."""
    if aoi_id:
        aoi = load_aoi(aoi_id)
    else:
        aoi = create_or_get_fixed_aoi()
        aoi_id = aoi["aoi_id"]

    _seed_state_from_latest_run_if_missing(aoi_id)
    metadata = _load_metadata(aoi_id)
    latest_coastline = _load_geojson(_latest_coastline_path(aoi_id))
    state_df = _load_state_timeseries(aoi_id)

    last_obs = None
    if not state_df.empty and "datetime" in state_df.columns:
        last_dt = pd.to_datetime(state_df["datetime"], utc=True, errors="coerce").max()
        if pd.notna(last_dt):
            last_obs = last_dt.isoformat()

    if "polygon_latlng" in aoi and isinstance(aoi["polygon_latlng"], list):
        polygon_latlng = aoi["polygon_latlng"]
    else:
        coords = list(aoi["polygon"].exterior.coords) if aoi.get("polygon") is not None else []
        polygon_latlng = [[float(lat), float(lon)] for lon, lat in coords[:-1]] if len(coords) > 1 else []

    return {
        "aoi_id": aoi_id,
        "aoi_name": aoi.get("name"),
        "bbox_wgs84": aoi["bbox_wgs84"],
        "utm_epsg": int(aoi["utm_epsg"]),
        "is_fixed_model_aoi": bool(aoi.get("is_fixed_model_aoi", False)),
        "polygon_latlng": polygon_latlng,
        "latest_waterline_geojson": latest_coastline,
        "state": {
            "timeseries_rows": int(len(state_df)),
            "last_observation_datetime": last_obs,
            "last_assimilation_at": metadata.get("last_assimilation_at"),
            "last_retrained_at": metadata.get("last_retrained_at"),
            "active_model_run_id": metadata.get("active_model_run_id"),
        },
    }


def _phase(phase_callback, job_id: str | None, phase: str, state: str, message: str) -> None:
    if phase_callback is None or not job_id:
        return
    phase_callback(job_id, phase, state, message)


def _pick_new_scene_ids(aoi_id: str, scenes: list[dict[str, Any]]) -> list[str]:
    existing = _load_state_timeseries(aoi_id)
    existing_scene_ids: set[str] = set()
    if not existing.empty and "scene_id" in existing.columns:
        existing_scene_ids = set(existing["scene_id"].dropna().astype(str).unique().tolist())

    ranked = sorted(scenes, key=lambda s: s.get("datetime", ""))
    return [str(scene["scene_id"]) for scene in ranked if str(scene.get("scene_id")) not in existing_scene_ids]


def _update_latest_waterline(aoi_id: str, run_id: str) -> None:
    waterlines_path = Path("data") / "runs" / run_id / "exports" / "waterlines.geojson"
    payload = _load_geojson(waterlines_path)
    features = payload.get("features", [])
    if not features:
        return

    def _dt_for_feature(feature: dict[str, Any]) -> datetime:
        props = feature.get("properties", {})
        dt = _parse_iso(str(props.get("datetime")))
        return dt if dt is not None else datetime.min.replace(tzinfo=timezone.utc)

    latest_feature = max(features, key=_dt_for_feature)
    _save_geojson(_latest_coastline_path(aoi_id), {"type": "FeatureCollection", "features": [latest_feature]})


def _append_run_state(aoi_id: str, run_id: str) -> tuple[int, str | None]:
    run_dir = Path("data") / "runs" / run_id
    run_ts_path = run_dir / "results" / "timeseries.csv"
    if not run_ts_path.exists():
        return 0, None

    new_df = pd.read_csv(run_ts_path)
    if new_df.empty:
        return 0, None

    new_df["datetime"] = pd.to_datetime(new_df["datetime"], utc=True, errors="coerce")
    old_df = _load_state_timeseries(aoi_id)

    merged = pd.concat([old_df, new_df], ignore_index=True)
    if "scene_id" in merged.columns and "transect_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["scene_id", "transect_id"], keep="last")
    merged = merged.sort_values(["datetime", "transect_id"], kind="stable").reset_index(drop=True)
    _save_state_timeseries(aoi_id, merged)

    # Keep a stable transect geometry in state DB (first one wins).
    state_transects = _transects_path(aoi_id)
    run_transects = run_dir / "results" / "transects.geojson"
    if not state_transects.exists() and run_transects.exists():
        state_transects.write_text(run_transects.read_text(encoding="utf-8"), encoding="utf-8")

    _update_latest_waterline(aoi_id, run_id)

    last_scene_dt = None
    max_dt = new_df["datetime"].max()
    if pd.notna(max_dt):
        last_scene_dt = max_dt.isoformat()
    return int(len(merged)), last_scene_dt


def _maybe_retrain_model(
    aoi_id: str,
    metadata: dict[str, Any],
    sequence_len_days: int,
    model_preference: str,
    retrain_interval_days: int,
    min_samples_for_retrain: int,
) -> tuple[bool, str | None]:
    state_df = _load_state_timeseries(aoi_id)
    if len(state_df) < int(min_samples_for_retrain):
        return False, "insufficient_samples"

    last_retrained = _parse_iso(metadata.get("last_retrained_at"))
    now = datetime.now(timezone.utc)
    if last_retrained is not None and (now - last_retrained) < timedelta(days=int(retrain_interval_days)):
        return False, "not_due"

    transects_path = _transects_path(aoi_id)
    if not transects_path.exists():
        return False, "missing_transects"

    # Build a synthetic run from DT state to periodically retrain model weights.
    run_id = f"dt_retrain_{aoi_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("data") / "runs" / run_id
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    state_df.to_csv(results_dir / "timeseries.csv", index=False)
    (results_dir / "transects.geojson").write_text(transects_path.read_text(encoding="utf-8"), encoding="utf-8")

    bootstrap = get_digital_twin_bootstrap(aoi_id)
    summary = {
        "run_id": run_id,
        "aoi_id": aoi_id,
        "generated_at": _now_iso(),
        "utm_epsg": int(bootstrap["utm_epsg"]),
        "source": "digital_twin_state",
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    run_prediction(
        run_id=run_id,
        sequence_len_days=max(2, int(sequence_len_days)),
        forecast_days=30,
        model_preference=model_preference,
        allow_training=True,
    )
    metadata["last_retrained_at"] = _now_iso()
    metadata["active_model_run_id"] = run_id
    return True, run_id


def run_assimilation_cycle(
    sentinel_hub_available: bool,
    aoi_id: str | None = None,
    max_cloud_pct: int = 30,
    max_images: int = 6,
    lookback_days: int = 45,
    sequence_len_days: int = 30,
    model_preference: str = "mamba_lstm",
    retrain_interval_days: int = 180,
    min_samples_for_retrain: int = 200,
    job_id: str | None = None,
    phase_callback=None,
) -> dict[str, Any]:
    """Async DT heartbeat: poll -> segment/extract -> append state -> periodic retrain."""
    aoi = load_aoi(aoi_id) if aoi_id else create_or_get_fixed_aoi()
    aoi_id = str(aoi["aoi_id"])
    metadata = _load_metadata(aoi_id)

    _phase(phase_callback, job_id, "poll", "running", "Polling Sentinel-2 for new AOI imagery")
    last_scene_dt = _parse_iso(metadata.get("last_scene_datetime"))
    end_date = datetime.utcnow().date()
    if last_scene_dt is not None:
        start_date = (last_scene_dt + timedelta(days=1)).date()
    else:
        start_date = end_date - timedelta(days=max(1, int(lookback_days)))

    if start_date > end_date:
        _phase(phase_callback, job_id, "poll", "completed", "No new time window to poll")
        return {
            "aoi_id": aoi_id,
            "status": "no_update",
            "reason": "up_to_date",
            "last_scene_datetime": metadata.get("last_scene_datetime"),
        }

    scenes = search_scenes(
        aoi_id=aoi_id,
        bbox_wgs84=aoi["bbox_wgs84"],
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        max_cloud_pct=int(max_cloud_pct),
        max_images=max(1, int(max_images)),
        sentinel_hub_available=sentinel_hub_available,
    )
    scene_ids = _pick_new_scene_ids(aoi_id, scenes)
    if not scene_ids:
        metadata["last_assimilation_at"] = _now_iso()
        _save_metadata(aoi_id, metadata)
        _phase(phase_callback, job_id, "poll", "completed", "No new scenes after deduplication")
        return {
            "aoi_id": aoi_id,
            "status": "no_update",
            "reason": "no_new_scenes",
            "searched_scene_count": len(scenes),
        }
    _phase(phase_callback, job_id, "poll", "completed", f"Found {len(scene_ids)} new scenes")

    # Reuse extraction pipeline as segmentation + intersection engine.
    extraction_result = execute_extraction_job(
        job_id=job_id or "assimilation",
        phase_callback=phase_callback or (lambda *_args, **_kwargs: None),
        aoi_id=aoi_id,
        scene_ids=scene_ids,
        sentinel_hub_available=sentinel_hub_available,
        transect_spacing_m=float(DEFAULT_PARAMS["transect_spacing_m"]),
        transect_length_m=float(DEFAULT_PARAMS["transect_length_m"]),
        offshore_ratio=float(DEFAULT_PARAMS["offshore_ratio"]),
        max_dist_ref_m=float(DEFAULT_PARAMS["max_dist_ref_m"]),
    )
    extracted_run_id = extraction_result["run_id"]

    _phase(phase_callback, job_id, "state_update", "running", "Appending extracted distances into DT state DB")
    total_rows, newest_scene_dt = _append_run_state(aoi_id, extracted_run_id)
    metadata["last_assimilation_at"] = _now_iso()
    metadata["last_scene_datetime"] = newest_scene_dt or metadata.get("last_scene_datetime")
    metadata["last_run_id"] = extracted_run_id
    metadata["timeseries_rows"] = int(total_rows)
    _save_metadata(aoi_id, metadata)
    _phase(phase_callback, job_id, "state_update", "completed", f"State updated ({total_rows} rows)")

    _phase(phase_callback, job_id, "retrain_check", "running", "Checking periodic retraining policy")
    retrained, retrain_ref = _maybe_retrain_model(
        aoi_id=aoi_id,
        metadata=metadata,
        sequence_len_days=sequence_len_days,
        model_preference=model_preference,
        retrain_interval_days=retrain_interval_days,
        min_samples_for_retrain=min_samples_for_retrain,
    )
    _save_metadata(aoi_id, metadata)
    if retrained:
        _phase(phase_callback, job_id, "retrain_check", "completed", f"Retrained model: {retrain_ref}")
    else:
        _phase(phase_callback, job_id, "retrain_check", "completed", f"Retrain skipped: {retrain_ref}")

    return {
        "aoi_id": aoi_id,
        "status": "updated",
        "new_scene_count": len(scene_ids),
        "extraction_run_id": extracted_run_id,
        "timeseries_rows": int(total_rows),
        "retrained": bool(retrained),
        "retrain_ref": retrain_ref,
    }


def _resolve_warm_start_checkpoint(metadata: dict[str, Any]) -> str | None:
    model_run_id = metadata.get("active_model_run_id")
    if model_run_id:
        path = Path("data") / "runs" / str(model_run_id) / "predictions" / "mamba_lstm_finetuned.pt"
        if path.exists():
            return str(path)
    resolved = resolve_mamba_checkpoint_path()
    return str(resolved) if resolved is not None else None


def predict_from_digital_twin_state(
    forecast_years: float,
    aoi_id: str | None = None,
    sequence_len_days: int = 30,
    lookback_days: int = 730,
    model_preference: str = "mamba_lstm",
) -> dict[str, Any]:
    """Low-latency sync inference using DT state without on-request retraining."""
    if forecast_years <= 0:
        raise ValueError("forecast_years must be positive")

    aoi = load_aoi(aoi_id) if aoi_id else create_or_get_fixed_aoi()
    aoi_id = str(aoi["aoi_id"])
    _seed_state_from_latest_run_if_missing(aoi_id)
    state_df = _load_state_timeseries(aoi_id)
    if state_df.empty:
        raise ValueError("Digital Twin state has no timeseries data yet. Run assimilation first.")

    state_df["datetime"] = pd.to_datetime(state_df["datetime"], utc=True, errors="coerce")
    now_ts = pd.Timestamp.now(tz="UTC")
    cutoff = now_ts - pd.Timedelta(days=max(1, int(lookback_days)))
    recent_df = state_df[state_df["datetime"] >= cutoff].copy()
    if recent_df.empty:
        recent_df = state_df.copy()

    transects_path = _transects_path(aoi_id)
    if not transects_path.exists():
        raise FileNotFoundError("Digital Twin state is missing transects geometry")

    infer_run_id = f"dt_infer_{aoi_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("data") / "runs" / infer_run_id
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    recent_df.to_csv(results_dir / "timeseries.csv", index=False)
    (results_dir / "transects.geojson").write_text(transects_path.read_text(encoding="utf-8"), encoding="utf-8")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"run_id": infer_run_id, "utm_epsg": int(aoi["utm_epsg"]), "aoi_id": aoi_id, "source": "digital_twin_inference"}, f, indent=2)

    latest_waterline = _load_geojson(_latest_coastline_path(aoi_id))
    forecast_days = max(1, int(round(float(forecast_years) * 365.25)))
    model_pref = str(model_preference).strip().lower()
    wants_mamba_shape = model_pref in {"mamba", "mamba_lstm", "auto"}
    fallback_warning: str | None = None
    if wants_mamba_shape:
        try:
            artifacts = run_mamba_coastline_prediction(
                run_id=infer_run_id,
                aoi_id=aoi_id,
                aoi_bbox_wgs84=aoi["bbox_wgs84"],
                latest_waterline_geojson=latest_waterline,
                forecast_days=forecast_days,
                checkpoint_path=None,
                history_len=5,
                n_points=256,
                lookback_days=max(30, int(lookback_days)),
                return_all_steps=False,
            )
            forecast_geojson = _load_geojson(run_dir / "predictions" / "shoreline_forecast.geojson")
            return {
                "aoi_id": aoi_id,
                "run_id": infer_run_id,
                "forecast_years": float(forecast_years),
                "forecast_days": forecast_days,
                "model_type": artifacts.summary.get("model_type"),
                "summary": artifacts.summary,
                "latest_waterline_geojson": latest_waterline,
                "forecast_geojson": forecast_geojson,
            }
        except Exception as exc:
            if model_pref in {"mamba", "mamba_lstm"}:
                raise ValueError(f"MambaLSTM coastline prediction failed: {exc}") from exc
            # `auto` may still degrade to the legacy predictor.
            fallback_warning = f"mamba_coastline_fallback:{exc}"

    metadata = _load_metadata(aoi_id)
    artifacts = run_prediction(
        run_id=infer_run_id,
        train_split_date=None,
        sequence_len_days=max(2, int(sequence_len_days)),
        forecast_days=forecast_days,
        model_preference=model_preference,
        allow_training=False,
        warm_start_model_path=_resolve_warm_start_checkpoint(metadata),
    )

    if fallback_warning:
        artifacts.summary["warning"] = fallback_warning

    forecast_geojson = _load_geojson(run_dir / "predictions" / "shoreline_forecast.geojson")
    return {
        "aoi_id": aoi_id,
        "run_id": infer_run_id,
        "forecast_years": float(forecast_years),
        "forecast_days": forecast_days,
        "model_type": artifacts.summary.get("model_type"),
        "summary": artifacts.summary,
        "latest_waterline_geojson": latest_waterline,
        "forecast_geojson": forecast_geojson,
    }


_scheduler_started = False
_scheduler_lock = threading.Lock()


def start_digital_twin_scheduler(job_manager, sentinel_hub_available: bool) -> None:
    """Start background heartbeat scheduler for periodic assimilation checks."""
    global _scheduler_started
    with _scheduler_lock:
        if _scheduler_started:
            return
        _scheduler_started = True

    enabled = os.getenv("TERRA_DT_ENABLE_SCHEDULER", "1").strip().lower() not in {"0", "false", "no"}
    if not enabled:
        return

    interval_days = max(1, int(os.getenv("TERRA_DT_ASSIMILATION_INTERVAL_DAYS", "5")))
    interval_seconds = interval_days * 24 * 60 * 60

    def _loop():
        next_run = time.time() + interval_seconds
        while True:
            now = time.time()
            sleep_for = max(5.0, min(300.0, next_run - now))
            time.sleep(sleep_for)
            if time.time() < next_run:
                continue
            next_run = time.time() + interval_seconds
            try:
                if hasattr(job_manager, "has_active_job") and job_manager.has_active_job("assimilate"):
                    continue
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
                job_manager.submit_job(
                    job_type="assimilate",
                    phases=phases,
                    fn=lambda **kwargs: run_assimilation_cycle(
                        sentinel_hub_available=sentinel_hub_available,
                        aoi_id=None,
                        job_id=kwargs.get("job_id"),
                        phase_callback=kwargs.get("phase_callback"),
                    ),
                )
            except Exception:
                continue

    thread = threading.Thread(target=_loop, name="terra-dt-scheduler", daemon=True)
    thread.start()
