import json

import pandas as pd
import pytest

pytest.importorskip("pyproj")
pytest.importorskip("shapely")


def _write_dt_state(tmp_path):
    dt_state = tmp_path / "data" / "dt" / "fixed_model_aoi" / "state"
    dt_state.mkdir(parents=True, exist_ok=True)

    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for transect_id in [0, 1]:
        for day in range(40):
            rows.append(
                {
                    "run_id": "seed",
                    "scene_id": f"s_{day}",
                    "datetime": (base + pd.Timedelta(days=day)).isoformat(),
                    "transect_id": transect_id,
                    "VE_distance_m": 2.0 + (0.2 * day) + transect_id,
                    "WL_distance_m": 1.0 + (0.1 * day) + transect_id,
                    "wl_lon": -2.85,
                    "wl_lat": 56.36,
                    "ve_lon": -2.84,
                    "ve_lat": 56.36,
                }
            )
    pd.DataFrame(rows).to_csv(dt_state / "timeseries.csv", index=False)

    transects = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"transect_id": 0},
                "geometry": {"type": "LineString", "coordinates": [[-2.86, 56.35], [-2.86, 56.37]]},
            },
            {
                "type": "Feature",
                "properties": {"transect_id": 1},
                "geometry": {"type": "LineString", "coordinates": [[-2.85, 56.35], [-2.85, 56.37]]},
            },
        ],
    }
    (dt_state / "transects.geojson").write_text(json.dumps(transects), encoding="utf-8")

    latest = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"datetime": "2025-02-10T00:00:00Z", "boundary_type": "waterline"},
                "geometry": {"type": "LineString", "coordinates": [[-2.86, 56.36], [-2.85, 56.36]]},
            }
        ],
    }
    (dt_state / "latest_waterline.geojson").write_text(json.dumps(latest), encoding="utf-8")

    metadata = {
        "aoi_id": "fixed_model_aoi",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "last_assimilation_at": "2025-02-10T00:00:00Z",
        "last_scene_datetime": "2025-02-10T00:00:00Z",
        "last_run_id": "seed",
        "last_retrained_at": None,
        "active_model_run_id": None,
        "timeseries_rows": 80,
    }
    (dt_state / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_digital_twin_bootstrap_and_predict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_dt_state(tmp_path)

    from src.terra_ugla.app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        boot = client.get("/dt/bootstrap")
        assert boot.status_code == 200
        payload = boot.get_json()
        assert payload["success"] is True
        assert payload["aoi_id"] == "fixed_model_aoi"
        assert payload["state"]["timeseries_rows"] > 0

        pred = client.post(
            "/dt/predict",
            json={
                "forecast_years": 1,
                "aoi_id": "fixed_model_aoi",
                "sequence_len_days": 10,
                "lookback_days": 730,
                "model_preference": "mamba_lstm",
            },
        )
        assert pred.status_code == 200
        pred_payload = pred.get_json()
        assert pred_payload["success"] is True
        assert "run_id" in pred_payload
        assert pred_payload["forecast_geojson"]["type"] == "FeatureCollection"
