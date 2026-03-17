import json

import pytest


pytest.importorskip("shapely")


def test_download_geojson_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.terra_ugla.app import app

    run_id = "run_artifacts"
    run_dir = tmp_path / "data" / "runs" / run_id
    results_dir = run_dir / "results"
    exports_dir = run_dir / "exports"
    pred_dir = run_dir / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    empty_fc = {"type": "FeatureCollection", "features": []}
    (results_dir / "intersections.geojson").write_text(json.dumps(empty_fc), encoding="utf-8")
    (results_dir / "transects.geojson").write_text(json.dumps(empty_fc), encoding="utf-8")
    (exports_dir / "waterlines.geojson").write_text(json.dumps(empty_fc), encoding="utf-8")
    (exports_dir / "veglines.geojson").write_text(json.dumps(empty_fc), encoding="utf-8")
    (pred_dir / "shoreline_forecast.geojson").write_text(json.dumps(empty_fc), encoding="utf-8")

    app.config["TESTING"] = True
    with app.test_client() as client:
        for artifact in ["intersections", "transects", "waterlines", "veglines", "forecast_shorelines"]:
            resp = client.get(f"/results/{run_id}/download?type=geojson&artifact={artifact}")
            assert resp.status_code == 200
            payload = resp.get_json()
            assert payload["type"] == "FeatureCollection"

        bad = client.get(f"/results/{run_id}/download?type=geojson&artifact=bad_artifact")
        assert bad.status_code == 400
