import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyproj")

from src.terra_ugla.services.prediction import run_prediction


def test_prediction_generates_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    run_id = "testrun"
    run_results = tmp_path / "data" / "runs" / run_id / "results"
    run_root = tmp_path / "data" / "runs" / run_id
    run_results.mkdir(parents=True, exist_ok=True)

    rows = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for tr in [0, 1]:
        for d in range(25):
            rows.append(
                {
                    "run_id": run_id,
                    "scene_id": f"s{d}",
                    "datetime": (base + pd.Timedelta(days=d)).isoformat(),
                    "transect_id": tr,
                    "VE_distance_m": 5.0 + tr + (0.1 * d),
                    "WL_distance_m": 2.0 + tr + (0.05 * d),
                    "wl_lon": -2.85,
                    "wl_lat": 56.36,
                    "ve_lon": -2.84,
                    "ve_lat": 56.36,
                }
            )

    pd.DataFrame(rows).to_csv(run_results / "timeseries.csv", index=False)
    (run_root / "summary.json").write_text('{"run_id":"testrun","utm_epsg":32630}', encoding="utf-8")
    (run_results / "transects.geojson").write_text(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"transect_id": 0},
      "geometry": {"type": "LineString", "coordinates": [[-2.86, 56.35], [-2.86, 56.36]]}
    },
    {
      "type": "Feature",
      "properties": {"transect_id": 1},
      "geometry": {"type": "LineString", "coordinates": [[-2.85, 56.35], [-2.85, 56.36]]}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    artifacts = run_prediction(run_id=run_id, sequence_len_days=5, forecast_days=7)

    assert artifacts.summary["run_id"] == run_id
    assert len(artifacts.forecast_df) > 0

    pred_dir = tmp_path / "data" / "runs" / run_id / "predictions"
    assert (pred_dir / "summary.json").exists()
    assert (pred_dir / "forecast.csv").exists()
    assert (pred_dir / "shoreline_forecast.geojson").exists()
