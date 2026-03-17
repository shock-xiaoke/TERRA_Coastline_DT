import pytest

pytest.importorskip("pyproj")

from src.terra_ugla.services.imagery import search_scenes


def test_scene_search_demo_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    scenes = search_scenes(
        aoi_id="aoi_1",
        bbox_wgs84=[-2.9, 56.3, -2.8, 56.4],
        start_date="2024-01-01",
        end_date="2024-01-31",
        max_cloud_pct=20,
        max_images=6,
        sentinel_hub_available=False,
    )

    assert len(scenes) > 0
    assert "scene_id" in scenes[0]
    assert "datetime" in scenes[0]
