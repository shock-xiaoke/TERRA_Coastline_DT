import pytest


pytest.importorskip("shapely")

from src.terra_ugla.services.aoi import create_aoi


def test_invalid_aoi_polygon_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError):
        create_aoi(
            name="bad_aoi",
            polygon_latlng=[
                [10.0, 20.0],
                [10.1, 20.1],
            ],
            close_polygon=True,
        )


def test_self_intersection_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError):
        create_aoi(
            name="self_intersecting",
            polygon_latlng=[
                [0.0, 0.0],
                [0.01, 0.01],
                [0.01, 0.0],
                [0.0, 0.01],
            ],
            close_polygon=True,
        )
