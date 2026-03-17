import pytest


pytest.importorskip("shapely")


def test_create_aoi_endpoint(client):
    response = client.post(
        "/aoi",
        json={
            "name": "test_aoi",
            "polygon_latlng": [
                [56.3600, -2.8600],
                [56.3600, -2.8500],
                [56.3700, -2.8500],
                [56.3700, -2.8600],
            ],
            "close_polygon": True,
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert "aoi_id" in payload


def test_legacy_endpoint_still_responds(client):
    response = client.post(
        "/save_shoreline",
        json={
            "name": "legacy_line",
            "coordinates": [[56.36, -2.86], [56.37, -2.85]],
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["deprecated"] is True


def test_home_page_is_dt_only(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Coastline Digital Twin and Prediction System" in response.data
    assert b"Predict Future Coastline" in response.data
    assert b"Draw AOI" not in response.data
    assert b"Search Scenes" not in response.data
    assert b"/baseline-workflow" not in response.data


def test_baseline_workflow_page(client):
    response = client.get("/baseline-workflow")
    assert response.status_code == 200
    assert b"Baseline and Fixed Transects" in response.data
