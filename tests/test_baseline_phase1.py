import pytest


pytest.importorskip("shapely")


def test_create_baseline_and_generate_transects(client):
    baseline_resp = client.post(
        "/baseline",
        json={
            "name": "st_andrews_baseline",
            "line_latlng": [
                [56.3715, -2.8025],
                [56.3692, -2.7950],
                [56.3658, -2.7852],
            ],
        },
    )
    assert baseline_resp.status_code == 200
    baseline_payload = baseline_resp.get_json()
    assert baseline_payload["success"] is True
    baseline_id = baseline_payload["baseline_id"]

    transect_resp = client.post(
        f"/baseline/{baseline_id}/transects",
        json={
            "spacing_m": 20,
            "transect_length_m": 500,
            "offshore_ratio": 0.7,
        },
    )
    assert transect_resp.status_code == 200
    transect_payload = transect_resp.get_json()
    assert transect_payload["success"] is True
    assert transect_payload["transect_count"] > 0

    geojson = transect_payload["geojson"]
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == transect_payload["transect_count"]
    assert "offshore_ratio" in transect_payload
    first_props = geojson["features"][0]["properties"]
    assert first_props["transect_id"].startswith("T")
    assert "distance_along_baseline" in first_props
    assert "offshore_ratio" in first_props


def test_sharp_baseline_smoothing_reduces_bowtie_crossing(client):
    from shapely.geometry import shape

    baseline_resp = client.post(
        "/baseline",
        json={
            "name": "sharp_turn_baseline",
            "line_latlng": [
                [56.3680, -2.8120],
                [56.3662, -2.8045],
                [56.3684, -2.7990],
                [56.3668, -2.7930],
            ],
        },
    )
    assert baseline_resp.status_code == 200
    baseline_id = baseline_resp.get_json()["baseline_id"]

    transect_resp = client.post(
        f"/baseline/{baseline_id}/transects",
        json={
            "spacing_m": 20,
            "transect_length_m": 220,
            "offshore_ratio": 0.7,
        },
    )
    assert transect_resp.status_code == 200
    payload = transect_resp.get_json()
    assert payload["success"] is True
    assert payload["transect_count"] > 3

    lines = [shape(feature["geometry"]) for feature in payload["geojson"]["features"]]
    # Adjacent transects should not cross after smoothing on this sharp baseline.
    for i in range(len(lines) - 1):
        assert not lines[i].crosses(lines[i + 1])
