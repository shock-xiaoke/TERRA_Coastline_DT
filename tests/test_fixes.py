import pytest


pytest.importorskip("shapely")

from shapely.geometry import LineString

from src.terra_ugla.services.intersections import generate_transects_from_baseline


def test_transects_generated_in_fixed_count():
    baseline = LineString([(-2.90, 56.35), (-2.80, 56.35)])
    transects = generate_transects_from_baseline(
        baseline_line_wgs84=baseline,
        utm_epsg=32630,
        spacing_m=100,
        length_m=500,
        offshore_ratio=0.7,
    )

    assert len(transects) > 2
    assert transects[0].line_utm.length == pytest.approx(500.0, rel=0.1)
