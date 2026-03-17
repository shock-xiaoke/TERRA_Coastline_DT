import os
from pathlib import Path

import pytest

pytest.importorskip("sentinelhub")

from src.terra_ugla.config import create_data_directories


def test_create_data_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_data_directories()

    expected = [
        "data/shorelines",
        "data/aoi",
        "data/transects",
        "data/satellite_images",
        "data/runs",
        "data/models",
        "data/exports",
    ]
    for rel in expected:
        assert (tmp_path / rel).exists(), rel
