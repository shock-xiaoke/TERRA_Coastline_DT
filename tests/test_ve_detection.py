import numpy as np
import pytest


pytest.importorskip("shapely")

from src.terra_ugla.coastguard_port.classification import classify_image_nn
from src.terra_ugla.coastguard_port.indices import nd_index


def test_nd_index_range():
    a = np.array([[0.6, 0.4], [0.2, 0.1]], dtype=float)
    b = np.array([[0.2, 0.2], [0.1, 0.1]], dtype=float)
    nd = nd_index(a, b, np.zeros_like(a, dtype=bool))
    assert np.nanmax(nd) <= 1.0
    assert np.nanmin(nd) >= -1.0


def test_classify_fallback_shapes():
    rows, cols = 20, 20
    im_ms = np.zeros((rows, cols, 5), dtype=float)
    im_ms[:, :, 2] = 0.2  # red
    im_ms[:, :, 3] = np.linspace(0.1, 0.8, rows).reshape(-1, 1)  # nir gradient
    cloud_mask = np.zeros((rows, cols), dtype=bool)

    im_classif, im_labels = classify_image_nn(im_ms, cloud_mask, model=None)
    assert im_classif.shape == (rows, cols)
    assert im_labels.shape == (rows, cols, 2)
