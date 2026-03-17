"""Core spectral/index operations adapted from COASTGUARD ToolBox/Veg routines."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def _as_mask(cloud_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if cloud_mask is None:
        return np.zeros(shape, dtype=bool)
    return cloud_mask.astype(bool)


def nd_index(im1: np.ndarray, im2: np.ndarray, cloud_mask: np.ndarray | None = None) -> np.ndarray:
    """Normalized difference index with cloud masking."""
    mask = _as_mask(cloud_mask, im1.shape)
    out = np.full(im1.shape, np.nan, dtype=float)

    valid = ~mask
    num = im1[valid].astype(float) - im2[valid].astype(float)
    den = im1[valid].astype(float) + im2[valid].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid] = np.where(den != 0, num / den, np.nan)
    return out


def savi_index(nir: np.ndarray, red: np.ndarray, cloud_mask: np.ndarray | None = None) -> np.ndarray:
    """Soil-adjusted vegetation index (L=0.5)."""
    mask = _as_mask(cloud_mask, nir.shape)
    out = np.full(nir.shape, np.nan, dtype=float)

    valid = ~mask
    with np.errstate(divide="ignore", invalid="ignore"):
        den = nir[valid] + red[valid] + 0.5
        out[valid] = np.where(den != 0, ((nir[valid] - red[valid]) / den) * 1.5, np.nan)
    return out


def rbnd_index(nir: np.ndarray, red: np.ndarray, blue: np.ndarray, cloud_mask: np.ndarray | None = None) -> np.ndarray:
    """RB-NDVI as used by COASTGUARD vegetation classification."""
    mask = _as_mask(cloud_mask, nir.shape)
    out = np.full(nir.shape, np.nan, dtype=float)

    valid = ~mask
    red_blue = red[valid] + blue[valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        den = nir[valid] + red_blue
        out[valid] = np.where(den != 0, (nir[valid] - red_blue) / den, np.nan)
    return out


def image_std(image: np.ndarray, radius: int) -> np.ndarray:
    """Moving-window standard deviation with reflected edges."""
    if radius < 1:
        return np.zeros_like(image, dtype=float)

    arr = image.astype(float)
    win = (radius * 2) + 1
    mean = uniform_filter(arr, size=win, mode="reflect")
    sq_mean = uniform_filter(arr * arr, size=win, mode="reflect")
    var = np.clip(sq_mean - (mean * mean), a_min=0.0, a_max=None)
    std = np.sqrt(var)
    std[np.isnan(arr)] = np.nan
    return std
