"""Contour extraction and thresholding routines adapted from COASTGUARD."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import rasterio
from shapely.geometry import LineString
from skimage import measure


def find_weighted_peaks_threshold(class_a: np.ndarray, class_b: np.ndarray) -> float:
    """Weighted-peaks threshold (0.2 lower mode + 0.8 upper mode)."""
    values_a = np.asarray(class_a).ravel()
    values_b = np.asarray(class_b).ravel()

    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) < 5 or len(values_b) < 5:
        if len(values_a) == 0 or len(values_b) == 0:
            return 0.0
        return float((np.nanmean(values_a) + np.nanmean(values_b)) / 2)

    try:
        from scipy import signal
        from sklearn.neighbors import KernelDensity

        all_values = np.concatenate([values_a, values_b]).reshape(-1, 1)
        if len(all_values) < 20:
            return float((np.nanmean(values_a) + np.nanmean(values_b)) / 2)

        kde = KernelDensity(kernel="gaussian", bandwidth=0.01)
        kde.fit(all_values)

        x_vals = np.linspace(np.nanmin(all_values), np.nanmax(all_values), 1000).reshape(-1, 1)
        probs = kde.score_samples(x_vals)
        peaks_idx, props = signal.find_peaks(probs, prominence=0.2)
        if len(peaks_idx) >= 2:
            prominences = props.get("prominences", np.zeros_like(peaks_idx, dtype=float))
            top_two = peaks_idx[np.argsort(prominences)[-2:]]
            peaks = np.sort(x_vals[top_two].flatten())
            return float((0.2 * peaks[0]) + (0.8 * peaks[1]))
    except Exception:
        pass

    return float((np.nanmean(values_a) + np.nanmean(values_b)) / 2)


def tz_values(class_a: np.ndarray, class_b: np.ndarray) -> tuple[float, float]:
    """Simple transition-zone bound estimate used for reporting."""
    values = np.asarray(class_a).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 0.0
    return float(np.percentile(values, 0.5)), float(np.percentile(values, 10))


def process_contours(contours: list[np.ndarray]) -> list[np.ndarray]:
    """Remove NaN points and short contours."""
    cleaned: list[np.ndarray] = []
    for contour in contours:
        if contour is None or len(contour) < 2:
            continue
        arr = np.asarray(contour)
        if np.isnan(arr).any():
            arr = arr[~np.isnan(arr).any(axis=1)]
        if len(arr) >= 3:
            cleaned.append(arr)
    return cleaned


def find_contours_weighted_peaks(
    index_arr: np.ndarray,
    class_a_mask: np.ndarray,
    class_b_mask: np.ndarray,
    ref_buffer_mask: np.ndarray,
    cloud_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], float]:
    """Find contours for an index using class-conditioned weighted peaks threshold."""
    if cloud_mask is None:
        cloud_mask = np.zeros(index_arr.shape, dtype=bool)

    valid = np.logical_and(ref_buffer_mask, ~cloud_mask)
    class_a_vals = index_arr[np.logical_and(valid, class_a_mask)]
    class_b_vals = index_arr[np.logical_and(valid, class_b_mask)]

    threshold = find_weighted_peaks_threshold(class_a_vals, class_b_vals)

    index_buffer = np.array(index_arr, copy=True)
    index_buffer[~ref_buffer_mask] = np.nan

    contours = measure.find_contours(index_buffer, threshold)
    contours = process_contours(contours)
    return contours, float(threshold)


def contour_pixels_to_lines(
    contours: Iterable[np.ndarray],
    transform,
    min_length: float = 10.0,
) -> list[LineString]:
    """Convert skimage contour pixel rows/cols to projected lines using raster transform."""
    lines: list[LineString] = []
    for contour in contours:
        if contour is None or len(contour) < 2:
            continue

        coords = []
        for row, col in contour:
            x, y = rasterio.transform.xy(transform, row, col)
            coords.append((float(x), float(y)))

        if len(coords) < 2:
            continue
        line = LineString(coords)
        if line.is_valid and line.length >= min_length:
            lines.append(line)

    return lines


def pick_primary_line(lines: list[LineString]) -> LineString | None:
    """Select the dominant line segment by length."""
    if not lines:
        return None
    return max(lines, key=lambda line: line.length)
