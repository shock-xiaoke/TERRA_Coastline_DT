"""Ported COASTGUARD-style algorithms used by TERRA extraction pipeline."""

from .indices import nd_index, savi_index, rbnd_index, image_std
from .classification import classify_image_nn, classify_image_nn_shore
from .contours import (
    find_weighted_peaks_threshold,
    find_contours_weighted_peaks,
    process_contours,
    contour_pixels_to_lines,
    pick_primary_line,
    tz_values,
)

__all__ = [
    "nd_index",
    "savi_index",
    "rbnd_index",
    "image_std",
    "classify_image_nn",
    "classify_image_nn_shore",
    "find_weighted_peaks_threshold",
    "find_contours_weighted_peaks",
    "process_contours",
    "contour_pixels_to_lines",
    "pick_primary_line",
    "tz_values",
]
