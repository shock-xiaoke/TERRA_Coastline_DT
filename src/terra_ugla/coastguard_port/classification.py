"""Pixel classification helpers adapted from COASTGUARD VegetationLine routines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import morphology

from .indices import image_std, nd_index, rbnd_index, savi_index


def _flat_mask(cloud_mask: np.ndarray) -> np.ndarray:
    return cloud_mask.reshape(cloud_mask.shape[0] * cloud_mask.shape[1]).astype(bool)


def _calculate_veg_features(im_ms: np.ndarray, cloud_mask: np.ndarray, im_bool: np.ndarray) -> np.ndarray:
    features = np.expand_dims(im_ms[im_bool, 0], axis=1)
    for band in range(1, im_ms.shape[2]):
        features = np.append(features, np.expand_dims(im_ms[im_bool, band], axis=1), axis=-1)

    nir = im_ms[:, :, 3]
    red = im_ms[:, :, 2]
    green = im_ms[:, :, 1]
    blue = im_ms[:, :, 0]

    im_nirr = nd_index(nir, red, cloud_mask)
    im_nirg = nd_index(nir, green, cloud_mask)
    im_rg = nd_index(red, green, cloud_mask)
    im_savi = savi_index(nir, red, cloud_mask)
    im_rbndvi = rbnd_index(nir, red, blue, cloud_mask)

    for index_arr in [im_nirr, im_nirg, im_rg, im_savi, im_rbndvi]:
        features = np.append(features, np.expand_dims(index_arr[im_bool], axis=1), axis=-1)

    for band in range(im_ms.shape[2]):
        features = np.append(features, np.expand_dims(image_std(im_ms[:, :, band], 1)[im_bool], axis=1), axis=-1)

    for index_arr in [im_nirr, im_nirg, im_rg, im_savi, im_rbndvi]:
        features = np.append(features, np.expand_dims(image_std(index_arr, 1)[im_bool], axis=1), axis=-1)

    return features


def _calculate_shore_features(im_ms: np.ndarray, cloud_mask: np.ndarray, im_bool: np.ndarray) -> np.ndarray:
    features = np.expand_dims(im_ms[im_bool, 0], axis=1)
    for band in range(1, im_ms.shape[2]):
        features = np.append(features, np.expand_dims(im_ms[im_bool, band], axis=1), axis=-1)

    nir = im_ms[:, :, 3]
    red = im_ms[:, :, 2]
    green = im_ms[:, :, 1]
    blue = im_ms[:, :, 0]

    swir = im_ms[:, :, 4] if im_ms.shape[2] > 4 else im_ms[:, :, 3]

    im_swirg = nd_index(swir, green, cloud_mask)
    im_swirnir = nd_index(swir, nir, cloud_mask)
    im_nirg = nd_index(nir, green, cloud_mask)
    im_nirr = nd_index(nir, red, cloud_mask)
    im_br = nd_index(blue, red, cloud_mask)

    for index_arr in [im_swirg, im_swirnir, im_nirg, im_nirr, im_br]:
        features = np.append(features, np.expand_dims(index_arr[im_bool], axis=1), axis=-1)

    for band in range(im_ms.shape[2]):
        features = np.append(features, np.expand_dims(image_std(im_ms[:, :, band], 1)[im_bool], axis=1), axis=-1)

    for index_arr in [im_swirg, im_swirnir, im_nirg, im_nirr, im_br]:
        features = np.append(features, np.expand_dims(image_std(index_arr, 1)[im_bool], axis=1), axis=-1)

    return features


def _predict_with_model(model, features: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    vec_mask = _flat_mask(cloud_mask)
    vec_nan = np.any(np.isnan(features), axis=1)
    vec_inf = np.any(np.isinf(features), axis=1)
    invalid = np.logical_or(vec_mask, np.logical_or(vec_nan, vec_inf))
    vec = features.copy()
    vec[np.isnan(vec)] = 1e-9

    labels = model.predict(vec[~invalid, :])
    vec_classif = np.nan * np.ones(cloud_mask.shape[0] * cloud_mask.shape[1])
    vec_classif[~invalid] = labels
    return vec_classif.reshape(cloud_mask.shape)


def classify_image_nn(
    im_ms: np.ndarray,
    cloud_mask: np.ndarray,
    model=None,
    min_patch_size: int = 200,
    fallback_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Vegetation-vs-non vegetation classifier with model/fallback paths."""
    if model is not None:
        try:
            features = _calculate_veg_features(im_ms, cloud_mask, np.ones(cloud_mask.shape, dtype=bool))
            im_classif = _predict_with_model(model, features, cloud_mask)
            im_veg = im_classif == 1
            im_nonveg = im_classif == 2
        except Exception:
            model = None

    if model is None:
        ndvi = nd_index(im_ms[:, :, 3], im_ms[:, :, 2], cloud_mask)
        im_veg = np.logical_and(~cloud_mask, ndvi >= fallback_threshold)
        im_nonveg = np.logical_and(~cloud_mask, ndvi < fallback_threshold)
        im_classif = np.zeros(cloud_mask.shape, dtype=float)
        im_classif[im_nonveg] = 2
        im_classif[im_veg] = 1
        im_classif[cloud_mask] = np.nan

    im_veg = morphology.remove_small_objects(im_veg, min_size=max(1, min_patch_size), connectivity=2)
    im_nonveg = morphology.remove_small_objects(im_nonveg, min_size=max(1, min_patch_size), connectivity=2)
    im_labels = np.stack((im_veg, im_nonveg), axis=-1)
    return im_classif, im_labels


def classify_image_nn_shore(
    im_ms: np.ndarray,
    cloud_mask: np.ndarray,
    model=None,
    min_patch_size: int = 200,
    fallback_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Shoreline-oriented classifier (sand/swash/water)."""
    if model is not None:
        try:
            features = _calculate_shore_features(im_ms, cloud_mask, np.ones(cloud_mask.shape, dtype=bool))
            im_classif = _predict_with_model(model, features, cloud_mask)
            im_sand = im_classif == 1
            im_swash = im_classif == 2
            im_water = im_classif == 3
        except Exception:
            model = None

    if model is None:
        mndwi = nd_index(im_ms[:, :, 1], im_ms[:, :, 4] if im_ms.shape[2] > 4 else im_ms[:, :, 3], cloud_mask)
        ndvi = nd_index(im_ms[:, :, 3], im_ms[:, :, 2], cloud_mask)

        im_water = np.logical_and(~cloud_mask, mndwi > fallback_threshold)
        im_sand = np.logical_and(~cloud_mask, np.logical_and(~im_water, ndvi < 0.2))
        im_swash = np.logical_and(~cloud_mask, np.logical_and(~im_water, ~im_sand))

        im_classif = np.zeros(cloud_mask.shape, dtype=float)
        im_classif[im_sand] = 1
        im_classif[im_swash] = 2
        im_classif[im_water] = 3
        im_classif[cloud_mask] = np.nan

    im_sand = morphology.remove_small_objects(im_sand, min_size=max(1, min_patch_size), connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=max(1, min_patch_size), connectivity=2)
    im_labels = np.stack((im_sand, im_swash, im_water), axis=-1)
    return im_classif, im_labels


def load_classifier(model_path: str | Path):
    """Load a pickled sklearn model if available, otherwise None."""
    path = Path(model_path)
    if not path.exists():
        return None

    try:
        from joblib import load

        return load(path)
    except Exception:
        return None
