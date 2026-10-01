"""Otsu-on-HSV tissue detection for histopathology WSIs.

Background pixels (white slide / glass) are near-grey with low saturation,
while H&E-stained tissue has high saturation. Otsu thresholding on the HSV
saturation channel reliably separates tissue from background without any
slide-specific tuning.

Used by :mod:`src.preprocessing.wsi_tiler` to skip empty tile positions
and by :mod:`src.preprocessing.quality_filter` to compute per-tile
``tissue_ratio``.

References:
    Design: docs/02-design/03-architecture.md §2 (Module A — WSI Preprocessing)
    Spec:   docs/02-design/02-data-spec.md §6.3 (tissue_ratio bounds)
"""

from __future__ import annotations

import numpy as np

try:  # OpenCV is the canonical implementation; scikit-image is the fallback.
    import cv2 as _cv2
except ImportError:  # pragma: no cover — covered by ImportError branch in tests.
    _cv2 = None


def _rgb_to_saturation(rgb: np.ndarray) -> np.ndarray:
    """Return the HSV saturation channel (uint8) for an RGB uint8 image."""
    if _cv2 is not None:
        hsv = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2HSV)
        return hsv[..., 1]
    # Manual fallback (scikit-image's rgb2hsv yields float in [0, 1]).
    from skimage.color import rgb2hsv

    sat_float = rgb2hsv(rgb)[..., 1]
    return (sat_float * 255.0).astype(np.uint8)


def _otsu_threshold(values: np.ndarray) -> int:
    """Compute Otsu threshold on a uint8 array; returns the threshold value."""
    if _cv2 is not None:
        thresh, _ = _cv2.threshold(values, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
        return int(thresh)
    from skimage.filters import threshold_otsu

    return int(threshold_otsu(values))


def detect_tissue(
    rgb: np.ndarray,
    *,
    min_saturation: int = 10,
) -> np.ndarray:
    """Return a boolean tissue mask for an RGB uint8 image.

    Procedure:

    1. Convert to HSV; take the saturation channel.
    2. **Unimodal-low short-circuit**: if the entire image is below
       ``min_saturation`` (essentially blank slide), return all-False.
       Without this, Otsu on a near-uniform histogram falsely flags noise.
    3. **Unimodal-high short-circuit**: if every pixel has saturation
       ``≥ 2 * min_saturation``, the image is uniformly stained tissue
       and Otsu would split *inside* that cluster (returning ~50% True
       for what is actually all tissue). Return all-True.
    4. Otherwise the image is bimodal (tissue + background) — Otsu's
       threshold separates them. We floor it at ``min_saturation`` so a
       very faint background lobe never pulls the threshold below the
       JPEG noise floor.

    Args:
        rgb: Input image, shape ``(H, W, 3)``, dtype ``uint8``.
        min_saturation: Saturation level (0–255) below which the image is
            treated as fully blank. The default of 10 matches the typical
            JPEG noise floor on white-balanced slide scans.

    Returns:
        Boolean mask of shape ``(H, W)``. ``True`` = tissue, ``False`` = background.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"detect_tissue expects (H, W, 3) RGB; got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"detect_tissue expects uint8; got {rgb.dtype}")

    saturation = _rgb_to_saturation(rgb)

    if int(saturation.max()) < min_saturation:
        return np.zeros(saturation.shape, dtype=bool)
    if int(saturation.min()) >= 2 * min_saturation:
        return np.ones(saturation.shape, dtype=bool)

    threshold = max(_otsu_threshold(saturation), min_saturation)
    return saturation > threshold


def tissue_ratio(rgb: np.ndarray, **kwargs) -> float:
    """Convenience wrapper: fraction of pixels classified as tissue."""
    mask = detect_tissue(rgb, **kwargs)
    return float(mask.mean())
