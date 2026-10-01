"""Per-tile quality filter — drops blurry or near-empty tiles.

Two metrics are computed for every candidate tile:

* ``tissue_ratio`` — fraction of pixels classified as tissue by
  :func:`src.preprocessing.tissue_detector.detect_tissue`. Tiles below
  ``tissue_threshold`` (default 0.5) are background-dominated and dropped.
* ``laplacian_var`` — variance of the Laplacian of the grayscale tile.
  This is the standard sharpness proxy from Pech-Pacheco et al. (2000).
  Tiles below ``blur_threshold`` (default 100) are out-of-focus.

Both metrics are persisted in ``coords.csv`` so that downstream stages
(and the data-spec verifier in `02-data-spec.md` §6.3) can audit which
tiles survived which filter.

References:
    Design: docs/02-design/03-architecture.md §2 / §2.A
    Spec:   docs/02-design/02-data-spec.md §6.3
    Config: configs/default.yaml `preprocessing.tissue_threshold`, `preprocessing.blur_threshold`
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.preprocessing.tissue_detector import detect_tissue

try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None


@dataclass(frozen=True)
class TileQualityMetrics:
    """Numeric per-tile metrics persisted to ``coords.csv``."""

    tissue_ratio: float
    laplacian_var: float


def _laplacian_variance(rgb: np.ndarray) -> float:
    """Variance of the Laplacian on a grayscale tile (focus measure)."""
    if _cv2 is not None:
        gray = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
        lap = _cv2.Laplacian(gray, _cv2.CV_64F)
        return float(lap.var())
    # Fallback: scikit-image's laplace returns float64 directly.
    from skimage.color import rgb2gray
    from skimage.filters import laplace

    gray = rgb2gray(rgb)
    return float(laplace(gray).var())


class TileQualityFilter:
    """Decide whether a tile should be kept based on tissue ratio and sharpness.

    Construct once with the project's thresholds (typically pulled from
    ``configs/default.yaml`` ``preprocessing:`` block) and call
    :meth:`evaluate` per tile. Use :meth:`passes` for a yes/no answer.
    """

    def __init__(
        self,
        *,
        tissue_threshold: float = 0.5,
        blur_threshold: float = 100.0,
    ) -> None:
        if not 0.0 <= tissue_threshold <= 1.0:
            raise ValueError(
                f"tissue_threshold must be in [0, 1]; got {tissue_threshold}"
            )
        if blur_threshold < 0:
            raise ValueError(f"blur_threshold must be >= 0; got {blur_threshold}")
        self.tissue_threshold = float(tissue_threshold)
        self.blur_threshold = float(blur_threshold)

    def evaluate(self, tile_rgb: np.ndarray) -> TileQualityMetrics:
        """Return both metrics without making a keep/drop decision.

        Useful for the orchestrator, which needs the numeric values for
        ``coords.csv`` regardless of whether the tile is kept.
        """
        ratio = float(detect_tissue(tile_rgb).mean())
        lap_var = _laplacian_variance(tile_rgb)
        return TileQualityMetrics(tissue_ratio=ratio, laplacian_var=lap_var)

    def passes(self, tile_rgb: np.ndarray) -> bool:
        """True iff the tile clears both thresholds."""
        m = self.evaluate(tile_rgb)
        return (
            m.tissue_ratio >= self.tissue_threshold
            and m.laplacian_var >= self.blur_threshold
        )
