"""Macenko stain normalization wrapper.

Wraps :class:`torchstain.normalizers.MacenkoNormalizer` (numpy backend)
behind a project-local API so the rest of the pipeline never has to know
about torchstain's signature quirks (3-tuple return, ``Io`` / ``alpha`` /
``beta`` defaults, fit-before-normalize ordering).

Macenko et al. (2009) factorise an H&E image into hematoxylin and eosin
stain vectors via SVD on the optical-density transform, then re-stain it
to a reference image's stain palette. This compensates for between-scanner
color drift, which the design (`02-data-spec.md`) treats as the dominant
nuisance variable on the CATCH dataset.

Two normalizer modes are supported:

* ``"macenko"`` — full Macenko fit-and-transform.
* ``"none"`` — pass-through identity (used by ablation Experiment 5).

The mode is selected via ``preprocessing.stain_normalization`` in the YAML
config.

References:
    Design: docs/02-design/03-architecture.md §2 (Module A)
    Spec:   docs/02-design/02-data-spec.md §6.3 (stain norm verification)
    Paper:  Macenko et al., ISBI 2009. "A method for normalizing
            histology slides for quantitative analysis."
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Iterator, Literal

import numpy as np

from src.utils.errors import ConfigError

logger = logging.getLogger(__name__)

NormalizationMode = Literal["macenko", "none"]


@contextmanager
def _silence_macenko_numpy_noise() -> Iterator[None]:
    """Suppress numpy RuntimeWarnings emitted *inside* torchstain's Macenko.

    On tiles that pass the project tissue/blur filters but are still dominated
    by near-white pixels, torchstain's beta=0.15 OD cutoff leaves ``ODhat``
    with too few rows. The downstream ``np.cov`` / ``np.linalg.eigh`` call
    chain then emits "Mean of empty slice", "Degrees of freedom <= 0",
    "divide by zero", and "invalid value" RuntimeWarnings before raising
    LinAlgError. The LinAlgError is caught and handled below; this context
    manager only mutes the cosmetic warnings so logs aren't drowned out.
    """
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


class StainNormalizer:
    """Project-facing stain normalizer.

    Use :meth:`fit` once with a reference tile (typically the
    Macenko-recommended target image, or a representative tile from a
    high-quality slide), then call :meth:`transform` per tile during
    tiling.

    The ``"none"`` mode is included so the same orchestrator code path
    can run the no-normalization ablation without conditional logic at
    every call site.
    """

    def __init__(self, mode: NormalizationMode = "macenko") -> None:
        if mode not in ("macenko", "none"):
            raise ConfigError(
                f"stain_normalization must be 'macenko' or 'none'; got {mode!r}"
            )
        self.mode = mode
        self._fitted = False
        self._impl = None
        # Per-slide counter; the tiler resets between slides and reads after.
        self.transform_failures = 0
        if mode == "macenko":
            # Lazy import keeps torchstain optional for none-mode users.
            import torchstain  # noqa: WPS433 (intentional lazy import)

            self._impl = torchstain.normalizers.MacenkoNormalizer(backend="numpy")

    def reset_counters(self) -> None:
        """Reset per-slide diagnostic counters. Called by the tiler before each slide."""
        self.transform_failures = 0

    def fit(self, reference_rgb: np.ndarray) -> None:
        """Fit the normalizer to a reference tile.

        For ``mode='none'`` this is a no-op so the call site can be uniform.
        """
        _validate_rgb_uint8(reference_rgb, "reference_rgb")
        if self.mode == "macenko":
            with _silence_macenko_numpy_noise():
                self._impl.fit(reference_rgb)
        self._fitted = True

    def transform(self, tile_rgb: np.ndarray) -> np.ndarray:
        """Return a stain-normalized copy of ``tile_rgb`` (uint8 RGB).

        Raises:
            ConfigError: if called before :meth:`fit` in macenko mode.
        """
        _validate_rgb_uint8(tile_rgb, "tile_rgb")
        if self.mode == "none":
            return tile_rgb.copy()
        if not self._fitted:
            raise ConfigError(
                "StainNormalizer must be fit() with a reference tile before transform()."
            )
        try:
            with _silence_macenko_numpy_noise():
                normalized, _h, _e = self._impl.normalize(I=tile_rgb)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            # torchstain's __compute_matrices is fragile on tiles whose OD
            # passes the project tissue filter but is dominated by
            # near-white pixels at Macenko's beta=0.15 cutoff: ODhat ends
            # up with too few rows, so np.cov returns NaN/inf and
            # np.linalg.eigh fails to converge. Mirror the fit-side guard
            # (wsi_tiler._maybe_fit_normalizer) by falling back to the
            # original tile and counting the event for metadata.
            self.transform_failures += 1
            logger.warning(
                "Macenko transform failed (%s); using unnormalized tile.", exc
            )
            return tile_rgb.copy()
        # torchstain returns uint8 already; copy to be safe against in-place writes.
        return np.asarray(normalized, dtype=np.uint8)


def _validate_rgb_uint8(arr: np.ndarray, name: str) -> None:
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must be (H, W, 3); got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8; got {arr.dtype}")
