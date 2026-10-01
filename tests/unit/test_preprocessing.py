"""Unit tests for ``src/preprocessing/`` (Phase 2, Module A).

Coverage map (design ``docs/02-design/03-architecture.md`` §2.A):

| Test                                  | Phase 2 task |
|---------------------------------------|--------------|
| test_tissue_detector_white_image      | 2.2          |
| test_tissue_detector_full_tissue      | 2.2          |
| test_tile_grid_dimensions             | 2.5          |
| test_tile_size_invariant              | 2.5          |
| test_blur_filter_drops_blurry         | 2.4          |
| test_blur_filter_keeps_sharp          | 2.4          |
| test_tissue_threshold_drops_low       | 2.4          |
| test_macenko_idempotent               | 2.3          |
| test_insufficient_tiles_raises        | 2.5          |

Synthetic image generators are deterministic (seeded numpy RNGs) so any
flake here is a real regression, not an RNG artefact.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import (
    StainNormalizer,
    TileQualityFilter,
    WSITiler,
    detect_tissue,
)
from src.utils.errors import InsufficientTilesError


# --------------------------------------------------------------------------- #
# Synthetic image fixtures
# --------------------------------------------------------------------------- #


def _white_image(shape: tuple[int, int, int] = (256, 256, 3)) -> np.ndarray:
    """Pure-white slide background — no tissue at all."""
    return np.full(shape, 255, dtype=np.uint8)


def _tissue_image(
    shape: tuple[int, int, int] = (256, 256, 3),
    seed: int = 0,
) -> np.ndarray:
    """Synthetic H&E-like tile: pink/purple textured patch with high saturation.

    We sample around (170, 90, 170) — a representative H&E hue — with
    independent noise per channel. Noise is bounded so saturation stays
    well above the unimodal-high cutoff in :func:`detect_tissue`, and
    high enough Laplacian variance to satisfy the blur filter.
    """
    rng = np.random.default_rng(seed)
    base = np.array([170, 90, 170], dtype=np.int16)
    noise = rng.integers(-25, 26, size=shape, dtype=np.int16)
    img = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
    return img


def _tissue_canvas_with_padding(
    canvas_shape: tuple[int, int, int],
    tissue_origin: tuple[int, int],
    tissue_size: tuple[int, int],
    seed: int = 0,
) -> np.ndarray:
    """White canvas with a tissue patch placed at ``tissue_origin``."""
    canvas = _white_image(canvas_shape)
    y, x = tissue_origin
    h, w = tissue_size
    canvas[y : y + h, x : x + w] = _tissue_image((h, w, 3), seed=seed)
    return canvas


# --------------------------------------------------------------------------- #
# Tissue detector
# --------------------------------------------------------------------------- #


def test_tissue_detector_white_image() -> None:
    """A pure-white slide must produce an all-False tissue mask."""
    mask = detect_tissue(_white_image())
    assert mask.dtype == bool
    assert mask.sum() == 0


def test_tissue_detector_full_tissue() -> None:
    """A fully tissue-colored patch must light up ≥ 95% of pixels."""
    mask = detect_tissue(_tissue_image(seed=1))
    assert mask.mean() >= 0.95


# --------------------------------------------------------------------------- #
# Tiler grid + tile-size invariants
# --------------------------------------------------------------------------- #


def test_tile_grid_dimensions() -> None:
    """A 1024×1024 fully-tissue canvas at stride 256 must yield 16 tiles."""
    canvas = _tissue_image((1024, 1024, 3), seed=2)
    tiler = WSITiler(
        tile_size=256,
        stride=256,
        tissue_threshold=0.5,
        blur_threshold=100.0,
        min_tiles_per_slide=1,
        stain_normalization="none",
    )
    kept = tiler.tile_array(canvas, slide_id="grid_dim")
    assert len(kept) == 16


def test_tile_size_invariant() -> None:
    """Every emitted tile must be exactly 256x256x3 uint8."""
    canvas = _tissue_image((1024, 1024, 3), seed=3)
    tiler = WSITiler(
        tile_size=256,
        stride=256,
        tissue_threshold=0.5,
        blur_threshold=100.0,
        min_tiles_per_slide=1,
        stain_normalization="none",
    )
    kept = tiler.tile_array(canvas, slide_id="size_inv")
    assert all(rec.rgb.shape == (256, 256, 3) for rec in kept)
    assert all(rec.rgb.dtype == np.uint8 for rec in kept)


# --------------------------------------------------------------------------- #
# Quality filter — sharpness + tissue ratio
# --------------------------------------------------------------------------- #


def test_blur_filter_drops_blurry() -> None:
    """A heavily blurred tile must fail the Laplacian-variance check."""
    pytest.importorskip("cv2")
    import cv2

    sharp = _tissue_image(seed=4)
    blurred = cv2.GaussianBlur(sharp, ksize=(0, 0), sigmaX=10)
    qf = TileQualityFilter(tissue_threshold=0.0, blur_threshold=100.0)
    metrics = qf.evaluate(blurred)
    assert metrics.laplacian_var < 100.0
    assert qf.passes(blurred) is False


def test_blur_filter_keeps_sharp() -> None:
    """A textured H&E-like tile must pass both checks."""
    sharp = _tissue_image(seed=5)
    qf = TileQualityFilter(tissue_threshold=0.5, blur_threshold=100.0)
    assert qf.passes(sharp) is True


def test_tissue_threshold_drops_low() -> None:
    """A tile with ~30% tissue must fail the tissue-ratio check."""
    # Place a 140×256 tissue strip on a 256×256 white tile → ~55% of pixels.
    # Then shrink to 80×256 → ~31% — under the 0.5 default threshold.
    canvas = _tissue_canvas_with_padding(
        canvas_shape=(256, 256, 3),
        tissue_origin=(0, 0),
        tissue_size=(80, 256),
        seed=6,
    )
    qf = TileQualityFilter(tissue_threshold=0.5, blur_threshold=100.0)
    metrics = qf.evaluate(canvas)
    assert metrics.tissue_ratio < 0.5
    assert qf.passes(canvas) is False


# --------------------------------------------------------------------------- #
# Stain normalizer — Macenko idempotency
# --------------------------------------------------------------------------- #


def test_macenko_idempotent() -> None:
    """Applying Macenko twice with the same fitted normalizer ≈ applying it once.

    The locked semantics from the design (``03-architecture.md`` §2):
    Macenko learns a target H&E palette from a *reference* image, then
    re-stains any input toward that palette. With the palette fixed,
    ``f(f(x))`` should be a no-op modulo numerical noise.

    A different seed for ``test_tile`` than for ``reference`` keeps the
    test from collapsing into the trivial case where the palette is the
    input itself.
    """
    pytest.importorskip("torchstain")
    reference = _tissue_image(seed=7)
    test_tile = _tissue_image(seed=42)

    normalizer = StainNormalizer(mode="macenko")
    normalizer.fit(reference)
    once = normalizer.transform(test_tile)
    twice = normalizer.transform(once)

    diff = np.abs(once.astype(np.int16) - twice.astype(np.int16)).mean()
    assert diff < 12.0, (
        f"Macenko not idempotent: mean abs diff {diff:.3f} > 12"
    )


def test_macenko_transform_falls_back_on_lin_alg_error() -> None:
    """A degenerate tile (eigh non-convergence inside torchstain) must not abort tiling.

    Real-world cause: a tile passes the project tissue/blur filters but,
    after torchstain's stricter beta=0.15 OD cutoff, leaves ODhat with
    too few rows — covariance becomes NaN/inf and ``np.linalg.eigh``
    raises ``LinAlgError``. The fix returns the original tile unchanged
    and increments ``transform_failures`` so the tiler can record it in
    metadata.json (mirrors the existing fit-side guard).

    We inject the failure via the underlying torchstain instance so the
    test is deterministic across torchstain versions.
    """
    pytest.importorskip("torchstain")
    reference = _tissue_image(seed=7)
    tile = _tissue_image(seed=42)

    normalizer = StainNormalizer(mode="macenko")
    normalizer.fit(reference)

    class _Boom:
        def normalize(self, **_kwargs: object) -> None:
            raise np.linalg.LinAlgError("Eigenvalues did not converge")

    normalizer._impl = _Boom()
    normalizer.reset_counters()

    result = normalizer.transform(tile)

    assert normalizer.transform_failures == 1
    assert np.array_equal(result, tile), "fallback must return the original tile bytes"
    assert result is not tile, "fallback must return a copy, not the same object"


# --------------------------------------------------------------------------- #
# Insufficient-tiles guardrail
# --------------------------------------------------------------------------- #


def test_insufficient_tiles_raises() -> None:
    """A near-empty WSI must raise ``InsufficientTilesError``, not silently emit zero tiles."""
    blank = _white_image((1024, 1024, 3))
    tiler = WSITiler(
        tile_size=256,
        stride=256,
        tissue_threshold=0.5,
        blur_threshold=100.0,
        min_tiles_per_slide=16,
        stain_normalization="none",
    )
    with pytest.raises(InsufficientTilesError, match="min_tiles_per_slide"):
        tiler.tile_array(blank, slide_id="empty_slide")
