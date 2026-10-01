"""Whole-slide image tiler — Stage 1 of the VetGigaGraph pipeline.

Reads a WSI through OpenSlide, walks a non-overlapping grid at the target
magnification, applies tissue + sharpness filters, optionally Macenko-
normalizes each surviving tile, and writes:

* ``<out_dir>/<slide_id>/tile_<idx>_<x>_<y>.png`` — one PNG per kept tile.
* ``<out_dir>/<slide_id>/coords.csv`` — locked schema
  ``{tile_idx, x, y, tissue_ratio, laplacian_var}`` (matches
  `02-data-spec.md` §6.3).
* ``<out_dir>/<slide_id>/metadata.json`` — slide-level provenance
  (slide id, magnification used, tile size, threshold values, kept count,
  encoder version stub).

The orchestrator raises :class:`InsufficientTilesError` when the kept tile
count falls below ``min_tiles_per_slide`` so a downstream batch script
can record the slide in ``logs/skipped_slides.csv`` instead of silently
producing an empty directory (the failure mode the design explicitly
forbids).

References:
    Design: docs/02-design/03-architecture.md §2, §2.A
    Spec:   docs/02-design/02-data-spec.md §6.3
    Config: configs/default.yaml `preprocessing:` block
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from src.preprocessing.quality_filter import TileQualityFilter, TileQualityMetrics
from src.preprocessing.stain_normalizer import NormalizationMode, StainNormalizer
from src.preprocessing.tissue_detector import detect_tissue
from src.utils.errors import InsufficientTilesError

logger = logging.getLogger(__name__)

#: Schema for ``coords.csv`` (locked — used by data verifier in §6.3).
COORDS_CSV_COLUMNS = ("tile_idx", "x", "y", "tissue_ratio", "laplacian_var")


@dataclass
class TileRecord:
    """One kept tile's row in ``coords.csv`` plus its pixel data."""

    tile_idx: int
    x: int  # level-0 pixel coordinate of top-left corner
    y: int
    tissue_ratio: float
    laplacian_var: float
    rgb: np.ndarray = field(repr=False)


# --------------------------------------------------------------------------- #
# WSI reader (thin OpenSlide wrapper)
# --------------------------------------------------------------------------- #


class OpenSlideWSI:
    """Lightweight wrapper around :class:`openslide.OpenSlide`.

    Centralises the level-selection logic so the tiler doesn't have to
    open/close slides directly. Supports ``.svs``, ``.ndpi``, ``.tif``
    (anything OpenSlide reads).
    """

    def __init__(self, path: str | Path) -> None:
        import openslide  # Lazy import: keeps the test suite importable
        # on machines without libopenslide installed.

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"WSI not found: {self.path}")
        self._slide = openslide.OpenSlide(str(self.path))

    # ----- context manager so callers can release file handles deterministically.
    def __enter__(self) -> "OpenSlideWSI":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self._slide.close()

    @property
    def dimensions(self) -> tuple[int, int]:
        """Level-0 ``(width, height)`` in pixels."""
        return self._slide.dimensions  # type: ignore[no-any-return]

    @property
    def level_count(self) -> int:
        return int(self._slide.level_count)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._slide.level_dimensions)

    def read_region_rgb(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Read an RGBA region and return it as an RGB uint8 array."""
        rgba = self._slide.read_region(location, level, size).convert("RGB")
        return np.asarray(rgba, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Tiler orchestrator
# --------------------------------------------------------------------------- #


class WSITiler:
    """Walk a WSI on a non-overlapping grid and emit kept tiles + metadata.

    Hyperparameters follow ``configs/default.yaml`` ``preprocessing:`` and
    the locked design defaults in `04-experiment-design.md`.
    """

    def __init__(
        self,
        *,
        tile_size: int = 256,
        stride: int = 256,
        tissue_threshold: float = 0.5,
        blur_threshold: float = 100.0,
        min_tiles_per_slide: int = 16,
        max_tiles_per_slide: int | None = None,
        stain_normalization: NormalizationMode = "macenko",
        level: int = 0,
        downsample: int = 1,
        prefilter_tissue_fraction: float = 0.0,
    ) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive; got {tile_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive; got {stride}")
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.min_tiles_per_slide = int(min_tiles_per_slide)
        self.max_tiles_per_slide = max_tiles_per_slide
        self.level = int(level)
        # downsample > 1 emulates a lower magnification when the pyramid has no
        # matching level (CATCH .svs: 40x, 10x, 2.5x only): read a
        # (tile_size * downsample)^2 region at level 0 and box-filter it down
        # to tile_size^2. Grid and stored coordinates stay in level-0 pixels.
        if downsample < 1 or (downsample > 1 and self.level != 0):
            raise ValueError("downsample must be >= 1 and requires level=0")
        self.downsample = int(downsample)
        # prefilter_tissue_fraction > 0: before reading a grid cell at level 0,
        # look up the fraction of tissue pixels (HSV saturation >= 10) in the
        # cell's footprint on a ~16x-downsampled thumbnail and skip the read
        # when it is below this value. Pure speed-up: the normalizer is only
        # fitted on kept tiles, so the output is unchanged as long as no kept
        # tile is skipped (validated at 0.01 on 57 slides / 261k kept tiles,
        # lowest thumbnail fraction of any kept tile 0.513).
        self.prefilter_tissue_fraction = float(prefilter_tissue_fraction)
        self.prefilter_skipped = 0
        self.quality_filter = TileQualityFilter(
            tissue_threshold=tissue_threshold,
            blur_threshold=blur_threshold,
        )
        self.normalizer = StainNormalizer(mode=stain_normalization)

    # --- public API ------------------------------------------------------- #

    def tile_array(
        self,
        rgb: np.ndarray,
        *,
        slide_id: str = "memory",
    ) -> list[TileRecord]:
        """Tile an in-memory RGB image. Used by tests and notebook smoke runs.

        The Macenko normalizer is fitted on the FIRST tile that passes the
        tissue test (``tissue_ratio > 0.1``). Slides that have no tissue at
        all therefore raise :class:`InsufficientTilesError` before any
        normalization is attempted.
        """
        self.normalizer.reset_counters()
        kept = list(self._iter_records(rgb))
        self._enforce_min(kept, slide_id)
        if self.max_tiles_per_slide is not None:
            kept = kept[: self.max_tiles_per_slide]
        return kept

    def tile(
        self,
        slide_path: str | Path,
        out_dir: str | Path,
        *,
        slide_id: str | None = None,
    ) -> Path:
        """Tile a WSI on disk and persist tiles + coords.csv + metadata.json.

        Args:
            slide_path: Path to the WSI (``.svs`` / ``.ndpi`` / ``.tif``).
            out_dir: Parent directory; per-slide outputs land in
                ``<out_dir>/<slide_id>/``.
            slide_id: Override the directory name. Defaults to the WSI's
                file stem.

        Returns:
            The per-slide output directory path.
        """
        slide_path = Path(slide_path)
        out_dir = Path(out_dir)
        sid = slide_id or slide_path.stem

        self.normalizer.reset_counters()
        with OpenSlideWSI(slide_path) as slide:
            width, height = slide.level_dimensions[self.level]
            # Slabs that span the whole slide can be GBs; the per-tile read
            # below stays within OpenSlide's memory budget.
            kept = list(self._iter_records_from_slide(slide, width, height))
        stain_norm_failures = self.normalizer.transform_failures

        self._enforce_min(kept, sid)
        if self.max_tiles_per_slide is not None:
            kept = kept[: self.max_tiles_per_slide]

        slide_out = out_dir / sid
        slide_out.mkdir(parents=True, exist_ok=True)
        self._write_outputs(kept, slide_out, sid, stain_norm_failures)
        if stain_norm_failures:
            logger.warning(
                "Tiled %s: %d tile(s) fell back to unnormalized RGB after Macenko failure.",
                sid,
                stain_norm_failures,
            )
        logger.info(
            "Tiled %s: kept %d tiles → %s", sid, len(kept), slide_out
        )
        return slide_out

    # --- internal --------------------------------------------------------- #

    def _grid_positions(
        self, width: int, height: int, scale: int = 1
    ) -> Iterator[tuple[int, int]]:
        """Yield ``(x, y)`` top-left coordinates of each tile in the grid.

        ``scale`` multiplies tile size and stride (level-0 footprint of a
        downsampled tile).
        """
        size, stride = self.tile_size * scale, self.stride * scale
        for y in range(0, height - size + 1, stride):
            for x in range(0, width - size + 1, stride):
                yield x, y

    def _maybe_fit_normalizer(self, rgb: np.ndarray) -> bool:
        """Fit Macenko on the first viable tile; return True once fitted."""
        # Cheap pre-check: fully blank tiles produce a near-singular OD matrix
        # and crash torchstain. Skip them.
        if float(detect_tissue(rgb).mean()) < 0.1:
            return False
        try:
            self.normalizer.fit(rgb)
        except Exception as exc:  # torchstain raises generic Exceptions
            logger.warning("Macenko fit failed on candidate tile (%s); will retry.", exc)
            return False
        return True

    def _iter_records(self, rgb: np.ndarray) -> Iterator[TileRecord]:
        """Yield kept tiles for an in-memory image."""
        height, width = rgb.shape[:2]
        idx = 0
        normalizer_ready = self.normalizer.mode == "none"
        for x, y in self._grid_positions(width, height):
            tile = rgb[y : y + self.tile_size, x : x + self.tile_size]
            if tile.shape[:2] != (self.tile_size, self.tile_size):
                continue
            metrics = self.quality_filter.evaluate(tile)
            if not _passes(metrics, self.quality_filter):
                continue
            if not normalizer_ready:
                normalizer_ready = self._maybe_fit_normalizer(tile)
            normalized = (
                self.normalizer.transform(tile) if normalizer_ready else tile.copy()
            )
            yield TileRecord(
                tile_idx=idx,
                x=int(x),
                y=int(y),
                tissue_ratio=metrics.tissue_ratio,
                laplacian_var=metrics.laplacian_var,
                rgb=normalized,
            )
            idx += 1

    def _iter_records_from_slide(
        self,
        slide: OpenSlideWSI,
        width: int,
        height: int,
    ) -> Iterator[TileRecord]:
        """Yield kept tiles for a WSI on disk."""
        idx = 0
        normalizer_ready = self.normalizer.mode == "none"
        d = self.downsample
        read = self.tile_size * d
        frac = self._thumbnail_tissue_fraction(slide, read) if self.prefilter_tissue_fraction > 0 else None
        self.prefilter_skipped = 0
        for x, y in self._grid_positions(width, height, scale=d):
            if frac is not None and frac[y // read, x // read] < self.prefilter_tissue_fraction:
                self.prefilter_skipped += 1
                continue
            tile = slide.read_region_rgb(
                location=(x, y),
                level=self.level,
                size=(read, read),
            )
            if d > 1:
                from PIL import Image
                tile = np.asarray(
                    Image.fromarray(tile).resize(
                        (self.tile_size, self.tile_size), Image.Resampling.BOX
                    ),
                    dtype=np.uint8,
                )
            metrics = self.quality_filter.evaluate(tile)
            if not _passes(metrics, self.quality_filter):
                continue
            if not normalizer_ready:
                normalizer_ready = self._maybe_fit_normalizer(tile)
            normalized = (
                self.normalizer.transform(tile) if normalizer_ready else tile.copy()
            )
            yield TileRecord(
                tile_idx=idx,
                x=int(x),
                y=int(y),
                tissue_ratio=metrics.tissue_ratio,
                laplacian_var=metrics.laplacian_var,
                rgb=normalized,
            )
            idx += 1

    @staticmethod
    def _thumbnail_tissue_fraction(slide: OpenSlideWSI, cell: int) -> np.ndarray:
        """Tissue-pixel fraction per ``cell``-px level-0 grid cell, from a
        ~16x-downsampled pyramid level (integral image over the thumbnail)."""
        import cv2
        raw = slide._slide
        lvl = raw.get_best_level_for_downsample(16)
        ds = raw.level_downsamples[lvl]
        thumb = np.asarray(raw.read_region((0, 0), lvl, raw.level_dimensions[lvl]).convert("RGB"))
        tissue = (cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)[..., 1] >= 10).astype(np.float64)
        ii = np.pad(tissue.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        width, height = raw.dimensions
        nx, ny = (width - cell) // cell + 1, (height - cell) // cell + 1
        xs = np.clip(np.round(np.arange(nx + 1) * cell / ds).astype(int), 0, tissue.shape[1])
        ys = np.clip(np.round(np.arange(ny + 1) * cell / ds).astype(int), 0, tissue.shape[0])
        x0, x1, y0, y1 = xs[:-1], xs[1:], ys[:-1], ys[1:]
        area = np.maximum((y1 - y0)[:, None] * (x1 - x0)[None, :], 1)
        return (ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]) / area

    def _enforce_min(self, kept: list[TileRecord], slide_id: str) -> None:
        if len(kept) < self.min_tiles_per_slide:
            raise InsufficientTilesError(
                f"Slide {slide_id!r} produced {len(kept)} tiles "
                f"(< min_tiles_per_slide={self.min_tiles_per_slide}). "
                "Record in logs/skipped_slides.csv and exclude from downstream steps."
            )

    def _write_outputs(
        self,
        kept: list[TileRecord],
        slide_out: Path,
        slide_id: str,
        stain_norm_failures: int = 0,
    ) -> None:
        """Persist PNGs, coords.csv, and metadata.json."""
        # Lazy imports keep the import-time graph small for unit tests
        # that only exercise the in-memory tile_array path.
        import pandas as pd
        from PIL import Image

        for rec in kept:
            fname = f"tile_{rec.tile_idx:06d}_{rec.x}_{rec.y}.png"
            Image.fromarray(rec.rgb).save(slide_out / fname, format="PNG")

        df = pd.DataFrame(
            [
                {
                    "tile_idx": rec.tile_idx,
                    "x": rec.x,
                    "y": rec.y,
                    "tissue_ratio": rec.tissue_ratio,
                    "laplacian_var": rec.laplacian_var,
                }
                for rec in kept
            ],
            columns=list(COORDS_CSV_COLUMNS),
        )
        df.to_csv(slide_out / "coords.csv", index=False)

        metadata = {
            "slide_id": slide_id,
            "num_tiles": len(kept),
            "tile_size": self.tile_size,
            "stride": self.stride,
            "level": self.level,
            "downsample": self.downsample,
            "prefilter_tissue_fraction": self.prefilter_tissue_fraction,
            "prefilter_skipped_cells": int(self.prefilter_skipped),
            "tissue_threshold": self.quality_filter.tissue_threshold,
            "blur_threshold": self.quality_filter.blur_threshold,
            "stain_normalization": self.normalizer.mode,
            "stain_norm_failures": int(stain_norm_failures),
        }
        (slide_out / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )


def _passes(metrics: TileQualityMetrics, qf: TileQualityFilter) -> bool:
    """Inline equivalent of ``qf.passes()`` that reuses already-computed metrics."""
    return (
        metrics.tissue_ratio >= qf.tissue_threshold
        and metrics.laplacian_var >= qf.blur_threshold
    )
