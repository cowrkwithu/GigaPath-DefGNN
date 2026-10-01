"""Custom exception hierarchy for the VetGigaGraph pipeline.

All pipeline-raised exceptions inherit from :class:`VetGigaGraphError` so
top-level scripts can catch a single base type without swallowing unrelated
errors (KeyboardInterrupt, IO, etc).

References:
    Design: docs/02-design/features/vetgigagraph.design.md §6 (Error Handling)
    Design: docs/02-design/03-architecture.md §2.A, §3.B, §4.C, §6.E
"""

from __future__ import annotations


class VetGigaGraphError(Exception):
    """Base class for all pipeline-raised exceptions.

    Top-level scripts (``scripts/01_preprocess.py`` etc.) catch only this base.
    Anything else propagates uncaught.
    """


class InsufficientTilesError(VetGigaGraphError):
    """A slide produced fewer than ``min_tiles_per_slide`` valid tiles.

    Raised by :mod:`src.preprocessing.quality_filter` after applying the
    tissue-ratio and Laplacian-variance filters. The slide is recorded in
    ``logs/skipped_slides.csv`` and excluded from downstream steps.
    """


class GraphDisconnectedError(VetGigaGraphError):
    """A graph constructor produced an empty ``edge_index``.

    Raised by :mod:`src.graph_construction.*` when extreme thresholds (e.g.
    ``feature_sim`` τ too high) leave the graph with zero edges. Hint: lower
    threshold or increase ``k``.
    """


class InsufficientFoldsError(VetGigaGraphError):
    """A statistical test received fewer fold scores than its minimum sample size.

    Raised by :mod:`src.evaluation.statistical_test` (e.g. paired t-test needs
    ≥ 5 paired observations).
    """


class ConfigError(VetGigaGraphError):
    """A configuration value is missing, malformed, or contradicts a locked default.

    Raised by :mod:`src.utils.config` during validation.
    """


class DataIntegrityError(VetGigaGraphError):
    """A data file (HDF5, PyG, CSV) violates its locked schema.

    Raised by :mod:`src.utils.io_utils` and verifier scripts when the on-disk
    artifact deviates from the schema declared in design docs.
    """


class FrozenEncoderError(VetGigaGraphError):
    """The GigaPath tile encoder unexpectedly received gradients.

    Raised by sanity-guard hooks in :mod:`src.feature_extraction` and
    :mod:`src.models` when the frozen-by-default encoder shows
    ``requires_grad=True`` or non-None ``grad`` after backward.
    """
