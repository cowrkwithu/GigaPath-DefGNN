"""Global seed setting for reproducibility.

Every script's main() should call :func:`set_global_seed` first, before any
RNG access. The seed list locked in design (one per fold) is published here.

References:
    Design: docs/02-design/08-statistics-reproducibility.md §2, §5
    Design: docs/02-design/04-experiment-design.md (Locked Default Hyperparameters → Evaluation)
"""

from __future__ import annotations

import logging
import os
import random
from typing import Final

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Per-fold seeds locked in design (one per 5-fold CV iteration).
LOCKED_FOLD_SEEDS: Final[tuple[int, ...]] = (42, 123, 456, 789, 1024)

#: Tracks whether :func:`set_global_seed` has been called this process.
#: Verifier scripts read this flag to satisfy run-validity check U2.
_seed_call_recorded: bool = False
_last_seed: int | None = None


def set_global_seed(seed: int, *, deterministic: bool = True, warn_only: bool = False) -> None:
    """Seed all RNGs (Python, NumPy, PyTorch CPU + CUDA) for reproducibility.

    Args:
        seed: Integer seed. Must be one of :data:`LOCKED_FOLD_SEEDS` for runs
            that will be aggregated into the 5-fold CV results, but any int is
            accepted (e.g. for ad-hoc smoke tests).
        deterministic: If True (default), enable PyTorch deterministic mode.
            This trades ~10–30% throughput for bit-identical fp32 results
            across runs. Set False when running in fp16 mixed-precision and
            you accept ≤ 1e-3 BACC drift across machines (per design L4 tolerance).
        warn_only: If True, ``torch.use_deterministic_algorithms`` is called
            with ``warn_only=True`` so non-deterministic ops issue a warning
            instead of an exception. Use during development only.

    Notes:
        Sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` per PyTorch determinism docs.
    """
    global _seed_call_recorded, _last_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # https://docs.pytorch.org/docs/stable/notes/randomness.html
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except (RuntimeError, AttributeError) as e:
            logger.warning("Could not enable full deterministic algorithms: %s", e)
    else:
        # Allow cuDNN heuristics — faster but non-bit-identical
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    _seed_call_recorded = True
    _last_seed = seed
    logger.info("Global seed set: %d (deterministic=%s)", seed, deterministic)


def seed_was_set() -> bool:
    """Return True if :func:`set_global_seed` has been called this process.

    Used by run validators (design `04-experiment-design.md` §6.1 U2) to
    confirm the entry point performed the locked seeding ritual.
    """
    return _seed_call_recorded


def last_seed() -> int | None:
    """Return the most recently set seed, or None if never set."""
    return _last_seed
