"""Statistical tests for cross-fold model comparison (Phase 8.3).

The proposal mandates rigorous significance testing before any "model A
beats model B" claim:

* :func:`paired_ttest` — paired Student's t-test over per-fold scores
  (assumes approximate normality; pairs match by fold index).
* :func:`wilcoxon` — non-parametric paired sign-rank test, used when
  fold scores are skewed or ordinal.
* :func:`mcnemar` — McNemar's test on per-slide correct/incorrect
  predictions for two classifiers.
* :func:`friedman_nemenyi` — Friedman omnibus + Nemenyi post-hoc for
  comparing >= 3 models across the same folds.
* :func:`bonferroni_correction` — multiple-comparison adjustment
  applied before ``p`` values are reported in the final paper.

All paired tests raise :class:`InsufficientFoldsError` when given
fewer than 5 paired observations, matching the design's
"5-fold + 5 seeds = 5 paired scores" minimum.

References:
    Design: docs/02-design/08-statistics-reproducibility.md §6 (sanity gates)
    Tests:  docs/02-design/03-architecture.md §6.E (paired/wilcoxon/mcnemar
            identical-input sanity, friedman power, bonferroni correction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

from src.utils.errors import InsufficientFoldsError

logger = logging.getLogger(__name__)

#: Minimum paired sample size before statistical tests are meaningful.
MIN_PAIRED_N = 5


@dataclass(frozen=True)
class TestResult:
    """Compact result type used by all paired tests.

    ``statistic`` is the named test statistic (t, W, χ², Q depending on
    the test); ``p`` is the two-sided p-value (already corrected if
    ``alpha`` was supplied at the call site).
    """

    statistic: float
    p: float
    n: int
    test_name: str


# --------------------------------------------------------------------------- #
# Paired t-test
# --------------------------------------------------------------------------- #


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Paired Student's t-test on two equal-length fold-score vectors."""
    a_arr, b_arr = _coerce_paired(a, b, "paired_ttest")
    if np.array_equal(a_arr, b_arr):
        # Exact-equal vectors give p = NaN from scipy; surface as p = 1.0
        # so the design's "identical inputs → p > 0.99" assertion holds.
        return TestResult(statistic=0.0, p=1.0, n=len(a_arr), test_name="paired_ttest")
    res = stats.ttest_rel(a_arr, b_arr)
    return TestResult(
        statistic=float(res.statistic),
        p=float(res.pvalue),
        n=len(a_arr),
        test_name="paired_ttest",
    )


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank
# --------------------------------------------------------------------------- #


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Wilcoxon signed-rank paired test."""
    a_arr, b_arr = _coerce_paired(a, b, "wilcoxon")
    if np.array_equal(a_arr, b_arr):
        return TestResult(statistic=0.0, p=1.0, n=len(a_arr), test_name="wilcoxon")
    # zero_method="zsplit" handles ties without zeroing the test out.
    res = stats.wilcoxon(a_arr, b_arr, zero_method="zsplit")
    return TestResult(
        statistic=float(res.statistic),
        p=float(res.pvalue),
        n=len(a_arr),
        test_name="wilcoxon",
    )


# --------------------------------------------------------------------------- #
# McNemar (per-slide classification agreement)
# --------------------------------------------------------------------------- #


def mcnemar(
    y_true: Sequence[int],
    y_a: Sequence[int],
    y_b: Sequence[int],
    *,
    exact: bool = True,
) -> TestResult:
    """McNemar's test on classifier-vs-classifier disagreement.

    ``y_true`` is the gold label; ``y_a`` and ``y_b`` are predictions.
    The 2×2 contingency table partitions slides by whether each
    classifier was correct, and the test asks whether the off-diagonal
    counts (one right, one wrong) are symmetric.
    """
    yt = _to_int(y_true)
    ya = _to_int(y_a)
    yb = _to_int(y_b)
    if yt.shape != ya.shape or yt.shape != yb.shape:
        raise ValueError(
            "mcnemar: shape mismatch — "
            f"y_true {yt.shape}, y_a {ya.shape}, y_b {yb.shape}"
        )
    if yt.size < 1:
        raise InsufficientFoldsError("mcnemar: empty inputs")

    correct_a = ya == yt
    correct_b = yb == yt
    # Identical predictions → identical correctness → off-diagonal = 0.
    # statsmodels' mcnemar would return p = NaN; surface as p = 1.0.
    if np.array_equal(ya, yb):
        return TestResult(
            statistic=0.0,
            p=1.0,
            n=int(yt.size),
            test_name="mcnemar",
        )

    table = np.array(
        [
            [int(np.sum(correct_a & correct_b)), int(np.sum(correct_a & ~correct_b))],
            [int(np.sum(~correct_a & correct_b)), int(np.sum(~correct_a & ~correct_b))],
        ],
        dtype=np.int64,
    )

    from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar

    res = _sm_mcnemar(table, exact=exact)
    return TestResult(
        statistic=float(res.statistic) if res.statistic is not None else 0.0,
        p=float(res.pvalue),
        n=int(yt.size),
        test_name="mcnemar",
    )


# --------------------------------------------------------------------------- #
# Friedman + Nemenyi post-hoc
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FriedmanResult:
    """Friedman omnibus + Nemenyi pairwise comparisons.

    ``post_hoc`` is a ``[K, K]`` matrix of pairwise p-values from the
    Nemenyi test (``post_hoc[i, j]`` = p-value testing whether models
    ``i`` and ``j`` differ). Diagonal is 1.0.
    """

    statistic: float
    p: float
    n_models: int
    n_folds: int
    test_name: str
    post_hoc: np.ndarray  # [K, K]


def friedman_nemenyi(scores: Sequence[Sequence[float]]) -> FriedmanResult:
    """Friedman omnibus then Nemenyi pairwise post-hoc.

    Args:
        scores: ``K`` rows, ``F`` columns — model × fold scores. Same
            fold ordering across all models is required.

    Returns:
        :class:`FriedmanResult`. ``p`` is the omnibus p-value;
        ``post_hoc[i, j]`` is the Nemenyi p-value for the pair.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"scores must be 2D [K, F]; got {arr.shape}")
    K, F = arr.shape
    if K < 3:
        raise ValueError(
            f"friedman_nemenyi requires K >= 3 models; got {K}"
        )
    if F < MIN_PAIRED_N:
        raise InsufficientFoldsError(
            f"friedman_nemenyi requires >= {MIN_PAIRED_N} folds; got {F}"
        )

    # Identical-rows shortcut: scipy returns p = NaN otherwise.
    if all(np.array_equal(arr[0], arr[i]) for i in range(1, K)):
        return FriedmanResult(
            statistic=0.0,
            p=1.0,
            n_models=K,
            n_folds=F,
            test_name="friedman_nemenyi",
            post_hoc=np.ones((K, K), dtype=np.float64),
        )

    # Friedman test expects samples-per-treatment as separate args.
    omnibus = stats.friedmanchisquare(*[arr[i] for i in range(K)])
    omnibus_p = float(omnibus.pvalue)
    omnibus_stat = float(omnibus.statistic)

    post_hoc = _nemenyi_post_hoc(arr)

    return FriedmanResult(
        statistic=omnibus_stat,
        p=omnibus_p,
        n_models=K,
        n_folds=F,
        test_name="friedman_nemenyi",
        post_hoc=post_hoc,
    )


def _nemenyi_post_hoc(arr: np.ndarray) -> np.ndarray:
    """Pairwise Nemenyi post-hoc p-values from a ``[K, F]`` score matrix.

    Implements the standard CD-based test (Demšar 2006). For each
    fold, models are ranked; mean ranks are compared against the
    studentised-range distribution.
    """
    K, F = arr.shape
    # Per-fold ranks: rank each column (fold), highest score → rank 1.
    # We use scipy's rankdata which assigns lowest rank 1 to smallest;
    # negate to get "highest score = rank 1".
    ranks = np.zeros_like(arr)
    for f in range(F):
        ranks[:, f] = stats.rankdata(-arr[:, f])
    mean_rank = ranks.mean(axis=1)  # [K]

    # Standard error for Nemenyi (Demšar 2006 eq. 4):
    #   SE = sqrt(K(K+1) / (6 * F))
    # q-value = (R_i - R_j) / SE; converted to two-sided p via
    # studentised-range distribution.
    se = np.sqrt(K * (K + 1) / (6.0 * F))
    p_mat = np.ones((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            q = abs(mean_rank[i] - mean_rank[j]) / se
            # studentised-range p-value: scipy provides studentized_range
            p_mat[i, j] = 1.0 - stats.studentized_range.cdf(q, K, np.inf)
    return p_mat


# --------------------------------------------------------------------------- #
# Bonferroni correction
# --------------------------------------------------------------------------- #


def bonferroni_correction(alpha: float, n_tests: int) -> float:
    """Return the corrected per-test α threshold (= α / n_tests).

    Raises:
        ValueError: if ``alpha`` is outside ``(0, 1)`` or ``n_tests <= 0``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n_tests <= 0:
        raise ValueError(f"n_tests must be positive; got {n_tests}")
    return alpha / n_tests


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _coerce_paired(
    a: Sequence[float],
    b: Sequence[float],
    test_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"{test_name}: shape mismatch — a {a_arr.shape}, b {b_arr.shape}"
        )
    if a_arr.size < MIN_PAIRED_N:
        raise InsufficientFoldsError(
            f"{test_name}: requires >= {MIN_PAIRED_N} paired observations; "
            f"got {a_arr.size}"
        )
    return a_arr, b_arr


def _to_int(x: Sequence) -> np.ndarray:
    return np.asarray(x, dtype=np.int64).ravel()
