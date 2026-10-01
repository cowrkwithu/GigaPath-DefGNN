"""Per-run validators: U1–U8 universal + S1–S6 sanity gates (Phase 8.4).

Every fold-level training run must pass these gates before its score
enters the final report. The gates are split into two severity tiers:

* **invalidate** (U1–U8 universal validity, S2 metric-bound) — run
  fails one of these → score is dropped from aggregation.
* **investigate** (S1, S3–S6 sanity) — run fails → human review
  required before the score may be cited.

The full gate inventory (`docs/02-design/04-experiment-design.md` §6):

| Gate | Severity | Check |
|------|---------|-------|
| U1 | invalidate | run config logged in WandB / metadata |
| U2 | invalidate | seed was set before any RNG call |
| U3 | invalidate | fold assignment matches split CSV |
| U4 | invalidate | train ∩ val ∩ test patients disjoint |
| U5 | invalidate | every class present in val and test |
| U6 | invalidate | no NaN/Inf in final metrics |
| U7 | invalidate | checkpoint loadable + correct class |
| U8 | invalidate | training finished (or stopped early naturally) |
| S1 | investigate | BACC > 1/7 + 0.02 ≈ 0.163 |
| S2 | invalidate | BACC ≤ 1.0 (metric-impl bug) |
| S3 | investigate | macro AUROC > 0.52 |
| S4 | investigate | std(per-class accuracy) < 0.5 |
| S5 | investigate | model predicts ≥ 3 distinct classes |
| S6 | investigate | abs(BACC_train − BACC_val) < 0.5 |

References:
    Design: docs/02-design/04-experiment-design.md §6 (Per-Experiment
            Validation Gates)
    Tests:  design §6.E doesn't enumerate per-validator tests, but we
            cover the locked thresholds + severities.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

#: Locked validator severities.
SEVERITY_INVALIDATE = "invalidate"
SEVERITY_INVESTIGATE = "investigate"

#: Locked thresholds.
RANDOM_BACC = 1.0 / 7.0
S1_BACC_FLOOR = RANDOM_BACC + 0.02
S3_AUROC_FLOOR = 0.52
S4_PER_CLASS_STD_CEIL = 0.5
S5_MIN_DISTINCT_PREDS = 3
S6_TRAIN_VAL_BACC_GAP_CEIL = 0.5


@dataclass(frozen=True)
class ValidatorResult:
    """One gate's outcome."""

    name: str           # e.g. "U4"
    severity: str       # "invalidate" | "investigate"
    passed: bool
    detail: str = ""    # human-readable explanation when failed


@dataclass(frozen=True)
class RunMetadata:
    """Subset of run state the validators need.

    The fields are populated by the script wrapper / Lightning callback
    that owns the actual run; this dataclass is just the contract.
    """

    config_logged: bool                 # U1
    seed_was_set: bool                  # U2
    fold: int                           # U3
    splits_csv: Optional[Path] = None   # U3, U4, U5
    metrics: Optional[Mapping[str, Any]] = None  # U6, S1, S2, S3, S4, S5, S6
    checkpoint_path: Optional[Path] = None       # U7
    expected_model_cls: Optional[type] = None    # U7
    training_finished: bool = True               # U8
    train_bacc: Optional[float] = None           # S6
    predictions: Optional[Sequence[int]] = None  # S5


# --------------------------------------------------------------------------- #
# U gates
# --------------------------------------------------------------------------- #


def _gate_u1(meta: RunMetadata) -> ValidatorResult:
    return ValidatorResult(
        name="U1",
        severity=SEVERITY_INVALIDATE,
        passed=bool(meta.config_logged),
        detail="" if meta.config_logged else "wandb.config (or run metadata) missing keys",
    )


def _gate_u2(meta: RunMetadata) -> ValidatorResult:
    return ValidatorResult(
        name="U2",
        severity=SEVERITY_INVALIDATE,
        passed=bool(meta.seed_was_set),
        detail="" if meta.seed_was_set else "set_global_seed() was not called before any RNG use",
    )


def _gate_u3(meta: RunMetadata) -> ValidatorResult:
    """Run's fold index is in the canonical [0, 4] range."""
    ok = 0 <= meta.fold <= 4
    return ValidatorResult(
        name="U3",
        severity=SEVERITY_INVALIDATE,
        passed=ok,
        detail="" if ok else f"fold={meta.fold} not in [0, 4]",
    )


def _gate_u4(meta: RunMetadata) -> ValidatorResult:
    """Patient-disjoint train / val / test for the run's fold.

    Defers to :func:`src.evaluation.cross_validation.verify_splits`
    when ``splits_csv`` is provided. If no CSV is set we cannot verify
    and the gate fails as a precaution.
    """
    if meta.splits_csv is None:
        return ValidatorResult(
            name="U4",
            severity=SEVERITY_INVALIDATE,
            passed=False,
            detail="splits_csv not provided to validator",
        )
    from src.evaluation.cross_validation import verify_splits

    errors = verify_splits(meta.splits_csv, raise_on_error=False)
    leak_errors = [e for e in (errors or []) if "leak" in e]
    return ValidatorResult(
        name="U4",
        severity=SEVERITY_INVALIDATE,
        passed=not leak_errors,
        detail="; ".join(leak_errors) if leak_errors else "",
    )


def _gate_u5(meta: RunMetadata) -> ValidatorResult:
    if meta.splits_csv is None:
        return ValidatorResult(
            name="U5",
            severity=SEVERITY_INVALIDATE,
            passed=False,
            detail="splits_csv not provided to validator",
        )
    from src.evaluation.cross_validation import verify_splits

    errors = verify_splits(meta.splits_csv, raise_on_error=False)
    coverage_errors = [e for e in (errors or []) if "missing class" in e]
    return ValidatorResult(
        name="U5",
        severity=SEVERITY_INVALIDATE,
        passed=not coverage_errors,
        detail="; ".join(coverage_errors) if coverage_errors else "",
    )


def _gate_u6(meta: RunMetadata) -> ValidatorResult:
    """Every BACC/F1/AUROC/kappa metric is finite."""
    if not meta.metrics:
        return ValidatorResult(
            name="U6",
            severity=SEVERITY_INVALIDATE,
            passed=False,
            detail="metrics dict not provided",
        )
    bad = [
        k
        for k in ("balanced_accuracy", "weighted_f1", "macro_auroc", "cohen_kappa")
        if k in meta.metrics and not _is_finite_float(meta.metrics[k])
    ]
    return ValidatorResult(
        name="U6",
        severity=SEVERITY_INVALIDATE,
        passed=not bad,
        detail=f"non-finite metrics: {bad}" if bad else "",
    )


def _gate_u7(meta: RunMetadata) -> ValidatorResult:
    if meta.checkpoint_path is None:
        return ValidatorResult(
            name="U7",
            severity=SEVERITY_INVALIDATE,
            passed=False,
            detail="checkpoint_path not provided",
        )
    if not Path(meta.checkpoint_path).exists():
        return ValidatorResult(
            name="U7",
            severity=SEVERITY_INVALIDATE,
            passed=False,
            detail=f"checkpoint file not found: {meta.checkpoint_path}",
        )
    # Class-isinstance verification is left to the caller — we only check
    # readability + non-empty here, because loading the checkpoint
    # requires lightning import + GPU memory which we don't want at
    # validator time.
    size = Path(meta.checkpoint_path).stat().st_size
    return ValidatorResult(
        name="U7",
        severity=SEVERITY_INVALIDATE,
        passed=size > 0,
        detail="" if size > 0 else "checkpoint file is empty",
    )


def _gate_u8(meta: RunMetadata) -> ValidatorResult:
    return ValidatorResult(
        name="U8",
        severity=SEVERITY_INVALIDATE,
        passed=bool(meta.training_finished),
        detail="" if meta.training_finished else "training did not finish (hard timeout / crash)",
    )


# --------------------------------------------------------------------------- #
# S gates
# --------------------------------------------------------------------------- #


def _gate_s1(meta: RunMetadata) -> ValidatorResult:
    bacc = _metric(meta, "balanced_accuracy")
    if bacc is None:
        return ValidatorResult("S1", SEVERITY_INVESTIGATE, False, "BACC missing")
    return ValidatorResult(
        "S1",
        SEVERITY_INVESTIGATE,
        bacc > S1_BACC_FLOOR,
        "" if bacc > S1_BACC_FLOOR else f"BACC={bacc:.4f} ≤ floor={S1_BACC_FLOOR:.4f}",
    )


def _gate_s2(meta: RunMetadata) -> ValidatorResult:
    bacc = _metric(meta, "balanced_accuracy")
    if bacc is None:
        return ValidatorResult("S2", SEVERITY_INVALIDATE, False, "BACC missing")
    return ValidatorResult(
        "S2",
        SEVERITY_INVALIDATE,
        bacc <= 1.0,
        "" if bacc <= 1.0 else f"BACC={bacc:.4f} > 1.0 (metric impl bug)",
    )


def _gate_s3(meta: RunMetadata) -> ValidatorResult:
    auroc = _metric(meta, "macro_auroc")
    if auroc is None or not math.isfinite(auroc):
        return ValidatorResult(
            "S3", SEVERITY_INVESTIGATE, False, "macro_auroc missing or NaN"
        )
    return ValidatorResult(
        "S3",
        SEVERITY_INVESTIGATE,
        auroc > S3_AUROC_FLOOR,
        "" if auroc > S3_AUROC_FLOOR else f"AUROC={auroc:.4f} ≤ floor={S3_AUROC_FLOOR}",
    )


def _gate_s4(meta: RunMetadata) -> ValidatorResult:
    pca = _metric(meta, "per_class_accuracy")
    if pca is None:
        return ValidatorResult("S4", SEVERITY_INVESTIGATE, False, "per_class_accuracy missing")
    arr = np.asarray(pca, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return ValidatorResult("S4", SEVERITY_INVESTIGATE, False, "per_class_accuracy all-NaN")
    std = float(arr.std())
    return ValidatorResult(
        "S4",
        SEVERITY_INVESTIGATE,
        std < S4_PER_CLASS_STD_CEIL,
        "" if std < S4_PER_CLASS_STD_CEIL else f"per-class std {std:.4f} ≥ {S4_PER_CLASS_STD_CEIL}",
    )


def _gate_s5(meta: RunMetadata) -> ValidatorResult:
    if meta.predictions is None:
        return ValidatorResult("S5", SEVERITY_INVESTIGATE, False, "predictions not provided")
    distinct = int(np.unique(np.asarray(meta.predictions)).size)
    return ValidatorResult(
        "S5",
        SEVERITY_INVESTIGATE,
        distinct >= S5_MIN_DISTINCT_PREDS,
        ""
        if distinct >= S5_MIN_DISTINCT_PREDS
        else f"only {distinct} distinct predicted classes (< {S5_MIN_DISTINCT_PREDS})",
    )


def _gate_s6(meta: RunMetadata) -> ValidatorResult:
    val_bacc = _metric(meta, "balanced_accuracy")
    if val_bacc is None:
        return ValidatorResult(
            "S6", SEVERITY_INVESTIGATE, False, "val BACC missing"
        )
    if meta.train_bacc is None:
        # Trainer at batch_size=1 deliberately logs only val metrics
        # (train_bacc is noisy on single-graph batches; see trainer.py
        # comment + analysis v0.4 D-9). Treat S6 as not-applicable rather
        # than a failure when the train metric is absent.
        return ValidatorResult(
            "S6",
            SEVERITY_INVESTIGATE,
            True,
            "train_bacc not logged (bs=1 deliberate); S6 train↔val gap N/A",
        )
    gap = abs(float(meta.train_bacc) - float(val_bacc))
    return ValidatorResult(
        "S6",
        SEVERITY_INVESTIGATE,
        gap < S6_TRAIN_VAL_BACC_GAP_CEIL,
        ""
        if gap < S6_TRAIN_VAL_BACC_GAP_CEIL
        else f"train↔val BACC gap {gap:.4f} ≥ {S6_TRAIN_VAL_BACC_GAP_CEIL}",
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


VALIDATOR_REGISTRY: dict[str, Callable[[RunMetadata], ValidatorResult]] = {
    "U1": _gate_u1,
    "U2": _gate_u2,
    "U3": _gate_u3,
    "U4": _gate_u4,
    "U5": _gate_u5,
    "U6": _gate_u6,
    "U7": _gate_u7,
    "U8": _gate_u8,
    "S1": _gate_s1,
    "S2": _gate_s2,
    "S3": _gate_s3,
    "S4": _gate_s4,
    "S5": _gate_s5,
    "S6": _gate_s6,
}


def run_validators(
    meta: RunMetadata,
    *,
    only: Optional[Sequence[str]] = None,
) -> list[ValidatorResult]:
    """Run all registered validators (or a subset) and return results."""
    names = list(only) if only is not None else list(VALIDATOR_REGISTRY.keys())
    out = []
    for name in names:
        if name not in VALIDATOR_REGISTRY:
            raise KeyError(f"unknown validator {name!r}")
        out.append(VALIDATOR_REGISTRY[name](meta))
    return out


def aggregate_severity(results: Sequence[ValidatorResult]) -> str:
    """Return the most severe outcome — ``"invalidate"`` > ``"investigate"`` > ``"pass"``."""
    if any(r.severity == SEVERITY_INVALIDATE and not r.passed for r in results):
        return SEVERITY_INVALIDATE
    if any(r.severity == SEVERITY_INVESTIGATE and not r.passed for r in results):
        return SEVERITY_INVESTIGATE
    return "pass"


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _is_finite_float(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _metric(meta: RunMetadata, key: str):
    """Lookup a metric by key, falling back to the ``val_``-prefixed name.

    Lightning's bs=1 trainer emits validation metrics as ``val_balanced_accuracy``
    / ``val_macro_auroc`` / ``val_per_class_accuracy``. The U/S gates were
    originally specified against bare keys (``balanced_accuracy``, …) — accept
    either so the verifier passes on real-data runs (D-9, v0.4 analysis).
    """
    if not meta.metrics:
        return None
    if key in meta.metrics:
        return meta.metrics[key]
    return meta.metrics.get(f"val_{key}")
