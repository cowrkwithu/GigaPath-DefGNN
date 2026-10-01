"""Stage 6 — Evaluation (Phase 8).

Public surface:

* :mod:`metrics` — BACC / weighted F1 / macro AUROC / per-class
  accuracy / Cohen's kappa / confusion matrix (sklearn-matching).
* :mod:`cross_validation` — :func:`make_5fold_splits`,
  :func:`verify_splits`.
* :mod:`statistical_test` — paired_ttest / wilcoxon / mcnemar /
  friedman_nemenyi / bonferroni_correction.
* :mod:`run_validators` — U1–U8 + S1–S6 + :func:`run_validators`.

See ``docs/02-design/03-architecture.md`` §6 and
``docs/02-design/08-statistics-reproducibility.md`` for the design
contract.
"""

from src.evaluation.cross_validation import (
    INNER_K,
    SPLIT_CSV_COLUMNS,
    SPLITS,
    make_5fold_splits,
    verify_splits,
    write_split_csv,
)
from src.evaluation.metrics import (
    CLASS_ORDER,
    NUM_CLASSES,
    balanced_accuracy,
    cohen_kappa,
    compute_confusion_matrix,
    compute_metrics,
    macro_auroc,
    per_class_accuracy,
    weighted_f1,
)
from src.evaluation.run_validators import (
    SEVERITY_INVALIDATE,
    SEVERITY_INVESTIGATE,
    VALIDATOR_REGISTRY,
    RunMetadata,
    ValidatorResult,
    aggregate_severity,
    run_validators,
)
from src.evaluation.statistical_test import (
    MIN_PAIRED_N,
    FriedmanResult,
    TestResult,
    bonferroni_correction,
    friedman_nemenyi,
    mcnemar,
    paired_ttest,
    wilcoxon,
)

__all__ = [
    "CLASS_ORDER",
    "FriedmanResult",
    "INNER_K",
    "MIN_PAIRED_N",
    "NUM_CLASSES",
    "RunMetadata",
    "SEVERITY_INVALIDATE",
    "SEVERITY_INVESTIGATE",
    "SPLITS",
    "SPLIT_CSV_COLUMNS",
    "TestResult",
    "VALIDATOR_REGISTRY",
    "ValidatorResult",
    "aggregate_severity",
    "balanced_accuracy",
    "bonferroni_correction",
    "cohen_kappa",
    "compute_confusion_matrix",
    "compute_metrics",
    "friedman_nemenyi",
    "macro_auroc",
    "make_5fold_splits",
    "mcnemar",
    "paired_ttest",
    "per_class_accuracy",
    "run_validators",
    "verify_splits",
    "weighted_f1",
    "wilcoxon",
    "write_split_csv",
]
