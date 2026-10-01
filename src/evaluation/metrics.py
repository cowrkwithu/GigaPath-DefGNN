"""Slide-level evaluation metrics (Phase 8.1).

Locked metric set per ``configs/default.yaml evaluation.metrics``:

* **balanced_accuracy** — primary metric (BACC)
* **weighted_f1**
* **macro_auroc** (one-vs-rest, macro-averaged)
* **per_class_accuracy** — diag(C) / row_sum(C)
* **cohen_kappa**
* **confusion_matrix** — in locked class order ``[MEL, MCT, SCC, PNST, PLC, TRB, HIS]``

Every metric is implemented as a thin wrapper over ``sklearn.metrics``
with the locked label list. The unit tests assert exact equality with
sklearn (``rtol=0``) so any drift is caught immediately.

References:
    Design: docs/02-design/03-architecture.md §6.2 (Evaluation metrics)
    Tests:  docs/02-design/03-architecture.md §6.E (8 metric tests)
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.utils.io_utils import INT_TO_LABEL, LABEL_TO_INT

logger = logging.getLogger(__name__)

#: Locked class order for the confusion matrix (axis labels).
#: Mirrors :data:`src.utils.io_utils.LABEL_TO_INT` keys, sorted by integer index.
CLASS_ORDER: tuple[str, ...] = tuple(
    INT_TO_LABEL[i] for i in range(len(INT_TO_LABEL))
)
NUM_CLASSES: int = len(CLASS_ORDER)
LABELS_INT: tuple[int, ...] = tuple(range(NUM_CLASSES))


# --------------------------------------------------------------------------- #
# Coercion
# --------------------------------------------------------------------------- #


def _to_int_array(x: Union[Sequence, np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.int64).ravel()
    if isinstance(x, np.ndarray):
        return x.astype(np.int64).ravel()
    return np.asarray(list(x), dtype=np.int64).ravel()


def _to_float_array(x: Union[Sequence, np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    return np.asarray(list(x), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Individual metric helpers (each calls sklearn 1:1)
# --------------------------------------------------------------------------- #


def balanced_accuracy(y_true, y_pred) -> float:
    """Macro-averaged recall (a.k.a. balanced accuracy)."""
    yt, yp = _to_int_array(y_true), _to_int_array(y_pred)
    return float(balanced_accuracy_score(yt, yp))


def weighted_f1(y_true, y_pred) -> float:
    yt, yp = _to_int_array(y_true), _to_int_array(y_pred)
    return float(
        f1_score(yt, yp, labels=LABELS_INT, average="weighted", zero_division=0)
    )


def macro_auroc(y_true, y_prob) -> float:
    """One-vs-rest macro-averaged AUROC.

    Returns ``float('nan')`` when fewer than 2 classes are present in
    ``y_true`` (sklearn raises in that case — we surface it as NaN so
    the U6 validator can flag the run instead of crashing).
    """
    yt = _to_int_array(y_true)
    yp = _to_float_array(y_prob)
    if yp.ndim != 2 or yp.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"y_prob must be [N, {NUM_CLASSES}]; got {yp.shape}"
        )
    present = np.unique(yt)
    if present.size < 2:
        logger.warning(
            "macro_auroc: only %d class present in y_true; returning NaN", present.size
        )
        return float("nan")
    try:
        return float(
            roc_auc_score(
                yt,
                yp,
                multi_class="ovr",
                average="macro",
                labels=LABELS_INT,
            )
        )
    except ValueError as e:
        # sklearn raises if any present class has no positive — surface as NaN.
        logger.warning("macro_auroc fallback to NaN due to: %s", e)
        return float("nan")


def cohen_kappa(y_true, y_pred) -> float:
    yt, yp = _to_int_array(y_true), _to_int_array(y_pred)
    return float(cohen_kappa_score(yt, yp, labels=LABELS_INT))


def per_class_accuracy(y_true, y_pred) -> np.ndarray:
    """Per-class recall — element ``c`` is ``count(true=c, pred=c) / count(true=c)``.

    Classes absent from ``y_true`` get NaN so the caller can mean over
    only the present classes (matches the sklearn balanced-accuracy
    convention of ignoring absent classes).
    """
    cm = confusion_matrix(
        _to_int_array(y_true),
        _to_int_array(y_pred),
        labels=LABELS_INT,
    )
    row_sum = cm.sum(axis=1)
    out = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        out[c] = cm[c, c] / row_sum[c] if row_sum[c] > 0 else float("nan")
    return out


def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Confusion matrix in the locked class order."""
    return confusion_matrix(
        _to_int_array(y_true),
        _to_int_array(y_pred),
        labels=LABELS_INT,
    )


# --------------------------------------------------------------------------- #
# Combined entry point
# --------------------------------------------------------------------------- #


def compute_metrics(
    y_true,
    y_pred,
    y_prob: Optional[Union[Sequence, np.ndarray, torch.Tensor]] = None,
) -> dict[str, Union[float, np.ndarray]]:
    """Compute all locked slide-level metrics in one call.

    Args:
        y_true: ground-truth class indices (any 1-D sequence).
        y_pred: predicted class indices (argmax of logits).
        y_prob: optional ``[N, num_classes]`` probability matrix for
            macro-AUROC. Pass ``None`` to skip AUROC.

    Returns:
        ``dict`` with keys
        ``balanced_accuracy, weighted_f1, macro_auroc, cohen_kappa,
        per_class_accuracy, confusion_matrix``. AUROC is ``float('nan')``
        if ``y_prob`` is None or only one class is present.
    """
    out: dict[str, Union[float, np.ndarray]] = {
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "weighted_f1": weighted_f1(y_true, y_pred),
        "cohen_kappa": cohen_kappa(y_true, y_pred),
        "per_class_accuracy": per_class_accuracy(y_true, y_pred),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
    }
    out["macro_auroc"] = (
        macro_auroc(y_true, y_prob) if y_prob is not None else float("nan")
    )
    return out


# --------------------------------------------------------------------------- #
# Convenience: pretty-print per-class table
# --------------------------------------------------------------------------- #


def format_per_class_table(per_class: np.ndarray) -> str:
    """Render the per-class accuracy vector with class names (for logs)."""
    lines = [f"{name}: {per_class[i]:.4f}" for i, name in enumerate(CLASS_ORDER)]
    return ", ".join(lines)
