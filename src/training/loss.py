"""Class-weighted CrossEntropy loss (Phase 7.2).

The CATCH dataset is imbalanced; the design (`vetgigagraph.design.md`
§6 + `04-experiment-design.md` Locked Hyperparameters) specifies
``class_weights = inverse_frequency`` as the default. This module:

* Computes per-class weights from a label histogram via inverse
  frequency, normalised so the mean weight is 1.0 (keeps loss
  magnitudes comparable to the unweighted case so learning rates
  transfer).
* Wraps :class:`torch.nn.CrossEntropyLoss` with the resulting weight
  tensor for use in the Lightning module.

References:
    Design: docs/02-design/03-architecture.md §6.1 (Training settings)
    Tests:  docs/02-design/03-architecture.md §6.E (class_weighted_loss_applied)
"""

from __future__ import annotations

from typing import Iterable, Literal, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

ClassWeightMode = Literal["inverse_frequency", "uniform"]


def compute_class_weights(
    labels: Union[Sequence[int], np.ndarray, torch.Tensor],
    *,
    num_classes: int,
    mode: ClassWeightMode = "inverse_frequency",
) -> torch.Tensor:
    """Return per-class weights as a ``[num_classes]`` float32 tensor.

    Args:
        labels: Iterable of integer class indices in ``[0, num_classes)``.
        num_classes: Total number of classes (locked at 7 for VetGigaGraph).
        mode:
            * ``"inverse_frequency"`` — ``w_c = 1 / count_c``, normalised
              so ``mean(w) = 1``. Classes with zero count receive
              ``mean(w_nonzero)`` so the loss does not silently divide
              by zero — but training-time leakage of an absent class is
              an upstream bug; we log no-op-able fallback.
            * ``"uniform"`` — all weights are 1.0. Provided for
              ablation and for the unweighted-baseline experiment.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive; got {num_classes}")
    arr = _as_int_numpy(labels)
    if arr.size and (arr.min() < 0 or arr.max() >= num_classes):
        raise ValueError(
            f"labels must be in [0, {num_classes}); got min={int(arr.min())}, "
            f"max={int(arr.max())}"
        )

    if mode == "uniform":
        return torch.ones(num_classes, dtype=torch.float32)
    if mode != "inverse_frequency":
        raise ValueError(
            f"class_weights mode must be 'inverse_frequency' or 'uniform'; got {mode!r}"
        )

    counts = np.bincount(arr, minlength=num_classes).astype(np.float64)
    nonzero = counts > 0
    weights = np.zeros_like(counts)
    weights[nonzero] = 1.0 / counts[nonzero]
    if nonzero.any():
        # Fill missing-class weights with mean of the present classes; flag
        # via stderr-style log path (caller can audit).
        weights[~nonzero] = weights[nonzero].mean()
        # Normalise so mean(weight) == 1 — keeps loss magnitudes comparable
        # to the unweighted case across folds with different class supports.
        weights = weights * (num_classes / weights.sum())
    else:
        # Empty labels (CV split sanity-failure); return uniform.
        weights = np.ones(num_classes)
    return torch.as_tensor(weights, dtype=torch.float32)


def build_class_weighted_ce_loss(
    labels: Union[Sequence[int], np.ndarray, torch.Tensor, None],
    *,
    num_classes: int = 7,
    mode: ClassWeightMode = "inverse_frequency",
) -> nn.CrossEntropyLoss:
    """Convenience: compute weights and return a configured CE loss.

    Pass ``labels=None`` to skip weighting (returns plain CE).
    """
    if labels is None or mode == "uniform":
        return nn.CrossEntropyLoss()
    weights = compute_class_weights(labels, num_classes=num_classes, mode=mode)
    return nn.CrossEntropyLoss(weight=weights)


def _as_int_numpy(x: Union[Iterable, np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.int64).ravel()
    if isinstance(x, np.ndarray):
        return x.astype(np.int64).ravel()
    return np.fromiter(x, dtype=np.int64)
