"""Learning-rate scheduler with linear warmup (Phase 7.3).

Wraps :class:`torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`
with a configurable linear warmup phase. During warmup (epochs
``0..warmup_epochs-1``) the LR linearly interpolates from 0 to
``base_lr``; afterwards Cosine annealing with restarts takes over.

The locked default (`configs/default.yaml training`):
    optimizer:           AdamW
    learning_rate:       1e-4
    scheduler:           cosine_warm_restarts
    warmup_epochs:       5

References:
    Design: docs/02-design/03-architecture.md §6.1 (Training settings)
    Tests:  docs/02-design/03-architecture.md §6.E (lr_scheduler_warmup)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineRestarts(LRScheduler):
    """Linear warmup + CosineAnnealingWarmRestarts.

    Args:
        optimizer: The PyTorch optimizer.
        warmup_epochs: Number of epochs over which to linearly ramp LR
            from 0 → ``base_lr``. Must be ``>= 0``.
        max_epochs: Total number of training epochs. Used to size the
            cosine restart period if ``T_0`` is omitted.
        T_0: First cosine restart period (epochs). Defaults to
            ``max_epochs - warmup_epochs`` (no restarts within the run).
        T_mult: Multiplier for restart period. Default ``1`` (constant).
        eta_min: Minimum LR at the trough of cosine. Default ``0``.
        last_epoch: PyTorch-standard sentinel for resuming.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        T_0: Optional[int] = None,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0; got {warmup_epochs}")
        if max_epochs < warmup_epochs:
            raise ValueError(
                f"max_epochs ({max_epochs}) must be >= warmup_epochs ({warmup_epochs})"
            )
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs = int(max_epochs)
        # ``T_0 == 0`` (warmup_epochs == max_epochs) is degenerate but legal —
        # the cosine branch is then unreachable. We still need a positive
        # period to avoid the integer-arithmetic loop below dividing by zero.
        derived_T0 = self.max_epochs - self.warmup_epochs
        self.T_0 = int(T_0) if T_0 is not None else max(1, derived_T0)
        self.T_mult = int(T_mult)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup from 0 to base_lr at epoch == warmup_epochs.
            scale = (epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]

        # Cosine annealing with restarts (post-warmup).
        post = epoch - self.warmup_epochs
        t_cur, period = post, self.T_0
        while t_cur >= period:
            t_cur -= period
            period *= self.T_mult
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * t_cur / period))
        return [
            self.eta_min + (base_lr - self.eta_min) * cos_factor
            for base_lr in self.base_lrs
        ]


def build_scheduler(
    optimizer: Optimizer,
    *,
    name: str = "cosine_warm_restarts",
    warmup_epochs: int = 5,
    max_epochs: int = 100,
    eta_min: float = 0.0,
    T_0: Optional[int] = None,
    T_mult: int = 1,
) -> LRScheduler:
    """Build the locked scheduler.

    Currently only ``cosine_warm_restarts`` is supported per
    ``configs/default.yaml`` ``training.scheduler``.
    """
    if name != "cosine_warm_restarts":
        raise ValueError(
            f"scheduler {name!r} not supported; expected 'cosine_warm_restarts'"
        )
    return WarmupCosineRestarts(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
    )
