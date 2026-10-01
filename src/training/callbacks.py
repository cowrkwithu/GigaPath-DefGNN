"""Lightning callbacks for the VetGigaGraph training loop (Phase 7.4).

* :class:`NaNGuard` — abort the current fold (and persist a marker file)
  when training loss or gradients become non-finite. The design's
  ``test_nan_guard_aborts_fold`` checks that an injected NaN loss
  triggers this callback and the trainer's ``should_stop`` flag.
* :func:`build_early_stopping` — :class:`EarlyStopping` configured to
  the locked defaults from ``configs/default.yaml`` ``training.early_stopping``
  (monitor ``val_balanced_accuracy``, mode ``max``, patience 15).
* :func:`build_model_checkpoint` — :class:`ModelCheckpoint` configured
  to keep the top-3 checkpoints (per ``logging.save_top_k`` in default
  config), monitoring the same primary metric.

References:
    Design: docs/02-design/03-architecture.md §6.1 (Training settings)
    Tests:  docs/02-design/03-architecture.md §6.E (nan_guard, early_stopping)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# NaN guard
# --------------------------------------------------------------------------- #


class NaNGuard(Callback):
    """Abort training when loss or gradients become non-finite.

    Mode of operation:

    * On every ``training_step`` end, inspect ``outputs["loss"]``.
    * On every ``backward`` end, inspect each parameter's ``.grad``
      (sampled — checking every grad is too slow).
    * On a hit, set ``trainer.should_stop = True`` and write a marker
      file (``<log_dir>/NAN_DETECTED``) so the script wrapper can move
      the fold to ``logs/skipped_folds.csv`` per the design contract.
    """

    #: Under fp16 mixed precision, gradients are inspected while still
    #: loss-scaled, and an occasional inf is GradScaler's normal overflow
    #: signal (it skips that step and lowers the scale). Only this many
    #: consecutive non-finite-gradient steps count as divergence.
    AMP_MAX_CONSECUTIVE_BAD_GRADS = 50

    def __init__(self, marker_dir: Optional[str | Path] = None) -> None:
        super().__init__()
        self.marker_dir = Path(marker_dir) if marker_dir is not None else None
        self.triggered = False
        self._bad_grad_streak = 0

    def on_train_batch_end(  # type: ignore[override]
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        loss = _extract_loss(outputs)
        if loss is None:
            return
        if not torch.isfinite(loss).all():
            self._trip(trainer, reason=f"non-finite loss at step {trainer.global_step}")

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Sample 8 params for speed — design contract is "abort cleanly", not
        # "scan every parameter every step" (which would dominate runtime).
        bad = False
        for i, (_, p) in enumerate(pl_module.named_parameters()):
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad = True
                break
            if i >= 7:
                break
        if not bad:
            self._bad_grad_streak = 0
            return
        if not _uses_amp_grad_scaling(trainer):
            self._trip(trainer, reason=f"non-finite grad at step {trainer.global_step}")
            return
        # Before 2026-09-26 this branch tripped on the first inf, aborting
        # every fold that hit a single GradScaler overflow (the published
        # GCN and GIN runs among them).
        self._bad_grad_streak += 1
        if self._bad_grad_streak >= self.AMP_MAX_CONSECUTIVE_BAD_GRADS:
            self._trip(trainer, reason=(
                f"{self._bad_grad_streak} consecutive non-finite (scaled) grads "
                f"at step {trainer.global_step}"))

    def _trip(self, trainer: Trainer, *, reason: str) -> None:
        if self.triggered:
            return
        self.triggered = True
        logger.error("NaNGuard tripped: %s. Aborting fold.", reason)
        trainer.should_stop = True
        if self.marker_dir is not None:
            self.marker_dir.mkdir(parents=True, exist_ok=True)
            (self.marker_dir / "NAN_DETECTED").write_text(reason + "\n", encoding="utf-8")


def _uses_amp_grad_scaling(trainer: Trainer) -> bool:
    """True when fp16 mixed precision (hence a GradScaler) is active."""
    return str(getattr(trainer, "precision", "")) in ("16-mixed", "16")


# --------------------------------------------------------------------------- #
# Early stopping (locked configuration)
# --------------------------------------------------------------------------- #


def build_early_stopping(
    *,
    monitor: str = "val_balanced_accuracy",
    mode: str = "max",
    patience: int = 15,
    min_delta: float = 0.0,
) -> EarlyStopping:
    """Construct EarlyStopping with the locked defaults."""
    return EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
        verbose=True,
    )


# --------------------------------------------------------------------------- #
# Model checkpoint (top-3, locked)
# --------------------------------------------------------------------------- #


def build_model_checkpoint(
    *,
    dirpath: Optional[str | Path] = None,
    monitor: str = "val_balanced_accuracy",
    mode: str = "max",
    save_top_k: int = 3,
    filename: str = "epoch{epoch:03d}-val_bacc{val_balanced_accuracy:.4f}",
) -> ModelCheckpoint:
    """Construct ModelCheckpoint with the locked defaults."""
    return ModelCheckpoint(
        dirpath=str(dirpath) if dirpath is not None else None,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _extract_loss(outputs) -> Optional[torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, dict) and "loss" in outputs:
        v = outputs["loss"]
        if isinstance(v, torch.Tensor):
            return v
    return None


# --------------------------------------------------------------------------- #
# Per-epoch history (loss curves)
# --------------------------------------------------------------------------- #


class EpochHistory(Callback):
    """Append each epoch's logged scalars to a JSON file, rewritten every epoch.

    Gives per-epoch train/val loss curves without depending on a W&B run.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.rows: list[dict] = []

    # Runs after the epoch's validation loop (Lightning 2.x), so both the
    # epoch-level train loss and this epoch's val metrics are available.
    def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        row = {"epoch": int(trainer.current_epoch)}
        for k, v in trainer.callback_metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.rows, indent=1), encoding="utf-8")
