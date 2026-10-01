"""Lightning training module + trainer factory (Phase 7.1, 7.5, 7.6).

:class:`VetGigaGraphLitModule` is a :class:`pytorch_lightning.LightningModule`
wrapping any model with the same calling convention as
:class:`src.models.VetGigaGraph` or any baseline:

* ``model.forward(*args)`` returning either ``logits[C]`` (baselines)
  or ``(logits[C], attention_dict)`` (VetGigaGraph).
* The Lightning module accepts a ``forward_fn`` so callers can adapt
  argument shapes without subclassing.

The factory :func:`build_trainer` assembles the locked training stack:

* class-weighted CE loss from :mod:`src.training.loss`
* AdamW optimizer + WarmupCosineRestarts scheduler from :mod:`src.training.scheduler`
* NaNGuard + EarlyStopping + ModelCheckpoint callbacks from :mod:`src.training.callbacks`
* Mixed precision (``fp16`` by default, configurable to ``bf16`` / ``32``)
* Optional WandB logger

References:
    Design: docs/02-design/03-architecture.md §6 (Module E)
    Tests:  docs/02-design/03-architecture.md §6.E
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)

from src.training.callbacks import (
    NaNGuard,
    build_early_stopping,
    build_model_checkpoint,
)
from src.training.loss import build_class_weighted_ce_loss
from src.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# LightningModule
# --------------------------------------------------------------------------- #


class VetGigaGraphLitModule(LightningModule):
    """Generic Lightning wrapper for VetGigaGraph or a baseline model.

    Args:
        model: The wrapped ``nn.Module``. Must accept the arguments
            produced by ``forward_fn(batch)`` and return either a
            tensor of logits or ``(logits, attention_dict)``.
        loss_fn: A criterion accepting ``(logits[C], target[1])`` and
            returning a scalar loss. Use
            :func:`src.training.loss.build_class_weighted_ce_loss`.
        forward_fn: Maps a Lightning ``batch`` to the positional
            arguments of ``model.forward``. Default: pass the batch
            unchanged (works for baseline ``forward(tile_embeddings)``).
        learning_rate / weight_decay / scheduler kwargs: Optimizer and
            scheduler hyperparameters; mirror ``configs/default.yaml``
            ``training``.
        num_classes: Locked at 7 for VetGigaGraph; configurable for
            tests with smaller class counts.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        forward_fn: Optional[Callable[[Any], tuple]] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn", "forward_fn"])
        self.model = model
        self.loss_fn = loss_fn
        self._forward_fn = forward_fn or _default_forward_fn
        self.num_classes = int(num_classes)

        # Metrics (val/test) — train metrics are noisy on bs=1, so we only
        # track loss during training.
        self.val_balanced_accuracy = MulticlassAccuracy(
            num_classes=num_classes, average="macro"
        )
        self.val_weighted_f1 = MulticlassF1Score(
            num_classes=num_classes, average="weighted"
        )

    # --- shared step ----------------------------------------------------- #

    def _shared_step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        args, target = self._forward_fn(batch)
        out = self.model(*args)
        logits = out[0] if isinstance(out, tuple) else out
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if target.ndim == 0:
            target = target.unsqueeze(0)
        loss = self.loss_fn(logits, target)
        # Baselines with a multi-term objective (e.g. ACMIL) expose the extra
        # terms; they apply to training only so val_loss stays comparable.
        aux = getattr(self.model, "auxiliary_loss", None)
        if aux is not None and self.model.training:
            # e.g. CLAM: 0.7 * slide loss + 0.3 * instance loss.
            loss = getattr(self.model, "bag_loss_weight", 1.0) * loss + aux(target, self.loss_fn)
        return loss, logits, target

    # --- Lightning hooks ------------------------------------------------- #

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        loss, logits, target = self._shared_step(batch)
        preds = logits.argmax(dim=-1)
        self.val_balanced_accuracy.update(preds, target)
        self.val_weighted_f1.update(preds, target)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        return loss

    def on_validation_epoch_end(self) -> None:  # type: ignore[override]
        self.log("val_balanced_accuracy", self.val_balanced_accuracy.compute(), prog_bar=True)
        self.log("val_weighted_f1", self.val_weighted_f1.compute(), prog_bar=True)
        self.val_balanced_accuracy.reset()
        self.val_weighted_f1.reset()

    def configure_optimizers(self):  # type: ignore[override]
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        sched = build_scheduler(
            optim,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


class MultiTaskVetGigaGraphLitModule(VetGigaGraphLitModule):
    """VetGigaGraph trained with an auxiliary per-tile classification head
    (µPDCA #8 Phase C). Saturation falsification test: does tile-level
    supervision lift slide-level performance?

    The aux head reads per-tile features (taken from the GAT's last hidden
    state) and predicts CATCH 13-way categories with masked CE loss
    (``-1`` labels are ignored). Joint loss::

        L = L_slide + aux_lambda * L_tile_masked

    Where ``aux_lambda=0`` reduces to the baseline single-task module
    (sanity invariant). ``L_tile_masked`` is the mean over the mapped
    (non-``-1``) tiles only; if a batch has zero mapped tiles, only
    ``L_slide`` is used.

    Constructor adds:
        aux_lambda: float (default 0.0 — no aux)
        n_tile_classes: int (default 13 = CATCH 1-13 indexed [0..12])
        tile_feature_dim: int (default 128 — must match GAT output_dim)
    """

    def __init__(
        self,
        *,
        aux_lambda: float = 0.0,
        n_tile_classes: int = 13,
        tile_feature_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aux_lambda = float(aux_lambda)
        self.n_tile_classes = int(n_tile_classes)
        # Lightweight tile classifier head (single linear).
        self.tile_head = nn.Linear(tile_feature_dim, n_tile_classes)
        # CE with ignore_index=-1 handles unmapped tiles automatically.
        self.tile_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def _compute_aux_loss(self, batch, tile_features):
        """Compute masked aux loss. tile_features: [N, D] per-tile vectors.
        Returns (loss, n_mapped). loss is 0 when no mapped tiles exist.

        Numerical-stability adaptations (post-M11 NaN root-cause):
            - Aux head + CE loss compute in fp32 even under mixed-precision
              training (the input features stay fp16 from the GAT backward).
            - Explicit no-mapped-tile guard returns zero loss (avoids
              CE(ignore_index=-1) undefined-when-all-ignored NaN).
        """
        # PyG batches: tile_labels lives directly on the Data/Batch object
        tl = getattr(batch, "tile_labels", None)
        if tl is None or tile_features is None or self.aux_lambda <= 0:
            return torch.zeros((), device=tile_features.device if tile_features is not None else "cpu"), 0
        # tl is LongTensor [N], -1 for unmapped, 1-13 for mapped CATCH cats.
        # Shift to [0, 12] for CE (the loss already handles -1 via ignore_index).
        # Mapped values 1-13 → 0-12; -1 stays -1.
        labels = tl.clone()
        mask = labels >= 0
        n_mapped = int(mask.sum().item())
        if n_mapped == 0:
            # All tiles unmapped → CE with ignore_index=-1 produces NaN.
            return torch.zeros((), device=tile_features.device), 0
        labels[mask] = labels[mask] - 1  # 1-13 → 0-12
        # Shape align with tile_features
        if labels.shape[0] != tile_features.shape[0]:
            return torch.zeros((), device=tile_features.device), 0
        # fp32 promotion for numerical stability under mixed precision
        feats_fp32 = tile_features.float()
        logits = self.tile_head(feats_fp32)  # [N, n_tile_classes]
        loss = self.tile_loss_fn(logits, labels.to(tile_features.device))
        # Belt-and-suspenders: zero out NaN/Inf if any slipped through
        if not torch.isfinite(loss):
            return torch.zeros((), device=tile_features.device), n_mapped
        return loss, n_mapped

    def _shared_step_mt(self, batch):
        """Like _shared_step but also extracts per-tile features from the model
        output (assumes the model's forward returns (logits, attention_dict)
        where attention_dict includes 'gnn.last_hidden' [N, D])."""
        args, target = self._forward_fn(batch)
        out = self.model(*args)
        # out can be: logits OR (logits, attn_dict)
        if isinstance(out, tuple):
            logits = out[0]
            attn_dict = out[1] if len(out) > 1 else {}
        else:
            logits = out
            attn_dict = {}
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if target.ndim == 0:
            target = target.unsqueeze(0)
        loss_slide = self.loss_fn(logits, target)

        # Extract per-tile features from attention_dict
        tile_features = attn_dict.get("gnn.last_hidden", None) if isinstance(attn_dict, dict) else None
        aux_loss = torch.zeros((), device=loss_slide.device)
        n_mapped = 0
        if self.aux_lambda > 0 and tile_features is not None:
            aux_loss, n_mapped = self._compute_aux_loss(batch, tile_features)

        total = loss_slide + self.aux_lambda * aux_loss
        return total, loss_slide, aux_loss, n_mapped, logits, target

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        total, l_slide, l_aux, n_mapped, _, _ = self._shared_step_mt(batch)
        self.log("train_loss", total, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log("train_loss_slide", l_slide, on_step=False, on_epoch=True, batch_size=1)
        self.log("train_loss_aux", l_aux, on_step=False, on_epoch=True, batch_size=1)
        self.log("train_n_mapped_tiles", float(n_mapped), on_step=False, on_epoch=True, batch_size=1)
        return total

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        # For validation use slide loss only — aux task is a training-time signal,
        # not a primary metric.
        loss, logits, target = self._shared_step(batch)
        preds = logits.argmax(dim=-1)
        self.val_balanced_accuracy.update(preds, target)
        self.val_weighted_f1.update(preds, target)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        return loss


def _default_forward_fn(batch: Any) -> tuple:
    """Default ``forward_fn``: ``(tile_embeddings, label) → ((tile_embeddings,), label)``.

    Suitable for baselines whose ``forward`` takes a single positional
    tensor. VetGigaGraph callers pass a custom ``forward_fn`` that
    handles ``(graph, embeddings, coords, label)``.
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
        return (x,), torch.as_tensor(y).long().squeeze()
    raise ValueError(
        "Default forward_fn expects batch=(tile_embeddings, label). "
        "Pass a custom forward_fn for VetGigaGraph (graph, embeddings, coords, label) batches."
    )


# --------------------------------------------------------------------------- #
# Trainer factory
# --------------------------------------------------------------------------- #


def build_trainer(
    *,
    max_epochs: int = 100,
    accumulate_grad_batches: int = 8,
    gradient_clip_val: float = 1.0,
    mixed_precision: Union[bool, str] = True,
    callbacks: Optional[Sequence[Callback]] = None,
    logger_obj: Optional[Logger] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    deterministic: bool = True,
    fast_dev_run: bool = False,
    accelerator: str = "auto",
    devices: Union[int, str] = "auto",
) -> Trainer:
    """Construct a PyTorch Lightning Trainer with the locked defaults.

    Args:
        mixed_precision: ``True`` (default) → ``"16-mixed"``;
            ``False`` → ``"32"``; or pass a Lightning precision string
            directly (e.g. ``"bf16-mixed"``).
        callbacks: Override the default callback list. ``None`` →
            ``[NaNGuard, EarlyStopping, ModelCheckpoint]``.
        checkpoint_dir: Where ModelCheckpoint writes ``.ckpt`` files.
            Maps to ``configs/default.yaml paths.checkpoints`` in the
            CLI wrapper.
        deterministic: PyTorch deterministic-algorithms flag. Default
            ``True`` per the design's reproducibility contract.
        fast_dev_run: Lightning's smoke flag — runs a single batch end
            to end. Used by the Phase-7 smoke test.
    """
    precision = _resolve_precision(mixed_precision)
    if callbacks is None:
        callbacks = [
            NaNGuard(),
            build_early_stopping(),
            build_model_checkpoint(dirpath=checkpoint_dir),
        ]
    return Trainer(
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        precision=precision,
        callbacks=list(callbacks),
        logger=logger_obj if logger_obj is not None else False,
        deterministic=deterministic,
        fast_dev_run=fast_dev_run,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def _resolve_precision(mixed_precision: Union[bool, str]) -> str:
    """Coerce common bool/string spellings to a Lightning precision flag."""
    if mixed_precision is True:
        return "16-mixed"
    if mixed_precision is False:
        return "32"
    if isinstance(mixed_precision, str):
        lower = mixed_precision.strip().lower()
        if lower in ("true", "yes", "on", "1"):
            return "16-mixed"
        if lower in ("false", "no", "off", "0"):
            return "32"
        return mixed_precision
    return str(mixed_precision)


# --------------------------------------------------------------------------- #
# Optional WandB logger (lazy import keeps unit tests light)
# --------------------------------------------------------------------------- #


def build_wandb_logger(
    *,
    project: str = "vetgigagraph",
    name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    api_key_required: bool = False,
) -> Optional[Logger]:
    """Return a :class:`WandbLogger` or ``None`` if WandB cannot be configured.

    The default is "best effort": if WandB isn't installed or the API
    key isn't set, log a warning and return ``None`` so the trainer
    runs without remote logging. Pass ``api_key_required=True`` to
    enforce a real WandB run for production training.
    """
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError as e:
        if api_key_required:
            raise
        logger.warning("WandbLogger unavailable (%s); training will run without remote logging.", e)
        return None

    from src.utils.env import get_wandb_api_key

    if api_key_required and get_wandb_api_key() is None:
        raise RuntimeError(
            "WANDB_API_KEY not set; refusing to start training run with api_key_required=True."
        )

    return WandbLogger(project=project, name=name, config=dict(config or {}))
