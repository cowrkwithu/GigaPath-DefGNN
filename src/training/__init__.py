"""Stage 5 — Training pipeline (Phase 7).

Public surface:

* :func:`compute_class_weights`, :func:`build_class_weighted_ce_loss`
* :class:`WarmupCosineRestarts`, :func:`build_scheduler`
* :class:`NaNGuard`, :func:`build_early_stopping`, :func:`build_model_checkpoint`
* :class:`VetGigaGraphLitModule`, :func:`build_trainer`, :func:`build_wandb_logger`

See ``docs/02-design/03-architecture.md`` §6 for the design contract.
"""

from src.training.callbacks import (
    NaNGuard,
    EpochHistory,
    build_early_stopping,
    build_model_checkpoint,
)
from src.training.dataset import (
    GraphSlideDataModule,
    GraphSlideDataset,
    baseline_forward_fn,
    vetgigagraph_forward_fn,
)
from src.training.loss import (
    build_class_weighted_ce_loss,
    compute_class_weights,
)
from src.training.scheduler import (
    WarmupCosineRestarts,
    build_scheduler,
)
from src.training.trainer import (
    MultiTaskVetGigaGraphLitModule,
    VetGigaGraphLitModule,
    build_trainer,
    build_wandb_logger,
)

__all__ = [
    "EpochHistory",
    "GraphSlideDataModule",
    "GraphSlideDataset",
    "MultiTaskVetGigaGraphLitModule",
    "NaNGuard",
    "VetGigaGraphLitModule",
    "WarmupCosineRestarts",
    "baseline_forward_fn",
    "build_class_weighted_ce_loss",
    "build_early_stopping",
    "build_model_checkpoint",
    "build_scheduler",
    "build_trainer",
    "build_wandb_logger",
    "compute_class_weights",
    "vetgigagraph_forward_fn",
]
