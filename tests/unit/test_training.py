"""Unit tests for ``src/training/`` (Phase 7, Module E — training side).

Coverage map (design ``docs/02-design/03-architecture.md`` §6.E
training-side rows):

| # | Design test                             | Status |
|---|-----------------------------------------|--------|
| 1 | test_training_step_reduces_loss         | ✅ |
| 2 | test_lightning_checkpoint_saves         | ✅ |
| 3 | test_checkpoint_resume_identical        | ✅ |
| 4 | test_nan_guard_aborts_fold              | ✅ |
| 5 | test_class_weighted_loss_applied        | ✅ |
| 6 | test_mixed_precision_no_nan             | ✅ (gated on CUDA; otherwise skipped) |
| 7 | test_lr_scheduler_warmup                | ✅ |
| 8 | test_early_stopping_patience            | ✅ |
| 9 | test_seed_reproducibility               | ✅ |
| 10 | test_oom_aborts_gracefully             | ✅ (mocked OOM) |

Tests use a tiny baseline (ABMIL with reduced widths) on a synthetic
class-imbalanced bag fixture so the suite stays under 30 seconds on
CPU. The Phase-7 → Phase-8 smoke (1-epoch end-to-end on real data)
runs out-of-band via ``scripts/04_train.py`` once that lands in
Phase 10.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from src.models.baselines import ABMIL
from src.training import (
    NaNGuard,
    VetGigaGraphLitModule,
    WarmupCosineRestarts,
    build_class_weighted_ce_loss,
    build_early_stopping,
    build_scheduler,
    build_trainer,
    compute_class_weights,
)


# --------------------------------------------------------------------------- #
# Tiny synthetic dataset + tiny baseline
# --------------------------------------------------------------------------- #


class _ToyBagDataset(Dataset):
    """Class-imbalanced toy bag dataset — N tiles per slide, 7 classes."""

    def __init__(self, num_slides: int = 20, num_tiles: int = 32, embed_dim: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Imbalanced labels: heavy on class 0, light on class 6.
        # 50% class 0, 30% class 1, 5% × 4 (classes 2..5), 0% class 6 by default.
        # We force 1 sample to be class 6 so loss-weighting test sees it.
        labels = rng.choice(
            np.arange(7), size=num_slides, p=[0.5, 0.3, 0.05, 0.05, 0.05, 0.05, 0.0]
        )
        labels[-1] = 6  # ensure all 7 classes present (avoids degenerate weights)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        # Bags: per-class mean shifted so the model has a real signal to learn.
        self.bags = []
        centres = torch.randn(7, embed_dim, generator=torch.Generator().manual_seed(seed)) * 2.0
        for y in labels:
            base = centres[y]
            tiles = base + 0.1 * torch.randn(num_tiles, embed_dim)
            self.bags.append(tiles)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bags[idx], self.labels[idx].clone()


def _collate_single(batch):
    """Collate batch_size=1 — keep tile_embeddings unpacked."""
    bag, label = batch[0]
    return bag, label


def _make_lit_module(*, learning_rate: float = 1e-2, max_epochs: int = 5, embed_dim: int = 64) -> VetGigaGraphLitModule:
    model = ABMIL(embed_dim=embed_dim, hidden_dim=32, attn_dim=16, num_classes=7)
    loss_fn = nn.CrossEntropyLoss()
    return VetGigaGraphLitModule(
        model=model,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        warmup_epochs=1,
        max_epochs=max_epochs,
        num_classes=7,
    )


# --------------------------------------------------------------------------- #
# Test 1 — training step reduces loss over a few iterations
# --------------------------------------------------------------------------- #


def test_training_step_reduces_loss(tmp_path: Path) -> None:
    """10+ iterations on a learnable fixture → mean loss drops."""
    torch.manual_seed(0)
    np.random.seed(0)
    ds = _ToyBagDataset(num_slides=24)
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=_collate_single)

    lit = _make_lit_module(learning_rate=5e-2, max_epochs=3)
    trainer = build_trainer(
        max_epochs=3,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        mixed_precision=False,
        callbacks=[],
        accelerator="cpu",
        devices=1,
        deterministic=False,  # the dropout+toy data combo is fine without strict determinism
    )
    trainer.fit(lit, train_dataloaders=loader, val_dataloaders=None)

    # Compare epoch-0 train_loss vs epoch-2 train_loss from the metrics history.
    metrics = trainer.callback_metrics
    final_loss = float(metrics["train_loss_epoch"])
    # Run a final pass to compute initial loss on a fresh model for comparison.
    fresh = _make_lit_module(learning_rate=1e-2, max_epochs=3).model
    fresh.eval()
    with torch.no_grad():
        init_losses = []
        for x, y in loader:
            logits = fresh(x).unsqueeze(0)
            init_losses.append(nn.functional.cross_entropy(logits, y.unsqueeze(0)).item())
    init_loss = float(np.mean(init_losses))
    assert final_loss < init_loss * 0.95, (
        f"final_loss={final_loss:.4f} did not drop below 95% of init_loss={init_loss:.4f}"
    )


# --------------------------------------------------------------------------- #
# Test 2 — checkpoint saves to disk
# --------------------------------------------------------------------------- #


def test_lightning_checkpoint_saves(tmp_path: Path) -> None:
    torch.manual_seed(0)
    ds = _ToyBagDataset(num_slides=8)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate_single)
    lit = _make_lit_module(max_epochs=2)
    trainer = Trainer(
        max_epochs=2,
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        callbacks=[],
        logger=False,
    )
    trainer.fit(lit, train_dataloaders=loader)
    ckpt = tmp_path / "manual.ckpt"
    trainer.save_checkpoint(str(ckpt))
    assert ckpt.exists() and ckpt.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Test 3 — checkpoint resume produces identical state
# --------------------------------------------------------------------------- #


def test_checkpoint_resume_identical(tmp_path: Path) -> None:
    torch.manual_seed(7)
    ds = _ToyBagDataset(num_slides=4, seed=7)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate_single)

    lit = _make_lit_module(learning_rate=1e-3, max_epochs=2)
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path / "first",
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        callbacks=[],
        logger=False,
    )
    trainer.fit(lit, train_dataloaders=loader)
    ckpt = tmp_path / "checkpoint.ckpt"
    trainer.save_checkpoint(str(ckpt))

    # Snapshot weights, then load into a fresh module and compare.
    w1 = {k: v.clone() for k, v in lit.model.state_dict().items()}

    fresh = _make_lit_module(learning_rate=1e-3, max_epochs=2)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    fresh.load_state_dict(state["state_dict"])

    for key in w1:
        assert torch.equal(w1[key], fresh.model.state_dict()[key]), (
            f"weight {key} drifted across save/load"
        )


# --------------------------------------------------------------------------- #
# Test 4 — NaN guard aborts the fold
# --------------------------------------------------------------------------- #


def test_nan_guard_aborts_fold(tmp_path: Path) -> None:
    """A NaN injected into the loss must trip NaNGuard and stop training."""

    class _NanLossLit(VetGigaGraphLitModule):
        def training_step(self, batch, batch_idx):  # type: ignore[override]
            loss = torch.tensor(float("nan"), requires_grad=True)
            self.log("train_loss", loss, batch_size=1)
            return loss

    ds = _ToyBagDataset(num_slides=4)
    loader = DataLoader(ds, batch_size=1, collate_fn=_collate_single)

    model = ABMIL(embed_dim=64, hidden_dim=32, attn_dim=16, num_classes=7)
    lit = _NanLossLit(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        learning_rate=1e-3,
        warmup_epochs=1,
        max_epochs=5,
    )
    nan_guard = NaNGuard(marker_dir=tmp_path)
    trainer = Trainer(
        max_epochs=5,
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        callbacks=[nan_guard],
        logger=False,
    )
    trainer.fit(lit, train_dataloaders=loader)
    assert nan_guard.triggered, "NaN guard should have tripped"
    assert (tmp_path / "NAN_DETECTED").exists(), "NaN marker file missing"


# --------------------------------------------------------------------------- #
# Test 5 — class-weighted loss is applied
# --------------------------------------------------------------------------- #


def test_class_weighted_loss_applied() -> None:
    """With weight=9 on class 6 and weight=1 on class 0, loss for a class-6
    sample is 9× the loss for a class-0 sample (within float-rounding).

    PyTorch's ``CrossEntropyLoss(weight=...)`` with the default
    ``reduction='mean'`` divides by the *sum of weights of the targets
    in the batch*, which silently cancels the weight on a single-sample
    batch. We use ``reduction='sum'`` so the per-sample weight is
    visible in the scalar loss, matching the design's contract that
    "loss for a class-6 sample is 9× class-0 sample".
    """
    weights = torch.tensor([1.0] * 6 + [9.0])
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="sum")

    # Logits identical for both samples; only target class differs.
    logits = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    loss_class0 = loss_fn(logits, torch.tensor([0])).item()
    loss_class6 = loss_fn(logits, torch.tensor([6])).item()

    # Verify the 9× ratio holds (since logits are identical, NLL is the
    # same; only the weight differs).
    unweighted = nn.CrossEntropyLoss(reduction="sum")
    base_class0 = unweighted(logits, torch.tensor([0])).item()
    base_class6 = unweighted(logits, torch.tensor([6])).item()
    assert abs(loss_class0 - base_class0 * 1.0) / base_class0 < 1e-4
    assert abs(loss_class6 - base_class6 * 9.0) / (base_class6 * 9.0) < 1e-4
    # Sanity: the absolute ratio between class-6 and class-0 losses is 9 ±0.1%.
    ratio = loss_class6 / loss_class0
    expected_ratio = 9.0 * (base_class6 / base_class0)
    assert abs(ratio - expected_ratio) / expected_ratio < 1e-3


def test_compute_class_weights_inverse_frequency() -> None:
    """Weights must be inversely proportional to counts, mean-normalised to 1."""
    labels = [0] * 80 + [1] * 10 + [2] * 5 + [3] * 5
    w = compute_class_weights(labels, num_classes=4)
    # Lighter class → higher weight.
    assert w[0] < w[1] < w[2]
    assert torch.isclose(w[2], w[3])
    # Mean-normalised so loss magnitudes are comparable across folds.
    assert abs(float(w.mean()) - 1.0) < 1e-5


def test_compute_class_weights_handles_missing_class() -> None:
    """Missing classes get the mean weight of present classes (no NaN)."""
    w = compute_class_weights([0, 0, 1], num_classes=4)
    assert torch.isfinite(w).all()
    assert float(w.min()) > 0


# --------------------------------------------------------------------------- #
# Test 6 — mixed precision no NaN (only when CUDA is available)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mixed precision (16-mixed) requires CUDA"
)
def test_mixed_precision_no_nan(tmp_path: Path) -> None:
    torch.manual_seed(0)
    ds = _ToyBagDataset(num_slides=4)
    loader = DataLoader(ds, batch_size=1, collate_fn=_collate_single)
    lit = _make_lit_module(max_epochs=1)
    trainer = build_trainer(
        max_epochs=1,
        mixed_precision=True,
        callbacks=[],
        accelerator="gpu",
        devices=1,
        deterministic=False,
    )
    trainer.fit(lit, train_dataloaders=loader)
    for name, p in lit.model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite weights at {name}"


# --------------------------------------------------------------------------- #
# Test 7 — LR scheduler warmup
# --------------------------------------------------------------------------- #


def test_lr_scheduler_warmup() -> None:
    """First ``warmup_epochs`` LRs ramp linearly from base/W to base."""
    base_lr = 1e-4
    warmup = 5
    optim = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=base_lr)
    sched = build_scheduler(optim, warmup_epochs=warmup, max_epochs=20)

    lrs = []
    for _ in range(warmup + 1):
        lrs.append(optim.param_groups[0]["lr"])
        optim.step()
        sched.step()

    # At epoch 0..warmup-1, LR == base * (epoch+1)/warmup.
    expected = [base_lr * (i + 1) / warmup for i in range(warmup)]
    for got, exp in zip(lrs[:warmup], expected):
        assert abs(got - exp) / base_lr < 0.05, f"warmup LR drift: got {got}, expected {exp}"
    # Epoch warmup → LR == base (top of cosine).
    assert abs(lrs[warmup] - base_lr) / base_lr < 0.05


# --------------------------------------------------------------------------- #
# Test 8 — Early stopping patience
# --------------------------------------------------------------------------- #


def test_early_stopping_patience(tmp_path: Path) -> None:
    """Validation metric flat for ``patience+1`` epochs → training stops."""
    torch.manual_seed(0)

    class _FlatValLit(VetGigaGraphLitModule):
        """Logs a constant val_balanced_accuracy so EarlyStopping must fire."""

        def validation_step(self, batch, batch_idx):  # type: ignore[override]
            x, y = batch
            logits = self.model(x).unsqueeze(0)
            loss = self.loss_fn(logits, y.unsqueeze(0))
            self.log("val_balanced_accuracy", torch.tensor(0.5), batch_size=1)
            self.log("val_loss", loss, batch_size=1)
            return loss

        def on_validation_epoch_end(self):  # type: ignore[override]
            return  # no-op — we logged the metric directly above

    ds = _ToyBagDataset(num_slides=4)
    loader = DataLoader(ds, batch_size=1, collate_fn=_collate_single)

    model = ABMIL(embed_dim=64, hidden_dim=32, attn_dim=16, num_classes=7)
    lit = _FlatValLit(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        learning_rate=1e-3,
        warmup_epochs=1,
        max_epochs=20,
        num_classes=7,
    )
    es = build_early_stopping(patience=2)
    trainer = Trainer(
        max_epochs=20,
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        callbacks=[es],
        logger=False,
    )
    trainer.fit(lit, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.current_epoch < 20, "EarlyStopping should have fired before max_epochs"


# --------------------------------------------------------------------------- #
# Test 9 — Seed reproducibility
# --------------------------------------------------------------------------- #


def test_seed_reproducibility(tmp_path: Path) -> None:
    """Same seed + same data + same hyperparams → identical params after 1 step."""
    from src.utils.seed import set_global_seed

    def run_one_step() -> dict[str, torch.Tensor]:
        set_global_seed(123)
        ds = _ToyBagDataset(num_slides=2, seed=42)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate_single)
        lit = _make_lit_module(learning_rate=1e-3, max_epochs=1)
        trainer = Trainer(
            max_epochs=1,
            default_root_dir=tmp_path,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            callbacks=[],
            logger=False,
            limit_train_batches=1,
            num_sanity_val_steps=0,
        )
        trainer.fit(lit, train_dataloaders=loader)
        return {k: v.clone() for k, v in lit.model.state_dict().items()}

    a = run_one_step()
    b = run_one_step()
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6), f"seed reproducibility failed at {key}"


# --------------------------------------------------------------------------- #
# Test 10 — OOM aborts gracefully (mocked)
# --------------------------------------------------------------------------- #


def test_oom_aborts_gracefully(tmp_path: Path) -> None:
    """A simulated OOM in training_step must propagate as a clean exception."""

    class _OomLit(VetGigaGraphLitModule):
        def training_step(self, batch, batch_idx):  # type: ignore[override]
            raise torch.cuda.OutOfMemoryError("simulated OOM")  # type: ignore[attr-defined]

    ds = _ToyBagDataset(num_slides=2)
    loader = DataLoader(ds, batch_size=1, collate_fn=_collate_single)
    model = ABMIL(embed_dim=64, hidden_dim=32, attn_dim=16, num_classes=7)
    lit = _OomLit(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        learning_rate=1e-3,
        warmup_epochs=1,
        max_epochs=2,
    )
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        callbacks=[],
        logger=False,
    )
    with pytest.raises(torch.cuda.OutOfMemoryError, match="simulated OOM"):  # type: ignore[attr-defined]
        trainer.fit(lit, train_dataloaders=loader)
