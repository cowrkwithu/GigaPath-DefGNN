"""GraphSlideDataset + DataModule (deferred from Phase 7, landed in Phase 11).

Loads PyG ``Data`` graphs produced by Phase 4
(:mod:`src.graph_construction`) using a split CSV produced by Phase 8
(:func:`src.evaluation.cross_validation.make_5fold_splits`).

Why batch_size = 1: WSIs vary wildly in tile count (and therefore graph
size) — a 50K-tile slide and a 200-tile slide can't be padded into one
batch without distorting attention. The locked
``configs/default.yaml training.batch_size = 1`` reflects this; the
``training.accumulation_steps = 8`` parameter compensates.

Forward-fn helpers wire the dataset's ``Data`` instance into the
positional arguments each model expects:

* :func:`baseline_forward_fn` — `(data.x,)` for ABMIL / DSMIL /
  TransMIL / CLAM.
* :func:`vetgigagraph_forward_fn` — `(data, data.x, data.pos)` for
  :class:`src.models.VetGigaGraph`.

References:
    Design: docs/02-design/03-architecture.md §6.1
    Design: docs/02-design/features/vetgigagraph.do.md Phase 7 + 11
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class GraphSlideDataset(Dataset):
    """One-graph-per-slide dataset, indexed by a split CSV row range.

    Args:
        splits_csv: CSV produced by ``make_5fold_splits``; columns
            ``slide_id, patient_id, tumor_class, fold, split``.
        graphs_root: Parent directory containing ``<slide_id>.pt``.
            Each ``.pt`` is a PyG :class:`Data` saved via
            :func:`src.utils.io_utils.save_pyg_data`.
        fold: Outer-CV fold index (0–4).
        split: One of ``"train"``, ``"val"``, ``"test"``.

    Returns from ``__getitem__``: a single :class:`Data` instance with
    ``y`` populated (graph-level label).
    """

    def __init__(
        self,
        splits_csv: Union[str, Path],
        graphs_root: Union[str, Path],
        *,
        fold: int,
        split: str,
        attach_tile_labels: bool = False,
    ) -> None:
        import pandas as pd

        self.splits_csv = Path(splits_csv)
        self.graphs_root = Path(graphs_root)
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test; got {split!r}")
        df = pd.read_csv(self.splits_csv)
        sub = df[(df["fold"] == fold) & (df["split"] == split)].reset_index(drop=True)
        if sub.empty:
            raise ValueError(
                f"empty split: fold={fold}, split={split!r}, csv={self.splits_csv}"
            )
        # A graph root may list slides the pipeline dropped (e.g. too few
        # tissue tiles at 20x) in ``excluded_slides.csv``; only those are
        # skipped, and any other missing graph still raises in __getitem__.
        excl_csv = self.graphs_root / "excluded_slides.csv"
        if excl_csv.is_file():
            excluded = set(pd.read_csv(excl_csv)["slide_id"].astype(str))
            dropped = sorted(set(sub["slide_id"].astype(str)) & excluded)
            if dropped:
                logger.warning("fold=%d split=%s: excluding %s (listed in %s)",
                               fold, split, dropped, excl_csv)
                sub = sub[~sub["slide_id"].astype(str).isin(excluded)].reset_index(drop=True)
        self._slide_ids = sub["slide_id"].astype(str).tolist()
        self._fold = int(fold)
        self._split = split
        # µPDCA #8 Phase C M10: optional tile-label attachment for multi-task
        # learning. When True, __getitem__ tries to load
        # ``/data/cia_outputs/annotations/tile_labels/<slide>.npy`` and attaches
        # it as ``data.tile_labels`` (LongTensor [N], values: 1-13 CATCH cats
        # or -1 unmapped). Silently leaves the field absent when the file is
        # missing — caller (LitModule) treats this as "no aux loss for this batch".
        self.attach_tile_labels = bool(attach_tile_labels)

    def __len__(self) -> int:
        return len(self._slide_ids)

    def __getitem__(self, idx: int) -> Data:
        slide_id = self._slide_ids[idx]
        pt = self.graphs_root / f"{slide_id}.pt"
        if not pt.exists():
            raise FileNotFoundError(
                f"graph file missing: {pt} (referenced by {self.splits_csv} "
                f"fold={self._fold} split={self._split})"
            )
        data: Data = torch.load(pt, map_location="cpu", weights_only=False)
        if not hasattr(data, "y") or data.y is None:
            raise ValueError(
                f"{pt}: graph has no 'y' attribute (label must be set at build time)."
            )
        if self.attach_tile_labels:
            from src.utils.io_utils import load_tile_labels
            tl = load_tile_labels(slide_id)
            if tl is not None:
                # Convert to LongTensor for CrossEntropyLoss compatibility.
                data.tile_labels = torch.as_tensor(tl, dtype=torch.long)
                data.slide_id = slide_id  # for debug / sanity
        return data


# --------------------------------------------------------------------------- #
# DataModule
# --------------------------------------------------------------------------- #


class GraphSlideDataModule(LightningDataModule):
    """Lightning DataModule wrapping :class:`GraphSlideDataset` per split."""

    def __init__(
        self,
        *,
        splits_csv: Union[str, Path],
        graphs_root: Union[str, Path],
        fold: int,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        attach_tile_labels: bool = False,
    ) -> None:
        super().__init__()
        self.splits_csv = Path(splits_csv)
        self.graphs_root = Path(graphs_root)
        self.fold = int(fold)
        self.num_workers = int(num_workers)
        self.attach_tile_labels = bool(attach_tile_labels)
        # GPU-first: pin host memory when CUDA is available so per-batch
        # host→device transfers can overlap with compute (non-blocking).
        # Single WSI graphs can hit ~365 MB at N≈62k, so the transfer is
        # large enough for pinning to matter. ``None`` → auto-detect;
        # callers can force-disable for benchmarking or constrained hosts.
        self.pin_memory = (
            torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        )
        self._train: Optional[GraphSlideDataset] = None
        self._val: Optional[GraphSlideDataset] = None
        self._test: Optional[GraphSlideDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:  # type: ignore[override]
        if stage in (None, "fit"):
            self._train = GraphSlideDataset(
                self.splits_csv, self.graphs_root, fold=self.fold, split="train",
                attach_tile_labels=self.attach_tile_labels,
            )
            self._val = GraphSlideDataset(
                self.splits_csv, self.graphs_root, fold=self.fold, split="val",
                attach_tile_labels=self.attach_tile_labels,
            )
        if stage in (None, "test"):
            self._test = GraphSlideDataset(
                self.splits_csv, self.graphs_root, fold=self.fold, split="test",
                attach_tile_labels=self.attach_tile_labels,
            )

    # All loaders use batch_size=1 (one graph per WSI; see module docstring).

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_single_graph_collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._val is not None
        return DataLoader(
            self._val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_single_graph_collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        assert self._test is not None
        return DataLoader(
            self._test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_single_graph_collate,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )


def _single_graph_collate(batch: Sequence[Data]) -> Data:
    """Trivial collate for batch_size=1 — return the lone graph unchanged."""
    if len(batch) != 1:
        raise ValueError(
            f"GraphSlideDataModule is locked at batch_size=1; got {len(batch)}"
        )
    return batch[0]


# --------------------------------------------------------------------------- #
# Forward-fn helpers
# --------------------------------------------------------------------------- #


def baseline_forward_fn(batch: Any) -> Tuple[tuple, torch.Tensor]:
    """Adapter for ABMIL / DSMIL / TransMIL / CLAM-SB / CLAM-MB.

    The Lightning batch from :class:`GraphSlideDataModule` is a single
    PyG :class:`Data`. Baselines consume ``data.x`` (tile embeddings)
    and the graph-level label ``data.y``.
    """
    data: Data = batch
    target = data.y
    if target.ndim > 1:
        target = target.squeeze()
    return (data.x,), target.long()


def vetgigagraph_forward_fn(batch: Any) -> Tuple[tuple, torch.Tensor]:
    """Adapter for :class:`src.models.VetGigaGraph`.

    VetGigaGraph's forward signature is
    ``forward(graph, tile_embeddings, coordinates) -> (logits, attn)``.
    We pass ``data.x`` for both ``graph.x`` and ``tile_embeddings``
    since they're the same tensor in this pipeline.
    """
    data: Data = batch
    target = data.y
    if target.ndim > 1:
        target = target.squeeze()
    return (data, data.x, data.pos), target.long()
