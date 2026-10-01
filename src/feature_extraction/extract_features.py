"""Tile-directory dataset + per-slide feature extraction (Stage 2).

Walks a per-slide tile directory (output of
:class:`src.preprocessing.WSITiler`), runs each tile through the frozen
:class:`GigaPathTileEncoder`, and writes the locked-schema HDF5 file
defined in ``docs/02-design/03-architecture.md`` §3.B.

The dataset reads ``coords.csv`` (the source of truth produced in
Phase 2) rather than globbing PNGs, so embeddings are emitted in the
same order as the coordinate rows. This 1:1 correspondence is enforced
by ``02-data-spec.md`` §6.3.

References:
    Design: docs/02-design/03-architecture.md §3 (Module B)
    Spec:   docs/02-design/02-data-spec.md §6.3 (HDF5 verification)
    Locked HDF5 schema: ``src/utils/io_utils.write_features_h5``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.feature_extraction.gigapath_encoder import (
    GIGAPATH_INPUT_SIZE,
    GigaPathTileEncoder,
)
from src.utils.errors import DataIntegrityError
from src.utils.io_utils import write_features_h5

logger = logging.getLogger(__name__)

#: Default ImageNet mean/std for ViT-style normalization. Matches the
#: GigaPath training-time normalization exactly.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def default_tile_transform(image_size: int = GIGAPATH_INPUT_SIZE) -> Callable:
    """Return a deterministic torchvision transform pipeline.

    Resize → ToTensor → ImageNet-normalize. No augmentations: the
    encoder is frozen, so any randomness here would just slow down
    feature extraction without changing the embeddings.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class TileDirectoryDataset(Dataset):
    """Dataset over a Phase-2 tile directory.

    The directory layout (locked in ``02-data-spec.md`` §6.3):

    .. code-block:: text

        <tiles_dir>/
            coords.csv                      # tile_idx, x, y, tissue_ratio, laplacian_var
            tile_<idx:06d>_<x>_<y>.png      # one PNG per tile, idx zero-padded

    Each ``__getitem__`` returns ``(tile_tensor, tile_idx, x, y)``.
    """

    def __init__(
        self,
        tiles_dir: Union[str, Path],
        *,
        transform: Optional[Callable] = None,
        image_size: int = GIGAPATH_INPUT_SIZE,
    ) -> None:
        import pandas as pd

        self.tiles_dir = Path(tiles_dir)
        coords_csv = self.tiles_dir / "coords.csv"
        if not coords_csv.exists():
            raise DataIntegrityError(
                f"Tile directory missing coords.csv: {self.tiles_dir}"
            )
        self.coords = pd.read_csv(coords_csv)
        self._validate_coords_schema()
        self.transform = transform or default_tile_transform(image_size)

    def _validate_coords_schema(self) -> None:
        required = {"tile_idx", "x", "y", "tissue_ratio", "laplacian_var"}
        missing = required - set(self.coords.columns)
        if missing:
            raise DataIntegrityError(
                f"coords.csv at {self.tiles_dir} missing columns: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int]:
        from PIL import Image

        row = self.coords.iloc[idx]
        tile_idx = int(row["tile_idx"])
        x = int(row["x"])
        y = int(row["y"])
        fname = f"tile_{tile_idx:06d}_{x}_{y}.png"
        path = self.tiles_dir / fname
        if not path.exists():
            raise DataIntegrityError(
                f"coords.csv lists {fname} but file is missing in {self.tiles_dir}"
            )
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, tile_idx, x, y


# --------------------------------------------------------------------------- #
# Encoder runner
# --------------------------------------------------------------------------- #


@torch.no_grad()
def encode_slide(
    encoder: nn.Module,
    tiles_dir: Union[str, Path],
    out_h5: Union[str, Path],
    *,
    batch_size: int = 256,
    num_workers: int = 8,
    device: Optional[Union[str, torch.device]] = None,
    slide_id: Optional[str] = None,
    encoder_version: str = GigaPathTileEncoder.DEFAULT_MODEL,
    magnification: Optional[int] = None,
    transform: Optional[Callable] = None,
) -> Path:
    """Encode every tile in ``tiles_dir`` and write a per-slide HDF5 file.

    Args:
        encoder: A frozen :class:`GigaPathTileEncoder` (or any
            ``nn.Module`` returning ``[B, embedding_dim]``).
        tiles_dir: Phase-2 output directory containing ``coords.csv``
            and ``tile_*.png``.
        out_h5: Destination HDF5 path. Parent dirs are auto-created.
        batch_size: DataLoader batch size. The locked default (256)
            matches ``configs/default.yaml feature_extraction.batch_size``.
        num_workers: DataLoader workers. Set to 0 in tests to keep them
            single-threaded and deterministic.
        device: Where to run inference. ``None`` → use the encoder's
            current device. CUDA is auto-selected if available and the
            encoder is still on CPU.
        slide_id: Override metadata. Defaults to ``Path(tiles_dir).name``.
        encoder_version: Locked metadata field; overridden in tests
            using a stand-in backbone.
        magnification: Optional metadata field (filled by the script
            wrapper that reads the slide's OpenSlide properties).
        transform: Per-tile preprocessing. Defaults to
            :func:`default_tile_transform`. Override only when the
            backbone needs a different normalization.

    Returns:
        Absolute path to the written HDF5 file.
    """
    encoder = encoder.eval()
    device = _resolve_device(encoder, device)
    encoder.to(device)

    embedding_dim = _get_embedding_dim(encoder)

    dataset = TileDirectoryDataset(tiles_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    n_tiles = len(dataset)
    embeddings = np.empty((n_tiles, embedding_dim), dtype=np.float32)
    coordinates = np.empty((n_tiles, 2), dtype=np.int32)
    tile_indices = np.empty((n_tiles,), dtype=np.int32)

    cursor = 0
    for tiles, idxs, xs, ys in loader:
        tiles = tiles.to(device, non_blocking=True)
        emb = encoder(tiles)
        # Defensive cast — torch.no_grad already prevents grad, but tests
        # expect float32 numpy regardless of mixed-precision contexts.
        emb_np = emb.detach().to(torch.float32).cpu().numpy()
        n = emb_np.shape[0]
        embeddings[cursor : cursor + n] = emb_np
        coordinates[cursor : cursor + n, 0] = xs.numpy().astype(np.int32)
        coordinates[cursor : cursor + n, 1] = ys.numpy().astype(np.int32)
        tile_indices[cursor : cursor + n] = idxs.numpy().astype(np.int32)
        cursor += n

    if cursor != n_tiles:
        raise DataIntegrityError(
            f"Expected to encode {n_tiles} tiles but only got {cursor}. "
            "DataLoader silently dropped rows — check num_workers and drop_last."
        )

    sid = slide_id or Path(tiles_dir).name
    metadata = {
        "num_tiles": int(n_tiles),
        "encoder_version": encoder_version,
        "embedding_dim": int(embedding_dim),
    }
    if magnification is not None:
        metadata["magnification"] = int(magnification)

    out_path = write_features_h5(
        out_h5,
        slide_id=sid,
        embeddings=embeddings,
        coordinates=coordinates,
        tile_indices=tile_indices,
        metadata=metadata,
    )
    logger.info(
        "Wrote %d × %d-d embeddings for slide %s → %s",
        n_tiles,
        embedding_dim,
        sid,
        out_path,
    )
    return out_path


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _resolve_device(
    module: nn.Module,
    requested: Optional[Union[str, torch.device]],
) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    # Best-effort: respect the module's current device, but auto-promote
    # to CUDA when the module is on CPU and a GPU is available. This
    # matches the design contract that production runs are GPU-only.
    try:
        current = next(module.parameters()).device
    except StopIteration:
        current = torch.device("cpu")
    if current.type == "cpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return current


def _get_embedding_dim(encoder: nn.Module) -> int:
    """Pull embedding_dim off the encoder; fall back to GigaPath default."""
    return int(getattr(encoder, "embedding_dim", GigaPathTileEncoder.EMBEDDING_DIM))
