"""Unit tests for ``src/feature_extraction/`` (Phase 3, Module B).

Coverage map (design ``docs/02-design/03-architecture.md`` §3.B):

| Test                                  | Phase 3 task | Design §3.B row |
|---------------------------------------|--------------|------------------|
| test_gigapath_loads                   | 3.1          | row 1 (skipped if HF_TOKEN absent) |
| test_encoder_frozen_on_init           | 3.4          | row 2 |
| test_assert_frozen_raises_on_unfreeze | 3.4          | row 2 (extension)  |
| test_embedding_shape                  | 3.1          | row 3 |
| test_embedding_no_nan                 | 3.1          | row 4 |
| test_embedding_l2_norm_bounded        | 3.1          | row 5 |
| test_grad_does_not_flow_into_encoder  | 3.4          | row 6 |
| test_deterministic_encoding           | 3.1          | row 7 |
| test_hdf5_schema                      | 3.3          | row 8 |

The actual GigaPath weights (1.13B params, gated HF Hub model) are not
downloaded in CI. The lightweight tests inject a small stand-in
backbone via ``model=...``; the heavy ``test_gigapath_loads`` is gated
on ``HF_TOKEN`` env var being present.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.feature_extraction import (
    GIGAPATH_EMBEDDING_DIM,
    GigaPathTileEncoder,
    TileDirectoryDataset,
    default_tile_transform,
    encode_slide,
)
from src.utils.errors import FrozenEncoderError
from src.utils.io_utils import read_features_h5

# --------------------------------------------------------------------------- #
# Lightweight stand-in backbone
# --------------------------------------------------------------------------- #


class _TinyBackbone(nn.Module):
    """Deterministic ViT-style stand-in: ``[B, 3, H, W] → [B, embedding_dim]``.

    Architecture:
        Conv7×7 stride 4 → AdaptiveAvgPool → Linear → L2-normalize × 5.

    Design intent: the network is small enough to exercise the encoder
    contract on CPU in <1s, deterministic enough to give bit-identical
    embeddings under a fixed seed, and produces L2 norms in the
    [0.5, 50] band the design contract requires.
    """

    def __init__(self, embedding_dim: int = GIGAPATH_EMBEDDING_DIM) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, embedding_dim)
        self.scale = 5.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        h = self.fc(h)
        # Normalize so L2 norm is exactly self.scale → predictably in
        # the design's [0.5, 50] acceptance band.
        return torch.nn.functional.normalize(h, dim=-1) * self.scale


def _build_test_encoder(seed: int = 0) -> GigaPathTileEncoder:
    torch.manual_seed(seed)
    backbone = _TinyBackbone()
    return GigaPathTileEncoder(model=backbone, pretrained=False)


# --------------------------------------------------------------------------- #
# Encoder tests
# --------------------------------------------------------------------------- #


def test_encoder_frozen_on_init() -> None:
    """All parameters must have ``requires_grad=False`` after construction."""
    enc = _build_test_encoder()
    leaks = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert leaks == [], f"Encoder leaked trainable params: {leaks}"


def test_assert_frozen_raises_on_unfreeze() -> None:
    """Manually re-enabling a parameter must trip the freeze guard."""
    enc = _build_test_encoder()
    next(enc.parameters()).requires_grad = True
    with pytest.raises(FrozenEncoderError, match="trainable params"):
        enc.assert_frozen()


def test_embedding_shape() -> None:
    """A batch of 16 tiles must yield ``[16, 1536]`` float32 embeddings."""
    enc = _build_test_encoder()
    x = torch.randn(16, 3, 224, 224)
    out = enc(x)
    assert out.shape == (16, GIGAPATH_EMBEDDING_DIM)
    assert out.dtype == torch.float32


def test_embedding_no_nan() -> None:
    """No NaN/Inf in encoder output for arbitrary inputs."""
    enc = _build_test_encoder()
    x = torch.randn(8, 3, 224, 224)
    out = enc(x)
    assert torch.isfinite(out).all(), "non-finite values in embeddings"


def test_embedding_l2_norm_bounded() -> None:
    """L2 norm of embeddings must lie in the design contract band [0.5, 50]."""
    enc = _build_test_encoder()
    x = torch.randn(32, 3, 224, 224)
    out = enc(x)
    norm_mean = float(out.norm(dim=-1).mean())
    assert 0.5 < norm_mean < 50.0, f"L2 norm {norm_mean:.3f} outside [0.5, 50]"


def test_grad_does_not_flow_into_encoder() -> None:
    """Backprop through a downstream loss must NOT touch encoder grads."""
    enc = _build_test_encoder()
    classifier = nn.Linear(GIGAPATH_EMBEDDING_DIM, 7)
    x = torch.randn(4, 3, 224, 224)
    logits = classifier(enc(x))
    loss = logits.pow(2).sum()
    loss.backward()
    encoder_leaks = [(n, p.grad) for n, p in enc.named_parameters() if p.grad is not None]
    assert encoder_leaks == [], (
        f"Gradient leaked into frozen encoder for params: {[n for n, _ in encoder_leaks]}"
    )


def test_deterministic_encoding() -> None:
    """Same seed + same input must produce bit-identical embeddings."""
    enc1 = _build_test_encoder(seed=42)
    enc2 = _build_test_encoder(seed=42)
    torch.manual_seed(0)
    x = torch.randn(4, 3, 224, 224)
    out1 = enc1(x)
    out2 = enc2(x)
    assert torch.equal(out1, out2), "Determinism broken across encoder instances"


@pytest.mark.skipif(
    os.environ.get("HF_TOKEN") in (None, "", "hf_replace_this_with_your_actual_token")
    or os.environ.get("VETGIGAGRAPH_RUN_HEAVY_TESTS") != "1",
    reason="GigaPath load is heavy (1.13B params); run with VETGIGAGRAPH_RUN_HEAVY_TESTS=1 + HF_TOKEN set.",
)
def test_gigapath_loads() -> None:  # pragma: no cover — gated by env vars
    """Loads the real Prov-GigaPath checkpoint from HF Hub. Heavy."""
    enc = GigaPathTileEncoder()
    n_params = sum(p.numel() for p in enc.parameters())
    assert n_params > 1.0e9, f"Loaded model has {n_params:,} params, expected ~1.13B"
    assert enc.embedding_dim == GIGAPATH_EMBEDDING_DIM


# --------------------------------------------------------------------------- #
# encode_slide / TileDirectoryDataset tests
# --------------------------------------------------------------------------- #


def _make_tile_dir(tmp_path: Path, n_tiles: int = 4, *, slide_id: str = "MEL_001") -> Path:
    """Synthesise a Phase-2-style tile directory with ``n_tiles`` colored tiles."""
    rng = np.random.default_rng(0)
    slide_dir = tmp_path / slide_id
    slide_dir.mkdir()
    rows = []
    for i in range(n_tiles):
        x, y = i * 256, 0
        img = (rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8))
        Image.fromarray(img).save(slide_dir / f"tile_{i:06d}_{x}_{y}.png")
        rows.append({"tile_idx": i, "x": x, "y": y, "tissue_ratio": 0.9, "laplacian_var": 200.0})
    pd.DataFrame(rows).to_csv(slide_dir / "coords.csv", index=False)
    return slide_dir


def test_dataset_yields_correct_shapes(tmp_path: Path) -> None:
    """Sanity: dataset returns a (tensor, idx, x, y) tuple with right shapes."""
    slide_dir = _make_tile_dir(tmp_path, n_tiles=3)
    ds = TileDirectoryDataset(slide_dir, transform=default_tile_transform(image_size=64))
    assert len(ds) == 3
    tensor, idx, x, y = ds[0]
    assert tensor.shape == (3, 64, 64)
    assert (idx, x, y) == (0, 0, 0)


def test_hdf5_schema(tmp_path: Path) -> None:
    """``encode_slide`` produces an HDF5 file matching the locked schema."""
    slide_dir = _make_tile_dir(tmp_path, n_tiles=5, slide_id="MEL_007")
    enc = _build_test_encoder()
    out_h5 = tmp_path / "features" / "MEL_007.h5"

    encode_slide(
        encoder=enc,
        tiles_dir=slide_dir,
        out_h5=out_h5,
        batch_size=2,
        num_workers=0,
        device="cpu",
        encoder_version="test/_TinyBackbone",
        magnification=40,
    )

    assert out_h5.exists()

    with h5py.File(out_h5, "r") as f:
        assert f["embeddings"].shape == (5, GIGAPATH_EMBEDDING_DIM)
        assert f["embeddings"].dtype == np.float32
        assert f["coordinates"].shape == (5, 2)
        assert f["coordinates"].dtype == np.int32
        assert f["tile_indices"].shape == (5,)
        assert f["tile_indices"].dtype == np.int32

        meta_raw = f["metadata"][()]
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        meta = json.loads(meta_raw)
        assert meta["slide_id"] == "MEL_007"
        assert meta["num_tiles"] == 5
        assert meta["encoder_version"] == "test/_TinyBackbone"
        assert meta["embedding_dim"] == GIGAPATH_EMBEDDING_DIM
        assert meta["magnification"] == 40

    # Round-trip via the io_utils reader to catch any contract drift.
    payload = read_features_h5(out_h5)
    assert payload["coordinates"].shape == (5, 2)
    np.testing.assert_array_equal(payload["tile_indices"], np.arange(5))
