"""Stage 2 — Feature extraction (GigaPath tile encoder).

Public surface:

* :class:`GigaPathTileEncoder` — frozen ViT wrapper, ``[B,3,H,W] → [B,1536]``.
* :class:`TileDirectoryDataset` — reads Phase-2 tile directories.
* :func:`encode_slide` — orchestrator: tiles → HDF5 with locked schema.
* :func:`default_tile_transform` — deterministic ImageNet-norm pipeline.

See ``docs/02-design/03-architecture.md`` §3 for the design contract.
"""

from src.feature_extraction.extract_features import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TileDirectoryDataset,
    default_tile_transform,
    encode_slide,
)
from src.feature_extraction.gigapath_encoder import (
    DEFAULT_GIGAPATH_MODEL,
    GIGAPATH_EMBEDDING_DIM,
    GIGAPATH_INPUT_SIZE,
    GigaPathTileEncoder,
)

__all__ = [
    "DEFAULT_GIGAPATH_MODEL",
    "GIGAPATH_EMBEDDING_DIM",
    "GIGAPATH_INPUT_SIZE",
    "GigaPathTileEncoder",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "TileDirectoryDataset",
    "default_tile_transform",
    "encode_slide",
]
