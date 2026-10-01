"""Deformable-attention GNN backbone for VetGigaGraph v2.

Implements the Fu et al. 2025 deformable-attention idea adapted to irregular
WSI tile graphs (no regular 2D feature map; we use coordinate-aware KNN
interpolation in the tile-coord space).

See `docs/02-design-v2.md` for the design rationale.
"""

from .kernel import coord_knn_sample
from .layer import DeformableAttentionLayer
from .model import (
    DeformableAttentionConfig,
    DeformableAttentionStack,
    VetGigaGraphDeformable,
    build_deformable_backbone,
    build_deformable_model,
)

__all__ = [
    "coord_knn_sample",
    "DeformableAttentionLayer",
    "DeformableAttentionConfig",
    "DeformableAttentionStack",
    "VetGigaGraphDeformable",
    "build_deformable_backbone",
    "build_deformable_model",
]
