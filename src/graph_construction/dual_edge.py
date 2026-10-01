"""Dual-edge hybrid graph (variant 3 of 5) — the proposed core variant.

Combines a spatial k-NN graph (k=8) with a feature k-NN graph (k=5) by
concatenating their edge sets. ``data.edge_type`` tags each edge as
``0`` (spatial) or ``1`` (feature) so a downstream GNN can use
heterogeneous attention if it wishes.

Per design §3 of ``docs/02-design/03-architecture.md``, this is the
proposed novelty over the baseline graphs: spatial proximity captures
tissue context; feature similarity captures phenotypic similarity even
across distant regions.

GPU-first: the feature-space k-NN over D=1536 GigaPath embeddings is
the slowest CPU operation in the whole pipeline (scipy's KDTree
degrades to brute force in high dimensions). On a 24 GB GPU the
chunked cdist+topk path (see ``base_graph._torch_knn_indices``) cuts
per-WSI build time by ~1–2 orders of magnitude.

References:
    Design: docs/02-design/03-architecture.md §4 (Graph 3)
    Tests:  docs/02-design/03-architecture.md §4.C (Dual-edge rows)
"""

from __future__ import annotations

import torch

from src.graph_construction.base_graph import (
    BaseGraphConstructor,
    DeviceLike,
    knn_indices,
    knn_to_directed_edges,
    symmetrize_directed,
)

_DISTANCE_EPS = 1e-8


class DualEdgeGraph(BaseGraphConstructor):
    """Spatial-knn ∪ feature-knn graph with edge_type tagging."""

    SPATIAL_TYPE = 0
    FEATURE_TYPE = 1

    def __init__(
        self,
        spatial_k: int = 8,
        feature_k: int = 5,
        *,
        device: DeviceLike = "auto",
    ) -> None:
        if spatial_k <= 0 or feature_k <= 0:
            raise ValueError(
                f"spatial_k, feature_k must be positive; got {spatial_k}, {feature_k}"
            )
        self.spatial_k = int(spatial_k)
        self.feature_k = int(feature_k)
        self.device = device

    def get_config(self) -> dict[str, int]:
        return {"spatial_k": self.spatial_k, "feature_k": self.feature_k}

    def _build_edges(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # --- spatial branch ---
        spatial_nbr = knn_indices(pos, k=self.spatial_k)
        spatial_dir = knn_to_directed_edges(spatial_nbr)
        d_s = torch.norm(pos[spatial_dir[0]] - pos[spatial_dir[1]], dim=-1)
        attr_s = (1.0 / (d_s + _DISTANCE_EPS)).unsqueeze(-1)
        spatial_idx, spatial_attr = symmetrize_directed(spatial_dir, attr_s)

        # --- feature branch (cosine k-NN via standard Euclidean k-NN
        # on L2-normalized embeddings — equivalent ranking to cosine).
        x_n = torch.nn.functional.normalize(x, dim=-1)
        feature_nbr = knn_indices(x_n, k=self.feature_k)
        feature_dir = knn_to_directed_edges(feature_nbr)
        # Cosine-similarity edge weight (computed from the normalized embeddings).
        sim = (x_n[feature_dir[0]] * x_n[feature_dir[1]]).sum(dim=-1)
        attr_f = sim.unsqueeze(-1)
        feature_idx, feature_attr = symmetrize_directed(feature_dir, attr_f)

        # --- combine ---
        edge_index = torch.cat([spatial_idx, feature_idx], dim=1)
        edge_attr = torch.cat([spatial_attr, feature_attr], dim=0)
        edge_type = torch.cat(
            [
                torch.full(
                    (spatial_idx.shape[1],),
                    self.SPATIAL_TYPE,
                    dtype=torch.long,
                    device=x.device,
                ),
                torch.full(
                    (feature_idx.shape[1],),
                    self.FEATURE_TYPE,
                    dtype=torch.long,
                    device=x.device,
                ),
            ],
            dim=0,
        )
        return {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_type": edge_type,
        }
