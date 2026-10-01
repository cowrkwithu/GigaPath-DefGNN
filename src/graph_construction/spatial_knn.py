"""Spatial k-NN patch graph (variant 1 of 5).

Each tile becomes a node; edges link a tile to its ``k`` spatially
nearest neighbors (Euclidean distance on coordinates). Edge weights
are inverse distance, so close-by tiles influence each other more
strongly during message passing.

The default ``k=8`` and the symmetrize-without-dedup convention follow
``docs/02-design/04-experiment-design.md`` Locked Hyperparameters and
``docs/02-design/03-architecture.md`` §4.C respectively.

GPU-first: when ``device='auto'`` (default) and CUDA is available, the
k-NN runs on GPU via chunked ``torch.cdist + topk`` (see
``base_graph.knn_indices``). 2D pos KNN is fast on either device, but
running on GPU keeps the whole build_graph path device-consistent.

References:
    Design: docs/02-design/03-architecture.md §4 (Module C, Graph 1)
    Tests:  docs/02-design/03-architecture.md §4.C (Spatial k-NN row)
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

#: Small offset to keep ``1/d`` finite when two tiles share coordinates
#: (which shouldn't happen for non-overlapping grid tiles, but might if a
#: caller passes synthetic data).
_DISTANCE_EPS = 1e-8


class SpatialKnnGraph(BaseGraphConstructor):
    """Spatial k-NN graph: edges link each tile to its ``k`` nearest neighbors."""

    def __init__(self, k: int = 8, *, device: DeviceLike = "auto") -> None:
        if k <= 0:
            raise ValueError(f"k must be positive; got {k}")
        self.k = int(k)
        self.device = device

    def get_config(self) -> dict[str, int]:
        return {"k": self.k}

    def _build_edges(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        nbr = knn_indices(pos, k=self.k)
        directed = knn_to_directed_edges(nbr)
        # 1/d edge weights, computed pre-symmetrization.
        u, v = directed[0], directed[1]
        d = torch.norm(pos[u] - pos[v], dim=-1)
        attr = (1.0 / (d + _DISTANCE_EPS)).unsqueeze(-1)

        edge_index, edge_attr = symmetrize_directed(directed, attr)
        return {"edge_index": edge_index, "edge_attr": edge_attr}
