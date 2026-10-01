"""Hierarchical cell-to-tissue graph (variant 4 of 5).

Two-level structure:

* **Level 1 (patch-level)** — spatial k-NN over individual tiles.
* **Level 2 (region-level)** — k-means cluster centroids of tile
  positions become super-nodes; edges between centroids are spatial
  k-NN at the region scale.
* **DiffPool soft assignment** — a row-stochastic matrix
  ``S ∈ R^{N×K}`` mapping each tile to a probability over regions.
  Each row sums to 1 (test ``test_diffpool_assignment_softmax``).

The implementation here uses k-means + softmax-of-distance as a
deterministic stand-in for the learned DiffPool layer the design
mentions. This keeps the graph constructor side deterministic and unit-
testable; the actual learnable DiffPool / MinCutPool comes in
``src/models/`` (Phase 5).

Device note: the patch / region k-NN and the soft-assignment softmax
run on the configured device (GPU by default). KMeans uses scikit-learn
(CPU-only) so position vectors are detached to CPU for clustering, then
centroids are returned to the working device for ``cdist`` / KNN.

References:
    Design: docs/02-design/03-architecture.md §4 (Graph 4)
    Tests:  docs/02-design/03-architecture.md §4.C (Hierarchical rows)
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from src.graph_construction.base_graph import (
    BaseGraphConstructor,
    DeviceLike,
    knn_indices,
    knn_to_directed_edges,
    symmetrize_directed,
)

_DISTANCE_EPS = 1e-8


class HierarchicalGraph(BaseGraphConstructor):
    """Two-level patch / region graph with a soft DiffPool assignment."""

    def __init__(
        self,
        *,
        k_level1: int = 8,
        k_level2: int = 4,
        n_regions: int = 16,
        kmeans_seed: int = 42,
        softmax_temperature: float = 1.0,
        device: DeviceLike = "auto",
    ) -> None:
        if n_regions < 2:
            raise ValueError(f"n_regions must be >= 2; got {n_regions}")
        if softmax_temperature <= 0:
            raise ValueError(
                f"softmax_temperature must be > 0; got {softmax_temperature}"
            )
        self.k_level1 = int(k_level1)
        self.k_level2 = int(k_level2)
        self.n_regions = int(n_regions)
        self.kmeans_seed = int(kmeans_seed)
        self.softmax_temperature = float(softmax_temperature)
        self.device = device

    def get_config(self) -> dict[str, int | float]:
        return {
            "k_level1": self.k_level1,
            "k_level2": self.k_level2,
            "n_regions": self.n_regions,
            "kmeans_seed": self.kmeans_seed,
            "softmax_temperature": self.softmax_temperature,
        }

    def _build_edges(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        n = x.shape[0]
        n_regions = min(self.n_regions, max(2, n // 2))

        # --- Level 1: patch spatial k-NN.
        nbr = knn_indices(pos, k=self.k_level1)
        directed = knn_to_directed_edges(nbr)
        d = torch.norm(pos[directed[0]] - pos[directed[1]], dim=-1)
        attr = (1.0 / (d + _DISTANCE_EPS)).unsqueeze(-1)
        edge_index, edge_attr = symmetrize_directed(directed, attr)

        # --- Level 2: k-means region centroids + their k-NN.
        centroids = self._kmeans_centroids(pos, n_regions=n_regions)

        # Soft assignment: softmax over negative distance from each tile
        # to each centroid. Rows sum to exactly 1 (test invariant).
        d_to_centroids = torch.cdist(pos, centroids)  # [N, K]
        S = F.softmax(-d_to_centroids / self.softmax_temperature, dim=-1)

        # Region-level node features: weighted sum of tile embeddings.
        # Normalised by S column-sum so each region's feature stays in
        # the same scale as a single tile's embedding.
        col_sum = S.sum(dim=0).clamp(min=_DISTANCE_EPS)  # [K]
        region_x = (S.t() @ x) / col_sum.unsqueeze(-1)  # [K, D]

        # Region-level edges (k-NN over centroids, k bounded by K-1).
        k_l2 = min(self.k_level2, n_regions - 1)
        region_nbr = knn_indices(centroids, k=k_l2)
        region_dir = knn_to_directed_edges(region_nbr)
        d_r = torch.norm(centroids[region_dir[0]] - centroids[region_dir[1]], dim=-1)
        attr_r = (1.0 / (d_r + _DISTANCE_EPS)).unsqueeze(-1)
        region_edge_index, region_edge_attr = symmetrize_directed(region_dir, attr_r)

        return {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "level2_x": region_x,
            "level2_pos": centroids,
            "level2_edge_index": region_edge_index,
            "level2_edge_attr": region_edge_attr,
            "pool_assignment": S,
        }

    # --- internals -------------------------------------------------------- #

    def _kmeans_centroids(
        self,
        pos: torch.Tensor,
        *,
        n_regions: int,
    ) -> torch.Tensor:
        """Run scikit-learn k-means and return ``[K, 2]`` centroid coords.

        sklearn is CPU-only; centroids are shipped back to ``pos.device``
        so downstream ``cdist`` / KNN keep running on the caller's device.
        """
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=n_regions,
            random_state=self.kmeans_seed,
            n_init="auto",
        ).fit(pos.detach().cpu().numpy())
        return torch.as_tensor(
            km.cluster_centers_, dtype=torch.float32, device=pos.device
        )
