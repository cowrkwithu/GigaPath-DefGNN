"""Heterogeneous tissue-aware graph (variant 5 of 5).

Each tile is assigned a tissue type (``0=tumor``, ``1=stroma``,
``2=inflammation``) by an injected classifier (or a deterministic
k-means fallback). Edges are spatial k-NN; ``edge_type`` is ``0`` for
intra-type edges (both endpoints share a tissue type) and ``1`` for
inter-type edges, so a downstream Heterogeneous GNN can route messages
through type-specific transforms.

The design (`03-architecture.md` §4) calls for tissue labels from CATCH
annotations or a separately trained classifier. The classifier
parameter is therefore intentionally pluggable; the k-means fallback
exists only so unit tests can exercise the constructor without a real
tissue model.

Device note: spatial k-NN runs on the configured device (GPU by default).
The k-means fallback uses scikit-learn (CPU-only); embeddings are
detached to CPU for clustering, then the integer labels are returned
to the working device so subsequent indexing stays consistent.

References:
    Design: docs/02-design/03-architecture.md §4 (Graph 5)
    Tests:  docs/02-design/03-architecture.md §4.C (Heterogeneous rows)
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional

import torch

from src.graph_construction.base_graph import (
    BaseGraphConstructor,
    DeviceLike,
    knn_indices,
    knn_to_directed_edges,
    symmetrize_directed,
)

#: Stable order: index → string label. Used by callers that want to
#: deserialize ``data.node_type`` back to human-readable tissue names.
NODE_TYPES = ("tumor", "stroma", "inflammation")
N_NODE_TYPES = len(NODE_TYPES)

#: Edge type tagging.
INTRA_TYPE = 0
INTER_TYPE = 1

_DISTANCE_EPS = 1e-8

#: Type alias for the injectable tissue classifier.
TissueClassifier = Callable[[torch.Tensor], torch.Tensor]


class HeterogeneousGraph(BaseGraphConstructor):
    """Tissue-typed nodes; spatial k-NN edges tagged intra / inter.

    Tissue assignment priority (µPDCA #8 M2):
        1. If ``tile_labels`` are provided AND ``use_gt_labels=True``, GT
           CATCH categories collapse to 3-way; unmapped (-1) tiles fall
           back to k-means on the unmapped subset only.
        2. Else if ``tissue_classifier`` is provided, use it directly.
        3. Else k-means on all embeddings (legacy fallback).
    """

    def __init__(
        self,
        *,
        k: int = 8,
        tissue_classifier: Optional[TissueClassifier] = None,
        kmeans_seed: int = 42,
        device: DeviceLike = "auto",
        use_gt_labels: bool = True,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive; got {k}")
        self.k = int(k)
        self.tissue_classifier = tissue_classifier
        self.kmeans_seed = int(kmeans_seed)
        self.device = device
        self.use_gt_labels = bool(use_gt_labels)

    def get_config(self) -> dict:
        # ``tissue_classifier`` is intentionally omitted — it's a callable
        # that may be a closure over heavy state. Round-trip tests verify
        # only the hashable hyperparameters; callers that need the
        # classifier persisted should serialize it separately.
        return {
            "k": self.k,
            "kmeans_seed": self.kmeans_seed,
            "use_gt_labels": self.use_gt_labels,
        }

    def build_graph(
        self,
        embeddings,
        coordinates,
        *,
        slide_id=None,
        y=None,
        tile_labels=None,
    ):
        """Override to thread ``tile_labels`` through to ``_build_edges``.

        BaseGraphConstructor.build_graph() doesn't know about per-WSI GT
        labels — this thin override stashes them on the instance for the
        single _build_edges() call that follows, then clears them so the
        instance stays stateless across slides.
        """
        self._next_tile_labels = tile_labels
        try:
            return super().build_graph(
                embeddings, coordinates, slide_id=slide_id, y=y
            )
        finally:
            self._next_tile_labels = None

    def _build_edges(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        tile_labels = getattr(self, "_next_tile_labels", None)
        node_type = self._classify_tiles(x, tile_labels=tile_labels)
        if node_type.shape != (x.shape[0],):
            raise ValueError(
                f"tissue_classifier returned shape {tuple(node_type.shape)}, "
                f"expected ({x.shape[0]},)"
            )
        if int(node_type.max()) >= N_NODE_TYPES or int(node_type.min()) < 0:
            raise ValueError(
                f"tissue_classifier emitted labels outside [0, {N_NODE_TYPES})"
            )

        nbr = knn_indices(pos, k=self.k)
        directed = knn_to_directed_edges(nbr)
        d = torch.norm(pos[directed[0]] - pos[directed[1]], dim=-1)
        attr = (1.0 / (d + _DISTANCE_EPS)).unsqueeze(-1)
        edge_index, edge_attr = symmetrize_directed(directed, attr)

        u, v = edge_index[0], edge_index[1]
        same = node_type[u] == node_type[v]
        edge_type = torch.where(
            same,
            torch.full_like(u, INTRA_TYPE),
            torch.full_like(u, INTER_TYPE),
        )

        return {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_type": edge_type,
            "node_type": node_type,
        }

    # --- internals -------------------------------------------------------- #

    def _classify_tiles(self, x: torch.Tensor, tile_labels=None) -> torch.Tensor:
        # Priority 1: GT CATCH labels (µPDCA #8) — collapse to 3-way + k-means
        # for the unmapped subset only. Preserves real tissue identity where
        # available; falls back to clustering only where GT is missing.
        if self.use_gt_labels and tile_labels is not None:
            return self._gt_with_kmeans_fallback(x, tile_labels)
        # Priority 2: injected classifier (kept for testability + future hook).
        if self.tissue_classifier is not None:
            return self.tissue_classifier(x).to(torch.long).to(x.device)
        # Priority 3: legacy k-means on all embeddings.
        return self._kmeans_labels(x)

    def _gt_with_kmeans_fallback(self, x: torch.Tensor, tile_labels) -> torch.Tensor:
        """Use GT CATCH categories (3-way) for mapped tiles; k-means on the
        unmapped subset to keep node_type values inside [0, N_NODE_TYPES).
        """
        import numpy as np
        from src.utils.io_utils import tile_labels_to_3way

        if isinstance(tile_labels, torch.Tensor):
            tile_labels_np = tile_labels.detach().cpu().numpy()
        else:
            tile_labels_np = np.asarray(tile_labels)
        if tile_labels_np.shape != (x.shape[0],):
            raise ValueError(
                f"tile_labels shape {tile_labels_np.shape} does not match "
                f"embeddings ({x.shape[0]},)"
            )
        three_way = tile_labels_to_3way(tile_labels_np)  # [N], values in {-1,0,1,2}
        unmapped = three_way == -1
        if unmapped.any():
            # Cluster only the unmapped subset to keep determinism + speed.
            x_unmapped = x[torch.as_tensor(unmapped, device=x.device)]
            if x_unmapped.shape[0] >= N_NODE_TYPES:
                fallback_labels = self._kmeans_labels(x_unmapped).cpu().numpy()
            else:
                # Too few unmapped tiles to cluster — assign all to "stroma"
                # (most common tissue class per pre-analysis).
                fallback_labels = np.ones(int(unmapped.sum()), dtype=np.int64)
            three_way = three_way.copy()
            three_way[unmapped] = fallback_labels
        return torch.as_tensor(three_way, dtype=torch.long, device=x.device)

    def _kmeans_labels(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic k-means on embeddings → N_NODE_TYPES clusters."""
        # sklearn is CPU-only — round-trip through numpy and ship labels
        # back to whatever device the caller is computing on.
        from sklearn.cluster import KMeans

        n = x.shape[0]
        n_clusters = min(N_NODE_TYPES, n)  # avoid k > N for tiny subsets
        if n_clusters < 2:
            return torch.zeros(n, dtype=torch.long, device=x.device)
        labels = KMeans(
            n_clusters=n_clusters,
            random_state=self.kmeans_seed,
            n_init="auto",
        ).fit_predict(x.detach().cpu().numpy())
        return torch.as_tensor(labels, dtype=torch.long, device=x.device)
