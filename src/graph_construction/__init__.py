"""Stage 3 — Graph construction (5 plug-in variants).

Public surface:

* :class:`BaseGraphConstructor` — ABC; subclasses implement ``_build_edges``.
* :class:`SpatialKnnGraph` — variant 1 (spatial k-NN).
* :class:`FeatureSimGraph` — variant 2 (cosine-threshold).
* :class:`DualEdgeGraph` — variant 3 (proposed; spatial ∪ feature).
* :class:`HierarchicalGraph` — variant 4 (2-level + DiffPool).
* :class:`HeterogeneousGraph` — variant 5 (tissue-typed nodes).
* :data:`GRAPH_VARIANTS` — registry of ``name → cls`` for the script wrapper.

See ``docs/02-design/03-architecture.md`` §4 for the design contract.
"""

from src.graph_construction.base_graph import (
    BaseGraphConstructor,
    knn_indices,
    knn_to_directed_edges,
    symmetrize_directed,
)
from src.graph_construction.dual_edge import DualEdgeGraph
from src.graph_construction.feature_sim import FeatureSimGraph
from src.graph_construction.heterogeneous import (
    INTER_TYPE,
    INTRA_TYPE,
    NODE_TYPES,
    HeterogeneousGraph,
)
from src.graph_construction.hierarchical import HierarchicalGraph
from src.graph_construction.spatial_knn import SpatialKnnGraph

#: Locked registry — the ``--graph-type`` CLI flag in
#: ``scripts/03_build_graphs.py`` (Phase 9) maps to these names.
GRAPH_VARIANTS: dict[str, type[BaseGraphConstructor]] = {
    "spatial_knn": SpatialKnnGraph,
    "feature_sim": FeatureSimGraph,
    "dual_edge": DualEdgeGraph,
    "hierarchical": HierarchicalGraph,
    "heterogeneous": HeterogeneousGraph,
}

__all__ = [
    "BaseGraphConstructor",
    "DualEdgeGraph",
    "FeatureSimGraph",
    "GRAPH_VARIANTS",
    "HeterogeneousGraph",
    "HierarchicalGraph",
    "INTER_TYPE",
    "INTRA_TYPE",
    "NODE_TYPES",
    "SpatialKnnGraph",
    "knn_indices",
    "knn_to_directed_edges",
    "symmetrize_directed",
]
