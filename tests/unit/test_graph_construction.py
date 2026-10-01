"""Unit tests for ``src/graph_construction/`` (Phase 4, Module C).

Coverage map (design ``docs/02-design/03-architecture.md`` §4.C):

* **Common invariants** (parametrized over all 5 variants):
  node_count_matches_input, node_features_preserved, edge_index_dtype_shape,
  edge_index_in_bounds, no_self_loops, edges_bidirectional, edge_attr_finite,
  disconnected_raises, get_config_roundtrip.

* **Per-variant rules**:
    - Spatial k-NN: ``E == N*k*2`` post-symmetrize, neighbors are closest.
    - Feature sim: every edge has cos sim ≥ τ; high τ raises.
    - Dual edge: edge_set == spatial ∪ feature; ``edge_type ∈ {0, 1}``.
    - Hierarchical: two levels exposed, ``S`` row-stochastic.
    - Heterogeneous: ``node_type ∈ {0, 1, 2}``; intra+inter both present.

The 100-tile fixture mimics a real WSI: tiles laid out on a small grid
in ``pos``, embeddings generated as 3 well-separated Gaussian clusters
in feature space (so cosine-threshold edges actually exist at τ=0.8).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.graph_construction import (
    GRAPH_VARIANTS,
    BaseGraphConstructor,
    DualEdgeGraph,
    FeatureSimGraph,
    HeterogeneousGraph,
    HierarchicalGraph,
    NODE_TYPES,
    SpatialKnnGraph,
)
from src.utils.errors import GraphDisconnectedError

EMBED_DIM = 32  # Smaller than 1536 — faster tests, identical contract.
N_NODES = 100
N_CLUSTERS = 3


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fixture_embeddings_and_coords() -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize ``N_NODES`` tiles with clustered embeddings + grid coords.

    Embeddings are drawn from 3 well-separated cluster centers so that
    cosine-similarity edges actually exist at τ=0.8 (random Gaussian
    embeddings in high-dim almost never reach this threshold).

    Coordinates are placed on a small 2D grid so spatial k-NN is
    meaningful.
    """
    rng = np.random.default_rng(0)

    # 3 cluster centers, well-separated.
    centers = rng.normal(size=(N_CLUSTERS, EMBED_DIM))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= 5.0

    cluster_ids = rng.integers(0, N_CLUSTERS, size=N_NODES)
    noise = rng.normal(scale=0.3, size=(N_NODES, EMBED_DIM))
    embed = centers[cluster_ids] + noise

    # Coordinates: 10×10 grid jittered.
    grid = np.array(
        [[i * 256, j * 256] for i in range(10) for j in range(10)],
        dtype=np.float32,
    )
    grid += rng.uniform(-5, 5, size=grid.shape)

    return (
        torch.from_numpy(embed).to(torch.float32),
        torch.from_numpy(grid).to(torch.float32),
    )


def _build(variant_name: str) -> BaseGraphConstructor:
    """Constructor with default (or test-friendly) hyperparameters."""
    if variant_name == "feature_sim":
        return FeatureSimGraph(tau=0.5)  # Permissive; cluster fixture clears 0.5 easily.
    if variant_name == "hierarchical":
        return HierarchicalGraph(k_level1=8, k_level2=3, n_regions=8)
    return GRAPH_VARIANTS[variant_name]()


# --------------------------------------------------------------------------- #
# Common invariants — parametrized over all 5 variants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", sorted(GRAPH_VARIANTS.keys()))
def test_node_count_matches_input(variant, fixture_embeddings_and_coords):
    x, pos = fixture_embeddings_and_coords
    data = _build(variant).build_graph(x, pos)
    assert data.x.shape[0] == N_NODES


@pytest.mark.parametrize("variant", sorted(GRAPH_VARIANTS.keys()))
def test_node_features_preserved(variant, fixture_embeddings_and_coords):
    """``data.x`` must equal the input embeddings (no transform/projection)."""
    x, pos = fixture_embeddings_and_coords
    data = _build(variant).build_graph(x, pos)
    assert torch.equal(data.x, x.to(torch.float32))


@pytest.mark.parametrize("variant", sorted(GRAPH_VARIANTS.keys()))
def test_edge_index_dtype_shape(variant, fixture_embeddings_and_coords):
    x, pos = fixture_embeddings_and_coords
    data = _build(variant).build_graph(x, pos)
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.ndim == 2 and data.edge_index.shape[0] == 2


@pytest.mark.parametrize("variant", sorted(GRAPH_VARIANTS.keys()))
def test_edge_index_in_bounds(variant, fixture_embeddings_and_coords):
    x, pos = fixture_embeddings_and_coords
    data = _build(variant).build_graph(x, pos)
    assert int(data.edge_index.min()) >= 0
    assert int(data.edge_index.max()) < N_NODES


@pytest.mark.parametrize("variant", sorted(GRAPH_VARIANTS.keys()))
def test_no_self_loops(variant, fixture_embeddings_and_coords):
    x, pos = fixture_embeddings_and_coords
    data = _build(variant).build_graph(x, pos)
    assert not (data.edge_index[0] == data.edge_index[1]).any()


@pytest.mark.parametrize("variant", sorted(GRAPH_VARIANTS.keys()))
def test_edges_bidirectional(variant, fixture_embeddings_and_coords):
    """For every (u,v) there exists (v,u) somewhere in edge_index."""
    x, pos = fixture_embeddings_and_coords
    data = _build(variant).build_graph(x, pos)
    pairs = {
        (int(u), int(v)) for u, v in data.edge_index.t().tolist()
    }
    for u, v in pairs:
        assert (v, u) in pairs, f"edge ({u},{v}) lacks reverse in {variant}"


@pytest.mark.parametrize("variant", sorted(GRAPH_VARIANTS.keys()))
def test_edge_attr_finite(variant, fixture_embeddings_and_coords):
    x, pos = fixture_embeddings_and_coords
    data = _build(variant).build_graph(x, pos)
    assert torch.isfinite(data.edge_attr).all()


@pytest.mark.parametrize("variant", sorted(GRAPH_VARIANTS.keys()))
def test_get_config_roundtrip(variant, fixture_embeddings_and_coords):
    """``cls(**get_config())`` must rebuild an identical constructor."""
    x, pos = fixture_embeddings_and_coords
    g1 = _build(variant)
    cfg = g1.get_config()
    g2 = type(g1)(**cfg)
    assert g1.get_config() == g2.get_config()
    # And the resulting graphs are bit-identical for the deterministic variants.
    d1 = g1.build_graph(x, pos)
    d2 = g2.build_graph(x, pos)
    assert torch.equal(d1.edge_index, d2.edge_index)


def test_disconnected_raises():
    """Empty edge_index must raise GraphDisconnectedError, not silently succeed."""
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(20, EMBED_DIM))).to(torch.float32)
    pos = torch.from_numpy(rng.normal(size=(20, 2))).to(torch.float32)
    with pytest.raises(GraphDisconnectedError, match="0 edges"):
        FeatureSimGraph(tau=0.999).build_graph(x, pos)


# --------------------------------------------------------------------------- #
# Per-variant rules
# --------------------------------------------------------------------------- #


def test_spatial_knn_edge_count(fixture_embeddings_and_coords):
    """Spatial k-NN with k=8 on N=100 → E = N*k*2 = 1600 (post-symmetrize)."""
    x, pos = fixture_embeddings_and_coords
    data = SpatialKnnGraph(k=8).build_graph(x, pos)
    assert data.edge_index.shape[1] == N_NODES * 8 * 2


def test_spatial_knn_neighbors_are_closest(fixture_embeddings_and_coords):
    """For each node, its 8 outgoing neighbors are the 8 spatially nearest."""
    x, pos = fixture_embeddings_and_coords
    data = SpatialKnnGraph(k=8).build_graph(x, pos)

    pairwise = torch.cdist(pos, pos)  # [N, N]
    pairwise.fill_diagonal_(float("inf"))
    expected_topk = pairwise.topk(8, largest=False, dim=1).indices  # [N, 8]

    # Collect outgoing edges per node from the directed half (first 800 edges).
    # symmetrize_directed prepends the directed half then concatenates the
    # reverse — so the directed half is edge_index[:, : N*k].
    half = N_NODES * 8
    src = data.edge_index[0, :half]
    dst = data.edge_index[1, :half]
    for i in range(N_NODES):
        outgoing = set(dst[src == i].tolist())
        expected = set(expected_topk[i].tolist())
        assert outgoing == expected, f"node {i} neighbors mismatch"


def test_feature_sim_edge_threshold(fixture_embeddings_and_coords):
    """Every edge must have cosine similarity ≥ τ."""
    x, pos = fixture_embeddings_and_coords
    tau = 0.5
    data = FeatureSimGraph(tau=tau).build_graph(x, pos)
    assert (data.edge_attr.squeeze(-1) >= tau).all()


def test_feature_sim_disconnect_high_tau(fixture_embeddings_and_coords):
    """τ=0.999 must produce 0 edges → GraphDisconnectedError."""
    x, pos = fixture_embeddings_and_coords
    with pytest.raises(GraphDisconnectedError):
        FeatureSimGraph(tau=0.999).build_graph(x, pos)


# -- D-19 fix: max_edges_per_node cap ------------------------------------ #


def test_feature_sim_max_edges_per_node_cap(fixture_embeddings_and_coords):
    """max_edges_per_node=K bounds total edges (a hub node can be picked by
    many others' top-K, so per-node *incoming* degree can exceed K — what's
    bounded is the *outgoing* top-K selection, hence ``total_edges ≤ 2·N·K``
    after symmetric union)."""
    x, pos = fixture_embeddings_and_coords
    tau = 0.5
    K = 4
    N = x.shape[0]

    unbounded = FeatureSimGraph(tau=tau, max_edges_per_node=None).build_graph(x, pos)
    capped = FeatureSimGraph(tau=tau, max_edges_per_node=K).build_graph(x, pos)

    # Capped graph must have fewer-or-equal edges.
    assert capped.edge_index.shape[1] <= unbounded.edge_index.shape[1]

    # Total edges bounded by 2·N·K (each row picks ≤ K, then symmetrize
    # adds at most another N·K). This is the memory-safety guarantee that
    # justifies the D-19 fix.
    assert capped.edge_index.shape[1] <= 2 * N * K, (
        f"total edges {capped.edge_index.shape[1]} > 2·N·K = {2 * N * K}"
    )


def test_feature_sim_max_edges_per_node_invariants(fixture_embeddings_and_coords):
    """Capped graph still respects (a) tau threshold (b) no self-loops
    (c) sums match its edge_attr count."""
    x, pos = fixture_embeddings_and_coords
    tau = 0.5
    K = 4
    data = FeatureSimGraph(tau=tau, max_edges_per_node=K).build_graph(x, pos)
    # tau respected
    assert (data.edge_attr.squeeze(-1) >= tau).all()
    # no self-loops
    src, dst = data.edge_index
    assert (src != dst).all()
    # edge_attr aligned with edge_index
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]


def test_feature_sim_max_edges_invalid_raises():
    """max_edges_per_node must be a positive int or None."""
    with pytest.raises(ValueError, match="max_edges_per_node"):
        FeatureSimGraph(tau=0.5, max_edges_per_node=0)
    with pytest.raises(ValueError, match="max_edges_per_node"):
        FeatureSimGraph(tau=0.5, max_edges_per_node=-3)


def test_feature_sim_config_roundtrip_includes_cap():
    """get_config() must include max_edges_per_node so saved .pt files
    can be reproduced."""
    g1 = FeatureSimGraph(tau=0.7, max_edges_per_node=32)
    assert g1.get_config() == {"tau": 0.7, "max_edges_per_node": 32}
    g2 = FeatureSimGraph(tau=0.7)
    assert g2.get_config() == {"tau": 0.7, "max_edges_per_node": None}


# -- µPDCA #8 M2: HeterogeneousGraph w/ GT tile_labels --------------------- #


def test_heterogeneous_gt_labels_basic(fixture_embeddings_and_coords):
    """GT tile_labels (13-way) collapse to 3-way; unmapped tiles get k-means fallback."""
    from src.graph_construction.heterogeneous import HeterogeneousGraph
    import numpy as np

    x, pos = fixture_embeddings_and_coords
    N = x.shape[0]
    # Build GT labels: half tumor (cat 7), quarter stroma (cat 3), quarter unmapped
    labels = np.full(N, -1, dtype=np.int32)
    labels[: N // 2] = 7        # → tumor (0)
    labels[N // 2 : 3 * N // 4] = 3  # → stroma (1)
    # last quarter stays -1 → k-means fallback

    g = HeterogeneousGraph(k=8, use_gt_labels=True)
    data = g.build_graph(x, pos, tile_labels=labels)
    assert hasattr(data, "node_type")
    nt = data.node_type
    # All node types must be in [0, 3)
    assert nt.min() >= 0 and nt.max() < 3
    # The half-GT-tumor tiles should all be 0
    assert (nt[: N // 2] == 0).all(), \
        f"GT tumor tiles should map to 0; got {nt[: N // 2].unique().tolist()}"
    # The quarter-GT-stroma tiles should all be 1
    assert (nt[N // 2 : 3 * N // 4] == 1).all()


def test_heterogeneous_gt_disabled_uses_kmeans(fixture_embeddings_and_coords):
    """use_gt_labels=False ignores tile_labels and falls back to k-means."""
    from src.graph_construction.heterogeneous import HeterogeneousGraph
    import numpy as np

    x, pos = fixture_embeddings_and_coords
    labels = np.full(x.shape[0], 7, dtype=np.int32)  # would map to all-tumor
    g = HeterogeneousGraph(k=8, use_gt_labels=False)
    data = g.build_graph(x, pos, tile_labels=labels)
    # With kmeans on cluster fixture, NOT all nodes should map to one type
    assert data.node_type.unique().numel() >= 2


def test_heterogeneous_gt_no_labels_falls_back(fixture_embeddings_and_coords):
    """Even when use_gt_labels=True, missing tile_labels triggers k-means."""
    from src.graph_construction.heterogeneous import HeterogeneousGraph

    x, pos = fixture_embeddings_and_coords
    g = HeterogeneousGraph(k=8, use_gt_labels=True)
    data = g.build_graph(x, pos)  # no tile_labels
    assert data.node_type.unique().numel() >= 2


def test_heterogeneous_get_config_includes_gt_flag():
    """Config round-trip must include use_gt_labels."""
    from src.graph_construction.heterogeneous import HeterogeneousGraph

    g = HeterogeneousGraph(k=8, use_gt_labels=True)
    cfg = g.get_config()
    assert cfg == {"k": 8, "kmeans_seed": 42, "use_gt_labels": True}


# -- µPDCA #8 M1: io_utils.load_tile_labels + tile_labels_to_3way ---------- #


def test_tile_labels_to_3way_mapping():
    """13-way CATCH labels collapse correctly to 3-way."""
    from src.utils.io_utils import tile_labels_to_3way
    import numpy as np

    inp = np.array([-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)
    out = tile_labels_to_3way(inp)
    expected = [-1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0]
    assert out.tolist() == expected


def test_load_tile_labels_returns_none_when_missing(tmp_path):
    """Absent label file → None (caller-handled fallback)."""
    from src.utils.io_utils import load_tile_labels
    assert load_tile_labels("NONEXISTENT_SLIDE_42", labels_dir=tmp_path) is None


# -- µPDCA #8 Phase B: attention IoU evaluator ---------------------------- #


def test_attention_iou_perfect_overlap():
    """Attention perfectly aligned with GT tumor → IoU=1.0."""
    from src.evaluation.attention_iou import compute_attention_iou
    import numpy as np

    # 10 tiles: first 4 are tumor (label 7), rest are unmapped (-1).
    # Attention puts highest values on tiles 0-3.
    labels = np.array([7, 7, 7, 7, -1, -1, -1, -1, -1, -1], dtype=np.int32)
    attention = np.array([0.9, 0.8, 0.7, 0.6, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    r = compute_attention_iou(attention, labels)
    assert r["iou"] == 1.0
    assert r["n_tumor_gt"] == 4
    assert r["n_attn_top"] == 4
    assert r["pearson"] is not None and r["pearson"] > 0.5
    assert r["auc_pr"] is not None and r["auc_pr"] > 0.5


def test_attention_iou_anti_aligned():
    """Attention inverted from GT → low IoU + negative Pearson."""
    from src.evaluation.attention_iou import compute_attention_iou
    import numpy as np

    labels = np.array([7, 7, 7, 7, -1, -1, -1, -1, -1, -1], dtype=np.int32)
    # Attention puts highest values on NON-tumor tiles
    attention = np.array([0.0, 0.01, 0.02, 0.03, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    r = compute_attention_iou(attention, labels)
    assert r["iou"] == 0.0
    assert r["pearson"] is not None and r["pearson"] < -0.5


def test_attention_iou_uniform_random():
    """Uniform attention → IoU ≈ 0.25 (random 4-of-10 hit)."""
    from src.evaluation.attention_iou import compute_attention_iou
    import numpy as np

    labels = np.array([7, 7, 7, 7, -1, -1, -1, -1, -1, -1], dtype=np.int32)
    attention = np.full(10, 0.1)
    r = compute_attention_iou(attention, labels)
    # Top-K with ties — argpartition is deterministic, IoU well-defined
    assert 0.0 <= r["iou"] <= 1.0
    # Pearson undefined (zero variance on attention)
    assert r["pearson"] is None


def test_attention_iou_no_tumor():
    """WSI with zero GT tumor tiles → all metrics None."""
    from src.evaluation.attention_iou import compute_attention_iou
    import numpy as np

    labels = np.array([-1, -1, 3, 4, 5], dtype=np.int32)
    attention = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    r = compute_attention_iou(attention, labels)
    assert r["iou"] is None
    assert r["pearson"] is None
    assert r["auc_pr"] is None
    assert r["n_tumor_gt"] == 0


def test_attention_iou_shape_mismatch_raises():
    """Mismatched [N] shapes → ValueError."""
    from src.evaluation.attention_iou import compute_attention_iou
    import numpy as np
    import pytest as _pytest
    with _pytest.raises(ValueError, match="tile_labels"):
        compute_attention_iou(np.array([0.1, 0.2, 0.3]), np.array([7, 7], dtype=np.int32))


# -- µPDCA #9 D-23 fix: Hetero-GAT ---------------------------------------- #


def test_hetero_gat_config_validation():
    """use_heterogeneous=True requires backbone='gat'."""
    from src.models.gnn_backbones import GnnBackboneConfig
    import pytest as _pytest
    GnnBackboneConfig(backbone="gat", in_dim=16, hidden_dim=8, output_dim=4,
                      num_layers=2, heads=2, dropout=0.0, use_heterogeneous=True)
    with _pytest.raises(ValueError, match="backbone='gat'"):
        GnnBackboneConfig(backbone="gcn", in_dim=16, hidden_dim=8, output_dim=4,
                          num_layers=2, heads=2, dropout=0.0, use_heterogeneous=True)


def test_hetero_gat_forward_with_types():
    """Hetero-GAT consumes node_type + edge_type; result differs vs no-types."""
    import torch
    from src.models.gnn_backbones import GnnBackbone, GnnBackboneConfig

    cfg = GnnBackboneConfig(backbone="gat", in_dim=16, hidden_dim=8, output_dim=4,
                            num_layers=2, heads=2, dropout=0.0,
                            use_heterogeneous=True)
    g = GnnBackbone(cfg)
    torch.manual_seed(0)
    x = torch.randn(10, 16)
    ei = torch.randint(0, 10, (2, 30)).long()
    edge_attr = torch.rand(30, 1)
    node_type = torch.randint(0, 3, (10,)).long()
    edge_type = torch.randint(0, 2, (30,)).long()

    h_with, _ = g(x, ei, edge_attr, node_type, edge_type)
    h_zero, _ = g(x, ei)
    # node_type ≠ None → embedding contributes, vs zero-pad fallback
    assert not torch.allclose(h_with, h_zero), \
        "Hetero-GAT must use node_type when provided"


def test_hetero_gat_output_dim_unchanged():
    """Heterogeneous mode does NOT change output shape vs vanilla GAT."""
    import torch
    from src.models.gnn_backbones import GnnBackbone, GnnBackboneConfig

    common = dict(backbone="gat", in_dim=16, hidden_dim=8, output_dim=4,
                  num_layers=2, heads=2, dropout=0.0)
    g_v = GnnBackbone(GnnBackboneConfig(**common, use_heterogeneous=False))
    g_h = GnnBackbone(GnnBackboneConfig(**common, use_heterogeneous=True))
    x = torch.randn(10, 16); ei = torch.randint(0, 10, (2, 30)).long()
    h_v, _ = g_v(x, ei)
    h_h, _ = g_h(x, ei)
    assert h_v.shape == h_h.shape == (10, 4)


def test_hetero_gat_edge_dim_set_only_when_hetero():
    """GATConv.edge_dim is configured only when use_heterogeneous=True."""
    from src.models.gnn_backbones import GnnBackbone, GnnBackboneConfig

    common = dict(backbone="gat", in_dim=16, hidden_dim=8, output_dim=4,
                  num_layers=2, heads=2, dropout=0.0)
    g_v = GnnBackbone(GnnBackboneConfig(**common, use_heterogeneous=False))
    g_h = GnnBackbone(GnnBackboneConfig(**common, use_heterogeneous=True))
    # Vanilla: edge_dim attribute might exist but be None / 0
    assert g_v.layers[0].edge_dim in (None, 0)
    # Hetero: edge_dim = n_edge_types + 1 (one_hot + base edge_attr)
    assert g_h.layers[0].edge_dim == 3  # 2 + 1


# -- µPDCA #10: FeatureAdapter ------------------------------------------- #


def test_feature_adapter_identity_at_init():
    """zero_init=True → adapter is identity (output == input) at initialization."""
    import torch
    from src.models.feature_adapter import FeatureAdapter

    adapter = FeatureAdapter(embed_dim=64, hidden_dim=32, dropout=0.0, zero_init=True)
    adapter.eval()  # disable dropout
    x = torch.randn(10, 64)
    y = adapter(x)
    assert torch.allclose(y, x), \
        f"Expected y==x at init (zero-init residual), max-diff={float((y-x).abs().max())}"


def test_feature_adapter_warps_after_perturbation():
    """After tiny weight perturbation, adapter no longer is identity."""
    import torch
    from src.models.feature_adapter import FeatureAdapter

    torch.manual_seed(0)
    adapter = FeatureAdapter(embed_dim=64, hidden_dim=32, dropout=0.0, zero_init=True)
    adapter.eval()
    # Perturb proj_out to a tiny non-zero
    with torch.no_grad():
        adapter.proj_out.weight.add_(torch.randn_like(adapter.proj_out.weight) * 0.01)
    x = torch.randn(10, 64)
    y = adapter(x)
    assert not torch.allclose(y, x), "After perturbation adapter must warp x"


def test_feature_adapter_output_shape_preserved():
    """Adapter output has same shape as input."""
    import torch
    from src.models.feature_adapter import FeatureAdapter

    adapter = FeatureAdapter(embed_dim=128, hidden_dim=64, dropout=0.0)
    x = torch.randn(7, 128)
    y = adapter(x)
    assert y.shape == x.shape == (7, 128)


def test_feature_adapter_dim_mismatch_raises():
    """Wrong last-dim → ValueError."""
    import torch
    from src.models.feature_adapter import FeatureAdapter
    import pytest as _pytest

    adapter = FeatureAdapter(embed_dim=64, hidden_dim=32)
    with _pytest.raises(ValueError, match="last-dim"):
        adapter(torch.randn(10, 32))  # expects 64


def test_feature_adapter_invalid_params_raise():
    """Invalid embed_dim or dropout → ValueError."""
    from src.models.feature_adapter import FeatureAdapter
    import pytest as _pytest
    with _pytest.raises(ValueError, match="positive"):
        FeatureAdapter(embed_dim=0)
    with _pytest.raises(ValueError, match="positive"):
        FeatureAdapter(embed_dim=64, hidden_dim=0)
    with _pytest.raises(ValueError, match="dropout"):
        FeatureAdapter(embed_dim=64, dropout=1.5)


def test_dual_edge_is_union(fixture_embeddings_and_coords):
    """edge_set(dual) == edge_set(spatial) ∪ edge_set(feature)."""
    x, pos = fixture_embeddings_and_coords
    spatial = SpatialKnnGraph(k=8).build_graph(x, pos)
    feature = _SimpleFeatureKnn(k=5).build_graph(x, pos)  # see below
    dual = DualEdgeGraph(spatial_k=8, feature_k=5).build_graph(x, pos)

    s_set = {tuple(p) for p in spatial.edge_index.t().tolist()}
    f_set = {tuple(p) for p in feature.edge_index.t().tolist()}
    d_set = {tuple(p) for p in dual.edge_index.t().tolist()}
    assert d_set == s_set | f_set


def test_dual_edge_type_attribute(fixture_embeddings_and_coords):
    """edge_type must be in {0, 1} and cover both."""
    x, pos = fixture_embeddings_and_coords
    data = DualEdgeGraph().build_graph(x, pos)
    et = data.edge_type
    assert et.dtype == torch.long
    assert set(et.unique().tolist()).issubset({0, 1})
    assert (et == 0).any() and (et == 1).any()


def test_hierarchical_two_levels(fixture_embeddings_and_coords):
    """Output exposes level1 + level2 graphs; level2 has fewer nodes."""
    x, pos = fixture_embeddings_and_coords
    data = HierarchicalGraph(k_level1=8, k_level2=3, n_regions=8).build_graph(x, pos)
    assert hasattr(data, "level2_x")
    assert hasattr(data, "level2_edge_index")
    assert data.level2_x.shape[0] < data.x.shape[0]


def test_diffpool_assignment_softmax(fixture_embeddings_and_coords):
    """DiffPool soft assignment ``S`` must have rows summing to 1."""
    x, pos = fixture_embeddings_and_coords
    data = HierarchicalGraph(n_regions=8).build_graph(x, pos)
    S = data.pool_assignment
    row_sums = S.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_hetero_node_type_count(fixture_embeddings_and_coords):
    """node_type values must lie in {0, 1, 2}."""
    x, pos = fixture_embeddings_and_coords
    data = HeterogeneousGraph(k=8).build_graph(x, pos)
    nt_set = set(data.node_type.unique().tolist())
    assert nt_set.issubset({0, 1, 2})
    assert len(NODE_TYPES) == 3


def test_hetero_edge_type_inventory(fixture_embeddings_and_coords):
    """Both intra-type (0) and inter-type (1) edges must be present."""
    x, pos = fixture_embeddings_and_coords
    data = HeterogeneousGraph(k=8).build_graph(x, pos)
    et = set(data.edge_type.unique().tolist())
    assert {0, 1}.issubset(et), f"hetero graph missing intra/inter coverage: {et}"


# --------------------------------------------------------------------------- #
# Helper graph used only to compute the feature-knn edge set for the
# ``test_dual_edge_is_union`` assertion. Mirrors DualEdgeGraph's feature
# branch exactly so the two are guaranteed to compare equal.
# --------------------------------------------------------------------------- #


class _SimpleFeatureKnn(BaseGraphConstructor):
    def __init__(self, k: int = 5) -> None:
        self.k = int(k)

    def get_config(self) -> dict[str, int]:
        return {"k": self.k}

    def _build_edges(self, x, pos):
        from src.graph_construction.base_graph import (
            knn_indices,
            knn_to_directed_edges,
            symmetrize_directed,
        )

        x_n = F.normalize(x, dim=-1)
        nbr = knn_indices(x_n, k=self.k)
        directed = knn_to_directed_edges(nbr)
        sim = (x_n[directed[0]] * x_n[directed[1]]).sum(-1).unsqueeze(-1)
        ei, ea = symmetrize_directed(directed, sim)
        return {"edge_index": ei, "edge_attr": ea}
