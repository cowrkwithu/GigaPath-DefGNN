"""Unit tests for ``src/models/`` (Phase 5, Module D).

Coverage map (design ``docs/02-design/03-architecture.md`` §5.D):

| # | Design test                              | Status / Phase |
|---|------------------------------------------|----------------|
| 1 | test_forward_logits_shape                | ✅ Phase 5     |
| 2 | test_forward_attention_weights_schema    | ✅ Phase 5     |
| 3 | test_no_softmax_in_logits                | ✅ Phase 5     |
| 4 | test_tile_encoder_grad_guard             | ✅ Phase 5     |
| 5 | test_slide_encoder_unfrozen_by_default   | ✅ Phase 5     |
| 6 | test_slide_encoder_frozen_when_configured| ✅ Phase 5     |
| 7 | test_gnn_backbone_factory                | ✅ Phase 5     |
| 8 | test_fusion_factory_all_strategies       | ✅ Phase 5     |
| 9 | test_fusion_alpha_in_unit_interval       | ✅ Phase 5     |
| 10–13 | baseline ABMIL/DSMIL/TransMIL/CLAM    | ✅ Phase 6     |
| 14 | class_weighted_loss_correctness          | ⏭ Phase 7     |
| 15 | test_param_count_within_budget           | ✅ Phase 5     |

Heavy real-GigaPath load is gated separately on `HF_TOKEN +
VETGIGAGRAPH_RUN_HEAVY_TESTS=1` (same convention as Phase 3).
The default test path uses lightweight stand-ins for both encoders.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import pytest
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

from src.feature_extraction.gigapath_encoder import GigaPathTileEncoder
from src.models import (
    DEFAULT_PROJ_DIM,
    SUPPORTED_BACKBONES,
    SUPPORTED_FUSIONS,
    ConcatFusion,
    CrossAttentionFusion,
    FusionConfig,
    GatedFusion,
    GigaPathSlideEncoder,
    GlobalAttentionReadout,
    GnnBackbone,
    GnnBackboneConfig,
    GnnOnlyFusion,
    LearnableWeightedFusion,
    MlpClassifier,
    SlideOnlyFusion,
    VetGigaGraph,
    build_fusion_module,
    build_gnn_backbone,
)
from src.models.fusion import _FusionBase

# --------------------------------------------------------------------------- #
# Stand-in encoders + tiny dimensions for fast unit tests
# --------------------------------------------------------------------------- #

EMBED_DIM = 64        # smaller than 1536 — same contract, fast tests
PROJ_DIM = 32
N_NODES = 16
NUM_CLASSES = 7


class _TinyTileBackbone(nn.Module):
    """Stand-in for the GigaPath tile encoder (Phase 3 already covered freeze contracts)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=7, stride=4, padding=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


class _TinySlideBackbone(nn.Module):
    """Stand-in for the GigaPath slide encoder (LongNet).

    Returns ``(cls_output[embed_dim], cls_attention[N])``. The CLS
    attention is a softmax over a single linear gate, so it always
    sums to 1 — matches the design contract.
    """

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.gate = nn.Linear(embed_dim, 1)
        self.cls = nn.Linear(embed_dim, embed_dim)

    def forward(self, embeddings: torch.Tensor, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # embeddings: [N, embed_dim]
        weights = torch.softmax(self.gate(embeddings).squeeze(-1), dim=0)  # [N]
        pooled = (weights.unsqueeze(-1) * embeddings).sum(dim=0)  # [embed_dim]
        return self.cls(pooled), weights


def _build_test_gnn(backbone: str = "gat", num_layers: int = 3) -> GnnBackbone:
    cfg = GnnBackboneConfig(
        backbone=backbone,
        in_dim=EMBED_DIM,
        hidden_dim=PROJ_DIM,
        output_dim=PROJ_DIM,
        num_layers=num_layers,
        heads=4,
        dropout=0.1,
    )
    return GnnBackbone(cfg)


def _build_test_fusion(strategy: str = "learnable_weighted") -> _FusionBase:
    return build_fusion_module(FusionConfig(strategy=strategy, proj_dim=PROJ_DIM))


def _build_test_slide_encoder(*, frozen: bool = False) -> GigaPathSlideEncoder:
    return GigaPathSlideEncoder(
        model=_TinySlideBackbone(EMBED_DIM),
        embed_dim=EMBED_DIM,
        proj_dim=PROJ_DIM,
        frozen=frozen,
    )


def _build_test_classifier() -> MlpClassifier:
    return MlpClassifier(input_dim=PROJ_DIM, hidden_dim=PROJ_DIM, num_classes=NUM_CLASSES)


def _build_test_vetgigagraph(
    *,
    fusion: str = "learnable_weighted",
    slide_frozen: bool = False,
    num_layers: int = 3,
) -> VetGigaGraph:
    return VetGigaGraph(
        gnn=_build_test_gnn(num_layers=num_layers),
        slide_encoder=_build_test_slide_encoder(frozen=slide_frozen),
        fusion=_build_test_fusion(fusion),
        classifier=_build_test_classifier(),
        embed_dim=EMBED_DIM,
    )


def _make_synthetic_graph() -> tuple[Data, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(N_NODES, EMBED_DIM)
    pos = torch.randn(N_NODES, 2) * 100
    # Simple ring + skip connections so the graph is connected and bidirectional.
    src = torch.arange(N_NODES)
    dst = (src + 1) % N_NODES
    forward = torch.stack([src, dst], dim=0)
    backward = torch.stack([dst, src], dim=0)
    edge_index = torch.cat([forward, backward], dim=1).to(torch.long)
    edge_attr = torch.ones(edge_index.shape[1], 1)
    graph = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    return graph, x, pos


# --------------------------------------------------------------------------- #
# Test 1 — Forward shape (logits = [num_classes])
# --------------------------------------------------------------------------- #


def test_forward_logits_shape() -> None:
    model = _build_test_vetgigagraph()
    graph, embed, pos = _make_synthetic_graph()
    logits, _ = model(graph, embed, pos)
    assert logits.shape == (NUM_CLASSES,)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------- #
# Test 2 — attention_weights schema (locked per design §4.3)
# --------------------------------------------------------------------------- #


def test_forward_attention_weights_schema() -> None:
    """Attention dict must have all required keys, all detached + on CPU."""
    model = _build_test_vetgigagraph(fusion="learnable_weighted")
    graph, embed, pos = _make_synthetic_graph()
    logits, attn = model(graph, embed, pos)

    # GNN per-layer edge attention (3 layers)
    for i in (1, 2, 3):
        key = f"gnn.layer_{i}.edge_attention"
        assert key in attn, f"missing attention key {key}"
        a = attn[key]
        assert a.device.type == "cpu", f"{key} must be CPU, got {a.device}"
        assert not a.requires_grad, f"{key} must be detached"
        assert a.ndim == 2, f"{key} must be [E, H]; got {a.shape}"

    # GNN readout (sums to 1)
    node_attn = attn["gnn.readout.node_attention"]
    assert node_attn.shape == (N_NODES,)
    assert torch.allclose(node_attn.sum(), torch.tensor(1.0), atol=1e-5)

    # Slide CLS attention (uses_slide=True for learnable_weighted)
    cls_attn = attn["slide_encoder.cls_attention"]
    assert cls_attn.shape == (N_NODES,)
    assert cls_attn.device.type == "cpu"

    # Fusion alpha — present and bounded
    alpha = attn["fusion.alpha"]
    assert alpha.ndim == 0
    assert 0.0 <= float(alpha) <= 1.0


def test_attention_omits_slide_for_gnn_only() -> None:
    """gnn_only fusion → no slide_encoder.cls_attention key."""
    model = _build_test_vetgigagraph(fusion="gnn_only")
    graph, embed, pos = _make_synthetic_graph()
    _, attn = model(graph, embed, pos)
    assert "slide_encoder.cls_attention" not in attn
    assert "fusion.alpha" not in attn  # only learnable_weighted exposes alpha


def test_attention_omits_alpha_for_non_learnable_fusions() -> None:
    """Only learnable_weighted exposes fusion.alpha."""
    for strategy in ("gnn_only", "slide_only", "concat", "cross_attention", "gated"):
        model = _build_test_vetgigagraph(fusion=strategy)
        graph, embed, pos = _make_synthetic_graph()
        _, attn = model(graph, embed, pos)
        assert "fusion.alpha" not in attn, f"{strategy} should not expose alpha"


def test_gated_fusion_emits_fusion_gate_key() -> None:
    """`gated` fusion must emit the locked `fusion.gate [D]` schema key."""
    model = _build_test_vetgigagraph(fusion="gated")
    graph, embed, pos = _make_synthetic_graph()
    _, attn = model(graph, embed, pos)
    assert "fusion.gate" in attn, f"gated fusion should expose fusion.gate; got {list(attn)}"
    g = attn["fusion.gate"]
    # Per design §4.3 the gate is element-wise over the proj_dim vector.
    assert g.shape == (PROJ_DIM,), f"fusion.gate shape {tuple(g.shape)} != ({PROJ_DIM},)"
    # Sigmoid-bounded — every element must lie in (0, 1).
    assert (g >= 0).all() and (g <= 1).all()
    assert g.device.type == "cpu"


def test_cross_attention_fusion_emits_fusion_cross_attention_key() -> None:
    """`cross_attention` fusion must emit `fusion.cross_attention [1, 1]`."""
    model = _build_test_vetgigagraph(fusion="cross_attention")
    graph, embed, pos = _make_synthetic_graph()
    _, attn = model(graph, embed, pos)
    assert "fusion.cross_attention" in attn, (
        f"cross_attention fusion should expose fusion.cross_attention; got {list(attn)}"
    )
    score = attn["fusion.cross_attention"]
    assert score.shape == (1, 1), f"fusion.cross_attention shape {tuple(score.shape)} != (1, 1)"
    # Softmax-attention output → bounded in [0, 1].
    assert (score >= 0).all() and (score <= 1).all()
    assert score.device.type == "cpu"


# --------------------------------------------------------------------------- #
# Test 3 — Logits are raw (no softmax applied)
# --------------------------------------------------------------------------- #


def test_no_softmax_in_logits() -> None:
    """Logits must contain at least one negative value across reasonable inputs."""
    torch.manual_seed(0)
    model = _build_test_vetgigagraph()
    seen_negative = False
    for _ in range(5):
        graph, embed, pos = _make_synthetic_graph()
        logits, _ = model(graph, embed, pos)
        if (logits < 0).any():
            seen_negative = True
            break
    assert seen_negative, "logits never went negative — softmax may have been applied"


# --------------------------------------------------------------------------- #
# Test 4 — Tile encoder grad guard
# --------------------------------------------------------------------------- #


def test_tile_encoder_grad_guard() -> None:
    """Backprop through downstream loss must not touch the frozen tile encoder."""
    torch.manual_seed(0)
    tile_encoder = GigaPathTileEncoder(
        model=_TinyTileBackbone(),
        embedding_dim=EMBED_DIM,
        pretrained=False,
    )
    model = _build_test_vetgigagraph()

    tiles = torch.randn(N_NODES, 3, 64, 64)
    embed = tile_encoder(tiles)
    pos = torch.randn(N_NODES, 2) * 100
    edge_index = torch.tensor(
        [[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]], dtype=torch.long
    )
    graph = Data(
        x=embed,
        pos=pos,
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.shape[1], 1),
    )

    logits, _ = model(graph, embed, pos)
    loss = logits.pow(2).sum()
    loss.backward()
    leaks = [n for n, p in tile_encoder.named_parameters() if p.grad is not None]
    assert leaks == [], f"gradient leaked into frozen tile encoder: {leaks}"


# --------------------------------------------------------------------------- #
# Test 5 — Slide encoder unfrozen by default
# --------------------------------------------------------------------------- #


def test_slide_encoder_unfrozen_by_default() -> None:
    enc = _build_test_slide_encoder(frozen=False)
    assert enc.frozen is False
    trainable_count = sum(
        1 for _, p in enc.backbone.named_parameters() if p.requires_grad
    )
    assert trainable_count > 0, "default slide encoder should have trainable backbone params"


# --------------------------------------------------------------------------- #
# Test 6 — Slide encoder frozen when configured
# --------------------------------------------------------------------------- #


def test_slide_encoder_frozen_when_configured() -> None:
    enc = _build_test_slide_encoder(frozen=True)
    assert enc.frozen is True
    leaks = [n for n, p in enc.backbone.named_parameters() if p.requires_grad]
    assert leaks == [], f"backbone params still trainable after freeze: {leaks}"
    enc.assert_freeze_state()  # must not raise


def test_slide_encoder_freeze_unfreeze_cycle() -> None:
    enc = _build_test_slide_encoder(frozen=False)
    enc.freeze()
    assert all(not p.requires_grad for p in enc.backbone.parameters())
    enc.unfreeze()
    assert all(p.requires_grad for p in enc.backbone.parameters())


# --------------------------------------------------------------------------- #
# Test 7 — GNN backbone factory
# --------------------------------------------------------------------------- #


_BACKBONE_TYPE_MAP = {
    "gat": GATConv,
    "gcn": GCNConv,
    "graphsage": SAGEConv,
    "gin": GINConv,
}


@pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
def test_gnn_backbone_factory(backbone: str) -> None:
    """Factory must return a backbone whose layers match the requested type."""
    cfg = GnnBackboneConfig(
        backbone=backbone,
        in_dim=EMBED_DIM,
        hidden_dim=PROJ_DIM,
        output_dim=PROJ_DIM,
        num_layers=3,
        heads=4,
        dropout=0.1,
    )
    gnn = build_gnn_backbone(cfg)
    expected_cls = _BACKBONE_TYPE_MAP[backbone]
    for layer in gnn.layers:
        assert isinstance(layer, expected_cls), (
            f"backbone={backbone}: expected {expected_cls.__name__}, got {type(layer).__name__}"
        )


# --------------------------------------------------------------------------- #
# Test 8 — Fusion factory: all 6 strategies forward correctly
# --------------------------------------------------------------------------- #


_FUSION_CLASS_MAP = {
    "gnn_only": GnnOnlyFusion,
    "slide_only": SlideOnlyFusion,
    "concat": ConcatFusion,
    "learnable_weighted": LearnableWeightedFusion,
    "cross_attention": CrossAttentionFusion,
    "gated": GatedFusion,
}


@pytest.mark.parametrize("strategy", SUPPORTED_FUSIONS)
def test_fusion_factory_all_strategies(strategy: str) -> None:
    fm = build_fusion_module(FusionConfig(strategy=strategy, proj_dim=PROJ_DIM))
    assert isinstance(fm, _FUSION_CLASS_MAP[strategy])

    h_g = torch.randn(PROJ_DIM)
    h_s = torch.randn(PROJ_DIM)
    out, _ = fm(
        h_g if fm.uses_gnn else None,
        h_s if fm.uses_slide else None,
    )
    assert out.shape == (PROJ_DIM,)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Test 9 — Learnable-weighted alpha bounded
# --------------------------------------------------------------------------- #


def test_fusion_alpha_in_unit_interval() -> None:
    """sigmoid(alpha) must lie strictly in (0, 1) for any practical training logit.

    The design contract reads ``sigmoid(α) ∈ (0, 1)``. The interval is
    open in math but saturates at the float32 boundaries for very large
    |logit|: ``sigmoid(±50) → 0.0 / 1.0`` exactly. Real training keeps
    the logit well inside ±10 (gradients vanish past that), so the
    in-practice bound is what matters here.
    """
    fm = build_fusion_module(FusionConfig(strategy="learnable_weighted", proj_dim=PROJ_DIM))
    with torch.no_grad():
        for v in (-10.0, -1.0, 0.0, 1.0, 10.0):
            fm.alpha_logit.fill_(v)
            _, alpha = fm(torch.randn(PROJ_DIM), torch.randn(PROJ_DIM))
            assert alpha is not None
            assert 0.0 < float(alpha) < 1.0, (
                f"alpha out of (0, 1) for logit={v}: got {alpha.item()}"
            )


# --------------------------------------------------------------------------- #
# Test 15 — Param count budget
# --------------------------------------------------------------------------- #


def test_param_count_within_budget() -> None:
    """Total trainable params (excluding frozen tile encoder) must be < 100M."""
    model = _build_test_vetgigagraph()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_train < 100_000_000, f"trainable params {n_train:,} exceed 100M budget"


# --------------------------------------------------------------------------- #
# Bonus — readout sanity (used by integration test 1+2)
# --------------------------------------------------------------------------- #


def test_global_attention_readout_pools_correctly() -> None:
    pool = GlobalAttentionReadout(in_dim=PROJ_DIM)
    h = torch.randn(N_NODES, PROJ_DIM)
    out, w = pool(h)
    assert out.shape == (PROJ_DIM,)
    assert w.shape == (N_NODES,)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)


# --------------------------------------------------------------------------- #
# Phase 6 — Baselines (design §5.D rows 10–13, plus the 5th CLAM-MB case)
#
# Spec from `do.md` Phase 6: each baseline's forward() produces [7]
# logits on a 100-tile bag fixture without NaN. Phase 6 → Phase 7 gate
# is "all 5 baselines pass test_baseline_*_forward".
# --------------------------------------------------------------------------- #


from src.models.baselines import (  # noqa: E402 — grouped after Phase 5 imports
    ABMIL,
    BASELINE_REGISTRY,
    CLAM_MB,
    CLAM_SB,
    DSMIL,
    TransMIL,
)

#: Bag fixture size — design §5.D specifies "100-tile bag".
BAG_SIZE = 100


def _make_bag(seed: int = 0) -> torch.Tensor:
    """Synthesize a 100-tile bag of 1536-d embeddings (matches GigaPath shape)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BAG_SIZE, 1536, generator=g)


def test_baseline_abmil_forward() -> None:
    bag = _make_bag(seed=10)
    model = ABMIL(num_classes=NUM_CLASSES)
    out = model(bag)
    assert out.shape == (NUM_CLASSES,)
    assert torch.isfinite(out).all()


def test_baseline_dsmil_forward() -> None:
    bag = _make_bag(seed=11)
    model = DSMIL(num_classes=NUM_CLASSES)
    out = model(bag)
    assert out.shape == (NUM_CLASSES,)
    assert torch.isfinite(out).all()


def test_baseline_transmil_forward() -> None:
    bag = _make_bag(seed=12)
    model = TransMIL(num_classes=NUM_CLASSES)
    out = model(bag)
    assert out.shape == (NUM_CLASSES,)
    assert torch.isfinite(out).all()


def test_baseline_clam_sb_forward() -> None:
    bag = _make_bag(seed=13)
    model = CLAM_SB(num_classes=NUM_CLASSES)
    out = model(bag)
    assert out.shape == (NUM_CLASSES,)
    assert torch.isfinite(out).all()


def test_baseline_clam_mb_forward() -> None:
    bag = _make_bag(seed=14)
    model = CLAM_MB(num_classes=NUM_CLASSES)
    out = model(bag)
    assert out.shape == (NUM_CLASSES,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", sorted(BASELINE_REGISTRY.keys()))
def test_baseline_registry_forward(name: str) -> None:
    """Registry-driven smoke test: every entry must produce ``[NUM_CLASSES]`` logits."""
    bag = _make_bag(seed=hash(name) & 0xFFFF)
    model = BASELINE_REGISTRY[name](num_classes=NUM_CLASSES)
    out = model(bag)
    assert out.shape == (NUM_CLASSES,)
    assert torch.isfinite(out).all()


def test_baseline_registry_locked_names() -> None:
    """Registry keys are part of the public CLI contract — guard against typos."""
    assert sorted(BASELINE_REGISTRY) == [
        "abmil", "acmil", "clam_mb", "clam_mb_inst", "clam_sb", "clam_sb_inst",
        "dsmil", "transmil", "wikg",
    ]


def test_baseline_no_softmax_in_logits() -> None:
    """Baselines must return raw logits, not softmaxed probabilities.

    Softmaxed outputs would (a) sum to exactly 1 and (b) all lie in
    [0, 1]. We check both: at least one bag must violate one of those
    properties for each baseline (averaged over a few seeds to avoid a
    false positive from a small randomly-initialised baseline that
    happens to land inside the simplex by chance).
    """
    for name, cls in BASELINE_REGISTRY.items():
        torch.manual_seed(0)
        model = cls(num_classes=NUM_CLASSES)
        looked_softmaxed_every_time = True
        for seed in range(5):
            bag = _make_bag(seed=20 + seed)
            out = model(bag)
            sums_to_one = bool(torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-3))
            in_unit = bool(((out >= 0) & (out <= 1)).all())
            if not (sums_to_one and in_unit):
                looked_softmaxed_every_time = False
                break
        assert not looked_softmaxed_every_time, (
            f"{name}: outputs look softmaxed across 5 seeds — design contract requires raw logits"
        )


# --------------------------------------------------------------------------- #
# scripts/04_train.py::_build_model — D-7 jr refactor (micro-PDCA
# `_build_model-refactor`). See docs/02-design/features/_build_model-refactor.design.md
# --------------------------------------------------------------------------- #


def _load_build_model():
    """Load ``_build_model`` from ``scripts/04_train.py`` (filename starts with
    a digit, so can't ``import scripts.04_train`` directly)."""
    import importlib.util
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_build_model_under_test", repo / "scripts" / "04_train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_build_model_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_vetgiga_cfg(*, fusion_strategy: str = "gnn_only", **gnn_overrides):
    """Build a minimal top-level config dict matching ``configs/default.yaml`` shape."""
    return {
        "project": {"num_classes": NUM_CLASSES},
        "feature_extraction": {"embedding_dim": EMBED_DIM},
        "model": {
            "gnn": {
                "backbone": "gat",
                "layers": 2,
                "hidden_dim": 32,
                "output_dim": 32,
                "heads": 2,
                "dropout": 0.1,
                "gradient_checkpointing": False,
                **gnn_overrides,
            },
            "fusion": {
                "strategy": fusion_strategy,
                "proj_dim": 32,
                "init_alpha": 0.5,
            },
            "classifier": {"hidden_dim": 16, "dropout": 0.1},
            "slide_encoder": {"frozen": True, "proj_dim": 32, "enabled": True},
        },
    }


def test_build_model_vetgigagraph_gnn_only_reads_config() -> None:
    """Happy path: config-driven GNN hyperparameters land on the built model.

    Resolves drift D-7 (vetgigagraph analysis v0.4) — every locked GNN
    hyperparameter must come from ``cfg.model.gnn``, not a hardcoded literal.
    """
    mod = _load_build_model()
    cfg = _make_vetgiga_cfg(
        fusion_strategy="gnn_only",
        hidden_dim=64,
        output_dim=32,
        heads=2,
        layers=2,
        gradient_checkpointing=False,
    )
    model = mod._build_model(
        "vetgigagraph",
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        cfg=cfg,
        slide_backbone_loader=None,
    )
    assert isinstance(model, VetGigaGraph)
    assert isinstance(model.gnn, GnnBackbone)
    assert model.gnn.config.hidden_dim == 64
    assert model.gnn.config.heads == 2
    assert model.gnn.config.num_layers == 2
    assert model.gnn.config.gradient_checkpointing is False
    assert model.slide_encoder is None
    assert isinstance(model.fusion, GnnOnlyFusion)
    assert isinstance(model.classifier, MlpClassifier)


def test_build_model_vetgigagraph_raises_without_loader() -> None:
    """Pre-check: a slide-using fusion strategy without a loader must fail loudly.

    Without this guard, ``VetGigaGraph.from_config`` would call the default
    GigaPath slide-backbone loader (which raises ``NotImplementedError``)
    and the error message would not surface the actual missing piece.
    """
    from src.utils.errors import ConfigError

    mod = _load_build_model()
    cfg = _make_vetgiga_cfg(fusion_strategy="learnable_weighted")
    with pytest.raises(ConfigError) as excinfo:
        mod._build_model(
            "vetgigagraph",
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            cfg=cfg,
            slide_backbone_loader=None,
        )
    msg = str(excinfo.value)
    assert "slide_backbone_loader" in msg
    assert "slide-encoder-injection" in msg
    assert "learnable_weighted" in msg


def test_build_model_vetgigagraph_raises_without_cfg() -> None:
    """Pre-check: cfg is required for the vetgigagraph branch.

    Baselines don't need cfg (they take ``embed_dim`` directly), so this
    guard only fires for the proposed model.
    """
    from src.utils.errors import ConfigError

    mod = _load_build_model()
    with pytest.raises(ConfigError) as excinfo:
        mod._build_model(
            "vetgigagraph",
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            cfg=None,
        )
    assert "merged config" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# slide-encoder-tests µPDCA #4 — backfill for slide-encoder-injection µPDCA #3.
# Covers `_build_slide_backbone_loader` + `_UniformAttentionAdapter` + D-13
# invariant + HEAVY-gated production_slide_backbone_loader real-download smoke.
# See docs/01-plan/features/slide-encoder-tests.plan.md §4 Milestones.
# --------------------------------------------------------------------------- #


def _make_vetgiga_cfg_multimodal(*, fusion_strategy: str = "learnable_weighted",
                                  proj_dim: int = 32, output_dim: int = 32):
    """Multi-modal variant of ``_make_vetgiga_cfg`` — sets a slide-using fusion
    strategy + aligned ``proj_dim/output_dim`` by default (operator can mismatch
    them to exercise the D-13 invariant)."""
    cfg = _make_vetgiga_cfg(fusion_strategy=fusion_strategy)
    cfg["model"]["slide_encoder"]["proj_dim"] = proj_dim
    cfg["model"]["gnn"]["output_dim"] = output_dim
    return cfg


# -- M1: _build_slide_backbone_loader factory tests ------------------------- #


def test_build_slide_backbone_loader_gnn_only_returns_none() -> None:
    """Phase-12 fallback: gnn_only fusion needs no slide encoder → None loader.

    Lazy avoidance of the ~345 MB HF download when the operator runs a
    baseline model or the gnn_only fusion fallback.
    """
    mod = _load_build_model()
    cfg = {"model": {"fusion": {"strategy": "gnn_only"}}}
    assert mod._build_slide_backbone_loader(cfg) is None


@pytest.mark.parametrize("strategy", [
    "slide_only", "concat", "learnable_weighted", "cross_attention", "gated",
])
def test_build_slide_backbone_loader_other_returns_callable(strategy: str) -> None:
    """All non-gnn_only fusion strategies must request the production loader."""
    mod = _load_build_model()
    cfg = {"model": {"fusion": {"strategy": strategy}}}
    loader = mod._build_slide_backbone_loader(cfg)
    assert loader is not None
    assert callable(loader)
    # The returned callable is the production loader symbol itself — NOT invoked
    from src.models.gigapath_slide import production_slide_backbone_loader
    assert loader is production_slide_backbone_loader


# -- M3: _UniformAttentionAdapter shape contract --------------------------- #


def test_uniform_attention_adapter_shape_contract() -> None:
    """Synthetic backbone (no ``patch_embed``) triggers D-1 uniform fallback —
    must satisfy the design §4.3 (cls_output, cls_attention) contract regardless
    of 2D vs 3D input."""
    from src.models.gigapath_slide import _UniformAttentionAdapter

    class _Tiny(nn.Module):
        """Stand-in backbone — returns a 1-element list per upstream convention."""
        def forward(self, x, coord):
            # x: [B, N, D] — emit [B, D]
            return [x.mean(dim=1)]

    adapter = _UniformAttentionAdapter(_Tiny())

    # Path 1: 2D input [N, D] (legacy GigaPathSlideEncoder wrapper contract)
    N, D = 100, 64
    tile_2d = torch.randn(N, D)
    coord_2d = torch.randint(0, 100, (N, 2)).float()
    cls_2d, attn_2d = adapter(tile_2d, coord_2d)
    assert cls_2d.shape == (D,), f"2D path: cls.shape != ({D},); got {tuple(cls_2d.shape)}"
    assert attn_2d.shape == (N,), f"2D path: attn.shape != ({N},); got {tuple(attn_2d.shape)}"
    assert torch.isclose(attn_2d.sum(), torch.tensor(1.0), atol=1e-5), \
        f"Fallback attention must sum to 1.0; got {attn_2d.sum().item()}"
    # D-1 uniform fallback (no patch_embed → cosine path fails)
    assert torch.allclose(attn_2d, torch.full((N,), 1.0 / N))

    # Path 2: 3D input [B=1, N, D] (upstream LongNetViT contract)
    tile_3d = tile_2d.unsqueeze(0)
    coord_3d = coord_2d.unsqueeze(0)
    cls_3d, attn_3d = adapter(tile_3d, coord_3d)
    # 3D path: cls keeps batch dim [B, D]
    assert cls_3d.shape == (1, D), f"3D path: cls.shape != (1, {D}); got {tuple(cls_3d.shape)}"
    assert attn_3d.shape == (N,), f"3D path: attn.shape != ({N},); got {tuple(attn_3d.shape)}"


def test_attention_adapter_fallback_warns_when_patch_embed_missing(caplog) -> None:
    """When the backbone has no ``patch_embed`` attribute (synthetic test
    stand-ins), the D-2 cosine-similarity path must (a) fall back to D-1
    uniform and (b) log a WARN with the failure class name."""
    from src.models.gigapath_slide import _AttentionExtractingAdapter

    class _NoPatchEmbed(nn.Module):
        def forward(self, x, coord):
            return [x.mean(dim=1)]

    adapter = _AttentionExtractingAdapter(_NoPatchEmbed())
    tile = torch.randn(13, 8)
    coord = torch.zeros(13, 2)

    with caplog.at_level("WARNING", logger="src.models.gigapath_slide"):
        _, attn = adapter(tile, coord)

    # Uniform fallback engaged
    assert torch.allclose(attn, torch.full((13,), 1.0 / 13))
    # WARN log emitted
    warns = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("cosine-similarity attention failed" in r.getMessage() for r in warns), (
        f"expected WARN about cosine-similarity failure; got: "
        f"{[r.getMessage() for r in warns]}"
    )
    # Specifically mentions the failure class (AttributeError when patch_embed missing)
    assert any("AttributeError" in r.getMessage() for r in warns)


def test_attention_adapter_cosine_path_non_uniform_with_patch_embed() -> None:
    """When the backbone has a (trivially deterministic) ``patch_embed``, the
    D-2 cosine path must engage — produce a non-uniform [N] attention that
    sums to 1.0. Uses a tiny synthetic backbone; no HF download needed."""
    from src.models.gigapath_slide import _AttentionExtractingAdapter

    D_IN, D_OUT = 16, 8

    class _BackboneWithPatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            # Deterministic linear projection 16 → 8
            self.patch_embed = nn.Linear(D_IN, D_OUT, bias=False)
            # Different deterministic weight than patch_embed — so cls != mean(tile_proj)
            self.cls_head = nn.Linear(D_IN, D_OUT, bias=False)

        def forward(self, x, coord):
            # Slide-level CLS is a learned linear combination of tile mean — varies per slide
            slide_mean = x.mean(dim=1)         # [B, D_IN]
            cls = self.cls_head(slide_mean)    # [B, D_OUT]
            return [cls]

    torch.manual_seed(0)
    backbone = _BackboneWithPatchEmbed()
    # Lower tau makes cosine differences more visible at this synthetic scale.
    # Real GigaPath outputs have richer signal so production code keeps tau=1.0.
    adapter = _AttentionExtractingAdapter(backbone, tau=0.1)

    N = 25
    tile = torch.randn(N, D_IN)
    coord = torch.zeros(N, 2)
    cls, attn = adapter(tile, coord)

    assert cls.shape == (D_OUT,)
    assert attn.shape == (N,)
    assert torch.isclose(attn.sum(), torch.tensor(1.0), atol=1e-5)
    # Should be non-uniform (cosine sim varies per tile)
    assert attn.std() > 1e-3, (
        f"D-2 cosine path must produce non-uniform attention; got std={attn.std().item():.6f}"
    )
    # Entropy invariant — should be < log(N) - 0.05 (non-uniform)
    eps = 1e-12
    entropy = -(attn * (attn + eps).log()).sum().item()
    uniform_entropy = float(torch.tensor(N, dtype=torch.float32).log().item())
    assert entropy < uniform_entropy - 0.05, (
        f"expected entropy < log(N) - 0.05 = {uniform_entropy - 0.05:.4f}; "
        f"got {entropy:.4f}"
    )


@pytest.mark.skipif(
    not (os.getenv("VETGIGAGRAPH_RUN_HEAVY_TESTS") == "1" and os.getenv("HF_TOKEN")),
    reason="HEAVY: needs VETGIGAGRAPH_RUN_HEAVY_TESTS=1 + HF_TOKEN for HF download",
)
def test_attention_adapter_real_content_non_uniform_heavy() -> None:
    """End-to-end: load real GigaPath slide encoder, run D-2 cosine attention
    on synthetic input, verify the resulting cls_attention has entropy
    < log(N) - 0.05 (i.e., meaningfully non-uniform)."""
    from src.models.gigapath_slide import (
        _AttentionExtractingAdapter,
        production_slide_backbone_loader,
        EMBED_DIM,
    )

    backbone = production_slide_backbone_loader()
    # production_slide_backbone_loader returns an _AttentionExtractingAdapter
    # wrapping the loaded LongNetViT — re-wrap to get the .backbone
    inner = backbone.backbone if hasattr(backbone, "backbone") else backbone
    adapter = _AttentionExtractingAdapter(inner)

    torch.manual_seed(42)
    N = 256
    tile = torch.randn(N, EMBED_DIM)
    coord = torch.randint(0, 1024, (N, 2)).float()
    with torch.no_grad():
        _, attn = adapter(tile, coord)

    assert attn.shape == (N,)
    assert torch.isclose(attn.sum(), torch.tensor(1.0), atol=1e-4)
    eps = 1e-12
    entropy = -(attn * (attn + eps).log()).sum().item()
    uniform_entropy = float(torch.tensor(N, dtype=torch.float32).log().item())
    assert entropy < uniform_entropy - 0.05, (
        f"real D-2 attention should be non-uniform; got entropy={entropy:.4f}, "
        f"uniform={uniform_entropy:.4f}, threshold={uniform_entropy - 0.05:.4f}"
    )


# -- M4: D-13 invariant — proj_dim ↔ output_dim alignment ------------------ #


def test_build_model_vetgigagraph_d13_proj_output_dim_mismatch() -> None:
    """D-13 (slide-encoder-tests µPDCA): slide-using fusion with mismatched
    ``slide_encoder.proj_dim`` and ``gnn.output_dim`` must raise ConfigError
    at construction time, not deep inside fusion._require_pair."""
    from src.utils.errors import ConfigError

    mod = _load_build_model()
    cfg = _make_vetgiga_cfg_multimodal(
        fusion_strategy="learnable_weighted",
        proj_dim=256,    # mismatched
        output_dim=128,
    )
    # Provide a dummy loader so the pre-existing "no loader" check doesn't fire first
    with pytest.raises(ConfigError) as excinfo:
        mod._build_model(
            "vetgigagraph",
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            cfg=cfg,
            slide_backbone_loader=lambda: None,
        )
    msg = str(excinfo.value)
    assert "proj_dim" in msg
    assert "output_dim" in msg
    assert "256" in msg and "128" in msg
    assert "slide-encoder-tests" in msg


def test_build_model_vetgigagraph_d13_aligned_dims_passes_d13_check() -> None:
    """When proj_dim == output_dim, the D-13 invariant passes (downstream
    failure due to dummy loader is expected and tolerated — we only assert
    the D-13 branch did NOT fire)."""
    from src.utils.errors import ConfigError

    mod = _load_build_model()
    cfg = _make_vetgiga_cfg_multimodal(
        fusion_strategy="learnable_weighted",
        proj_dim=32,
        output_dim=32,
    )
    # Loader returns None → adapter init fails — but the failure should NOT be
    # a D-13 ConfigError. Verify by trying and inspecting the exception type/msg.
    try:
        mod._build_model(
            "vetgigagraph",
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            cfg=cfg,
            slide_backbone_loader=lambda: None,
        )
    except ConfigError as e:
        # If a ConfigError is raised, it must NOT be the D-13 one
        assert "proj_dim" not in str(e) or "output_dim" not in str(e), \
            f"D-13 invariant fired incorrectly with aligned dims: {e}"
    except Exception:
        # Any non-ConfigError exception means D-13 wasn't the failure mode — OK
        pass


# -- M2: HEAVY-gated production loader smoke -------------------------------- #


@pytest.mark.skipif(
    os.getenv("VETGIGAGRAPH_RUN_HEAVY_TESTS") != "1"
    or not os.getenv("HF_TOKEN"),
    reason="HEAVY test: requires VETGIGAGRAPH_RUN_HEAVY_TESTS=1 + HF_TOKEN in env",
)
def test_production_slide_backbone_loader_real_download() -> None:
    """HEAVY: actual ``prov-gigapath/prov-gigapath/slide_encoder.pth`` download
    + state_dict load. Mirrors the verification operator did during
    µPDCA #3 manual smoke (missing=0, unexpected=0). Confirms upstream HF
    artefact + vendored arch + xformers branch all still align.

    Skipped by default to avoid 345 MB download in routine CI.
    """
    from src.models.gigapath_slide import production_slide_backbone_loader

    model = production_slide_backbone_loader()
    assert isinstance(model, nn.Module)

    # 86M total / 85M trainable per µPDCA #3 verification
    n_params = sum(p.numel() for p in model.parameters())
    assert 80_000_000 < n_params < 90_000_000, \
        f"slide encoder params {n_params/1e6:.2f}M outside expected ~85M range"

    # Forward smoke at small N (won't OOM on CPU)
    N = 256
    tile_emb = torch.randn(1, N, EMBED_DIM)
    coord = torch.randint(0, 100, (1, N, 2)).float()
    model.eval()
    with torch.no_grad():
        cls_out, cls_attn = model(tile_emb, coord)
    # Adapter strips B=1 only for 2D input; 3D input keeps batch dim
    assert cls_out.shape == (1, 768), f"cls_output shape {tuple(cls_out.shape)} ≠ (1, 768)"
    assert cls_attn.shape == (N,), f"cls_attention shape {tuple(cls_attn.shape)} ≠ ({N},)"
    assert torch.isclose(cls_attn.sum(), torch.tensor(1.0), atol=1e-5)


