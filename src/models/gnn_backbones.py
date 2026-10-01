"""GNN backbone factory and stack (Phase 5.1).

Implements a uniform ``GnnBackbone`` that wraps a stack of one of four
message-passing variants (GAT / GCN / GraphSAGE / GIN) and exposes the
per-layer edge-attention dict that the locked
``attention_weights`` schema in
``docs/02-design/features/vetgigagraph.design.md`` §4.3 requires.

Real attention only flows through GAT (the layer ``return_attention_weights``
path returns the actual ``[E, H]`` tensor). For non-attention backbones
(GCN / GraphSAGE / GIN) we fabricate a deterministic ``[E, 1]`` tensor of
ones per layer so callers can assume the schema's ``H`` axis exists
regardless of backbone choice. This keeps downstream visualization,
gap-detector, and the test suite uniform.

References:
    Design: docs/02-design/03-architecture.md §5 (Module D)
    Design: docs/02-design/features/vetgigagraph.design.md §4.3 (attention schema)
    Tests:  docs/02-design/03-architecture.md §5.D rows 7 (factory) + attention rows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

logger = logging.getLogger(__name__)

#: Locked backbone names — match the literal in
#: ``configs/default.yaml model.gnn.backbone`` and
#: ``vetgigagraph.design.md`` §4.1 ablation rows.
SUPPORTED_BACKBONES = ("gat", "gcn", "graphsage", "gin")


@dataclass(frozen=True)
class GnnBackboneConfig:
    """Hyperparameters for the GNN stack — mirror of ``configs/default.yaml model.gnn``."""

    backbone: str = "gat"
    in_dim: int = 1536
    hidden_dim: int = 512
    output_dim: int = 256
    num_layers: int = 3
    heads: int = 8
    dropout: float = 0.25
    # Wrap each GAT layer forward in torch.utils.checkpoint so the per-edge
    # message tensor [E, heads, hidden] (up to ~20 GiB fp16 on 94k-node WSIs)
    # is recomputed during backward instead of retained. Trades ~30-40% extra
    # compute for ~3× peak-memory headroom. See v0.4 analysis D-7/D-8.
    gradient_checkpointing: bool = False
    # µPDCA #9 D-23 fix: when True, the GAT layer consumes node_type / edge_type
    # information via PyG GATConv's ``edge_dim`` channel (edge_type one-hot fused
    # with existing edge_attr) and a node-type embedding concatenated to ``x``
    # before the first message-passing step. Only activates when the input
    # graph carries the optional ``node_type`` + ``edge_type`` fields (the
    # heterogeneous variant), and only when backbone == "gat". Other backbones
    # ignore the flag (heterogeneity has no canonical GraphSAGE/GCN analogue).
    use_heterogeneous: bool = False
    n_node_types: int = 3       # tumor / stroma / inflammation
    n_edge_types: int = 2       # intra / inter
    # GIN diagnostics (1st-revision Reviewer 2 point 5). Defaults reproduce the
    # published GIN (sum aggregation, no normalisation inside the MLP).
    gin_aggr: str = "add"       # add | mean
    gin_norm: bool = False      # LayerNorm after each linear in the GIN MLP

    def __post_init__(self) -> None:
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone must be one of {SUPPORTED_BACKBONES}; got {self.backbone!r}"
            )
        if self.num_layers < 2 or self.num_layers > 4:
            raise ValueError(
                f"num_layers must be in [2, 4] per design §5; got {self.num_layers}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout}")
        if self.use_heterogeneous and self.backbone != "gat":
            # Heterogeneous edge weighting is only wired for GAT (uses edge_dim).
            # Silent-fallback would be confusing; require explicit choice.
            raise ValueError(
                f"use_heterogeneous=True requires backbone='gat'; got {self.backbone!r}"
            )
        if self.n_node_types < 1 or self.n_edge_types < 1:
            raise ValueError("n_node_types and n_edge_types must be positive")


# --------------------------------------------------------------------------- #
# Stack
# --------------------------------------------------------------------------- #


class GnnBackbone(nn.Module):
    """Stack of message-passing layers exposing attention per layer.

    The output of :meth:`forward` is a 2-tuple:

    * ``h`` — node features ``[N, output_dim]`` (post-relu, post-dropout
      omitted on the last layer to match common convention).
    * ``attentions`` — dict ``{"layer_<i>.edge_attention": Tensor[E_i, H]}``
      where ``E_i`` is the number of edges seen at layer ``i`` (GAT can
      add self-loops, so ``E_i`` may be slightly larger than the input
      ``E``). ``H`` is ``heads`` for GAT, ``1`` for the others.
    """

    def __init__(self, config: GnnBackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(self._build_layers(config))
        # µPDCA #9 D-23: optional node-type embedding concatenated to x before
        # the first GAT layer. Only active when use_heterogeneous and a graph
        # carrying node_type is presented.
        if config.use_heterogeneous and config.backbone == "gat":
            self.node_type_embedding = nn.Embedding(
                num_embeddings=config.n_node_types,
                embedding_dim=8,
            )
        else:
            self.node_type_embedding = None

    # --- forward --------------------------------------------------------- #

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        node_type: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attentions: dict[str, torch.Tensor] = {}
        h = x
        # µPDCA #9 D-23: heterogeneous edge_attr augmentation + node_type
        # embedding. Skipped when use_heterogeneous=False or the graph lacks
        # the optional fields (graceful — gnn_only / dual_edge work unchanged).
        edge_attr_aug = edge_attr
        if self.config.use_heterogeneous and self.config.backbone == "gat":
            # Node-type concat: always 8-d (zero-pad when node_type absent so
            # the first GAT layer's [in_dim+8] expectation is satisfied).
            if node_type is not None and self.node_type_embedding is not None:
                nt_emb = self.node_type_embedding(node_type.long())  # [N, 8]
            else:
                nt_emb = torch.zeros(h.shape[0], 8, device=h.device, dtype=h.dtype)
            h = torch.cat([h, nt_emb], dim=-1)

            # Edge-type one-hot fused into edge_attr (zero-pad when edge_type
            # absent so the configured edge_dim is satisfied).
            n_edges = edge_index.shape[1]
            if edge_type is not None:
                et_oh = F.one_hot(
                    edge_type.long(), num_classes=self.config.n_edge_types
                ).to(dtype=h.dtype)
            else:
                et_oh = torch.zeros(
                    n_edges, self.config.n_edge_types,
                    device=h.device, dtype=h.dtype,
                )
            if edge_attr is None:
                ea_base = torch.zeros(n_edges, 1, device=h.device, dtype=h.dtype)
            else:
                ea_base = edge_attr
                if ea_base.ndim == 1:
                    ea_base = ea_base.unsqueeze(-1)
                ea_base = ea_base.to(dtype=h.dtype)
            edge_attr_aug = torch.cat([ea_base, et_oh], dim=-1)

        last_idx = len(self.layers) - 1
        use_ckpt = (
            self.config.gradient_checkpointing
            and self.training
            and h.requires_grad
        )
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GATConv):
                # Pass edge_attr only when layer was built with edge_dim>0,
                # i.e. use_heterogeneous path. Vanilla GAT ignores it.
                ea_for_layer = edge_attr_aug if self._layer_supports_edge_attr(layer) else None
                if use_ckpt:
                    h, alpha = _ckpt(
                        _gat_layer_forward, layer, h, edge_index, ea_for_layer,
                        use_reentrant=False,
                    )
                else:
                    h, (_ei_used, alpha) = layer(
                        h, edge_index, edge_attr=ea_for_layer, return_attention_weights=True
                    )
                attentions[f"layer_{i + 1}.edge_attention"] = alpha
            else:
                h = layer(h, edge_index)
                attentions[f"layer_{i + 1}.edge_attention"] = torch.ones(
                    edge_index.shape[1], 1, device=h.device, dtype=torch.float32
                )
            if i < last_idx:
                h = F.relu(h)
                h = F.dropout(h, p=self.config.dropout, training=self.training)
        return h, attentions

    @staticmethod
    def _layer_supports_edge_attr(layer: nn.Module) -> bool:
        """Return True if a GATConv layer was built with non-None edge_dim."""
        if not isinstance(layer, GATConv):
            return False
        # PyG GATConv stores edge_dim as an attribute when set.
        edge_dim = getattr(layer, "edge_dim", None)
        return edge_dim is not None and edge_dim > 0

    # --- builders -------------------------------------------------------- #

    @staticmethod
    def _build_layers(config: GnnBackboneConfig) -> list[nn.Module]:
        """Return ``num_layers`` message-passing modules with the right widths.

        Layer widths follow the design diagram (`03-architecture.md` §5):

        * Layer 1 — ``in_dim → hidden_dim``
        * Layers 2..N-1 — ``hidden_dim → hidden_dim``
        * Layer N — ``hidden_dim → output_dim``

        For GAT we use ``concat=False`` so the per-layer output width is
        exactly ``hidden_dim`` / ``output_dim`` regardless of the head
        count (each head's contribution is averaged). This keeps the
        rest of the pipeline (fusion, classifier) head-count-agnostic.

        µPDCA #9 D-23: when ``use_heterogeneous=True``, the first layer's
        input width is bumped by ``+8`` (node_type embedding concat), and
        every GAT layer is built with ``edge_dim=n_edge_types + 1`` so it
        can consume the edge_type one-hot fused with the inverse-distance
        edge_attr from the heterogeneous variant builder.
        """
        in_dim_first = config.in_dim
        if config.use_heterogeneous and config.backbone == "gat":
            in_dim_first = config.in_dim + 8  # node_type embedding width
        widths = [in_dim_first] + [config.hidden_dim] * (config.num_layers - 1) + [config.output_dim]
        # edge_dim: existing edge_attr is [E, 1] (inverse distance); plus
        # one-hot of edge_type adds n_edge_types channels.
        edge_dim = None
        if config.use_heterogeneous and config.backbone == "gat":
            edge_dim = config.n_edge_types + 1
        out: list[nn.Module] = []
        for i in range(config.num_layers):
            in_dim, out_dim = widths[i], widths[i + 1]
            out.append(
                _make_layer(config.backbone, in_dim, out_dim,
                            heads=config.heads, edge_dim=edge_dim,
                            gin_aggr=config.gin_aggr, gin_norm=config.gin_norm)
            )
        return out


def _gat_layer_forward(
    layer: GATConv,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint-friendly wrapper that flattens GATConv's nested tuple return."""
    h, (_ei_used, alpha) = layer(
        x, edge_index, edge_attr=edge_attr, return_attention_weights=True
    )
    return h, alpha


def _make_layer(
    backbone: str,
    in_dim: int,
    out_dim: int,
    *,
    heads: int,
    edge_dim: int | None = None,
    gin_aggr: str = "add",
    gin_norm: bool = False,
) -> nn.Module:
    if backbone == "gat":
        # add_self_loops requires edge_dim handling: PyG fills in zero-edge_attr
        # for the self-loop rows automatically when edge_dim is set, but ONLY
        # if fill_value defaults are honored. PyG-2.4+ default fill_value="mean"
        # for edge_attr — safe for our inverse-distance + one-hot mix.
        return GATConv(
            in_dim, out_dim, heads=heads, concat=False,
            add_self_loops=True, edge_dim=edge_dim,
        )
    if backbone == "gcn":
        return GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
    if backbone == "graphsage":
        return SAGEConv(in_dim, out_dim, aggr="mean")
    if backbone == "gin":
        # GIN is normally instantiated with an internal MLP; we follow the
        # canonical 2-layer-MLP recipe from the original GIN paper.
        if gin_norm:
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(),
                nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim),
            )
        else:
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
        return GINConv(mlp, train_eps=True, aggr=gin_aggr)
    raise ValueError(f"Unknown backbone {backbone!r}; expected one of {SUPPORTED_BACKBONES}")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_gnn_backbone(config: Mapping[str, Any] | GnnBackboneConfig) -> GnnBackbone:
    """Build a :class:`GnnBackbone` from the project YAML config.

    Accepts either a :class:`GnnBackboneConfig` directly or any mapping
    that follows the ``configs/default.yaml`` ``model.gnn`` shape (e.g.
    an OmegaConf node). The mapping form must contain the keys listed
    in the :class:`GnnBackboneConfig` dataclass.
    """
    if isinstance(config, GnnBackboneConfig):
        cfg = config
    else:
        # Tolerant unwrap: support both top-level configs (``cfg.model.gnn``)
        # and already-narrowed configs (``cfg.gnn`` or ``gnn``).
        gnn_cfg = _narrow_to_gnn(config)
        emb_dim = _resolve_embedding_dim(config)
        cfg = GnnBackboneConfig(
            backbone=str(gnn_cfg["backbone"]),
            in_dim=int(emb_dim),
            hidden_dim=int(gnn_cfg["hidden_dim"]),
            output_dim=int(gnn_cfg["output_dim"]),
            num_layers=int(gnn_cfg["layers"]),
            heads=int(gnn_cfg["heads"]),
            dropout=float(gnn_cfg["dropout"]),
            gradient_checkpointing=bool(gnn_cfg.get("gradient_checkpointing", False)),
            # µPDCA #9 D-23: optional hetero-GAT activation. Default False so
            # the gnn_only / dual_edge / spatial_knn paths are unaffected.
            use_heterogeneous=bool(gnn_cfg.get("use_heterogeneous", False)),
            n_node_types=int(gnn_cfg.get("n_node_types", 3)),
            n_edge_types=int(gnn_cfg.get("n_edge_types", 2)),
            gin_aggr=str(gnn_cfg.get("gin_aggr", "add")),
            gin_norm=bool(gnn_cfg.get("gin_norm", False)),
        )
    logger.info(
        "Building GNN backbone %s: %d layers, in=%d, hidden=%d, out=%d, heads=%d",
        cfg.backbone, cfg.num_layers, cfg.in_dim, cfg.hidden_dim, cfg.output_dim, cfg.heads,
    )
    return GnnBackbone(cfg)


def _narrow_to_gnn(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "backbone" in config:
        return config  # already a gnn-section view
    if "gnn" in config:
        return config["gnn"]
    if "model" in config and "gnn" in config["model"]:
        return config["model"]["gnn"]
    raise KeyError(
        "build_gnn_backbone: could not locate `gnn` section in the supplied config"
    )


def _resolve_embedding_dim(config: Mapping[str, Any]) -> int:
    """Default to 1536 if the config doesn't reach feature_extraction.embedding_dim."""
    fe = config.get("feature_extraction") if isinstance(config, Mapping) else None
    if fe and "embedding_dim" in fe:
        return int(fe["embedding_dim"])
    return 1536
