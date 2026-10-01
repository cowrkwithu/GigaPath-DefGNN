"""Integrated VetGigaGraph model (Phase 5.5) — combines all four sub-modules.

Pipeline:

    graph (PyG Data with x[N,1536], pos[N,2], edge_index, edge_attr)
        ├── GNN backbone (3-layer GAT/GCN/GraphSAGE/GIN)
        │       ↓ per-node features [N, output_dim]
        │   Global attention pool → h_gnn [output_dim], node_attention [N]
        │
        ├── (optional) Slide encoder (GigaPath LongNet)
        │       ↓ slide-level CLS [embed_dim] + cls_attention [N]
        │   Linear projection → h_slide [proj_dim]
        │
        └── Fusion module (one of 6 strategies)
                ↓ h_fused [proj_dim]
            MLP classifier → logits [num_classes]

The slide-encoder branch is skipped automatically when the fusion's
``uses_slide`` flag is False (``gnn_only``), and the GNN branch is
skipped when ``uses_gnn`` is False (``slide_only``). This keeps each
ablation efficient at training time.

Forward returns ``(logits, attention_weights)`` matching the locked
schema in ``docs/02-design/features/vetgigagraph.design.md`` §4.3:

* ``gnn.layer_<i>.edge_attention`` — one per layer, ``[E_i, H]``
* ``gnn.readout.node_attention`` — ``[N]``, sums to 1
* ``slide_encoder.cls_attention`` — ``[N]``, present iff fusion uses slide
* ``fusion.alpha`` — scalar, present iff strategy == ``learnable_weighted``

All attention tensors are detached and CPU-pinned so the caller can
log them without blocking gradient compute.

References:
    Design: docs/02-design/features/vetgigagraph.design.md §4.3 (attention schema)
    Design: docs/02-design/03-architecture.md §5 (Module D)
    Tests:  docs/02-design/03-architecture.md §5.D rows 1, 2, 3, 4, 15
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.models.classifier import MlpClassifier
from src.models.fusion import _FusionBase, build_fusion_module
from src.models.gigapath_slide import GigaPathSlideEncoder
from src.models.gnn_backbones import GnnBackbone, build_gnn_backbone

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Global attention pooling (graph readout)
# --------------------------------------------------------------------------- #


class GlobalAttentionReadout(nn.Module):
    """Pool ``[N, D]`` node features into ``[D]`` via softmax attention.

    A small gate network (single linear) predicts a scalar per node;
    softmax over nodes gives per-node weights summing to 1; the slide
    representation is the weighted sum.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [N, D]. Gate logits: [N, 1].
        logits = self.gate(h).squeeze(-1)  # [N]
        weights = F.softmax(logits, dim=0)  # sums to 1
        pooled = (weights.unsqueeze(-1) * h).sum(dim=0)  # [D]
        return pooled, weights


# --------------------------------------------------------------------------- #
# VetGigaGraph
# --------------------------------------------------------------------------- #


class VetGigaGraph(nn.Module):
    """Composed model: GNN + (optional) slide encoder + fusion + classifier.

    Construct directly with sub-modules (e.g. in tests) or via
    :meth:`from_config` from a project YAML config.
    """

    def __init__(
        self,
        gnn: GnnBackbone,
        slide_encoder: Optional[GigaPathSlideEncoder],
        fusion: _FusionBase,
        classifier: MlpClassifier,
        *,
        embed_dim: int = 1536,
        feature_adapter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.gnn = gnn
        self.slide_encoder = slide_encoder
        self.fusion = fusion
        self.classifier = classifier
        self.readout = GlobalAttentionReadout(in_dim=gnn.config.output_dim)
        self.embed_dim = int(embed_dim)
        # µPDCA #10: optional trainable residual MLP between the frozen
        # tile features and the GNN input. None = baseline (no adapter).
        self.feature_adapter = feature_adapter

        if fusion.uses_slide and slide_encoder is None:
            raise ValueError(
                f"fusion.{fusion.name} uses_slide=True but no slide_encoder was provided"
            )

    # --- public API ------------------------------------------------------ #

    def forward(
        self,
        graph: Data,
        tile_embeddings: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attentions: dict[str, torch.Tensor] = {}

        # --- GNN branch ---
        if self.fusion.uses_gnn:
            # µPDCA #9 D-23: forward node_type / edge_type if the graph carries
            # them (heterogeneous variant). Vanilla GnnBackbone ignores these
            # when use_heterogeneous=False. Hetero-GAT (use_heterogeneous=True)
            # consumes them via node-type embedding + edge-type-augmented
            # edge_attr.
            extra_kwargs = {}
            node_type = getattr(graph, "node_type", None)
            edge_type = getattr(graph, "edge_type", None)
            if node_type is not None:
                extra_kwargs["node_type"] = node_type
            if edge_type is not None:
                extra_kwargs["edge_type"] = edge_type
            # µPDCA #10: apply trainable feature-space adapter on the frozen
            # tile features before GAT. None (default) → identity.
            gnn_x = self.feature_adapter(graph.x) if self.feature_adapter is not None else graph.x
            h_nodes, gnn_attns = self.gnn(
                gnn_x, graph.edge_index, graph.edge_attr, **extra_kwargs
            )
            for k, v in gnn_attns.items():
                attentions[f"gnn.{k}"] = _detach_cpu(v)
            h_gnn, node_weights = self.readout(h_nodes)
            attentions["gnn.readout.node_attention"] = _detach_cpu(node_weights)
            # µPDCA #8 Phase C M10: per-tile features for multi-task auxiliary
            # head. Kept WITH gradient so aux loss can backprop into the GNN.
            # All other entries in `attentions` are detached + on CPU (used
            # only for analysis); this key is the exception.
            attentions["gnn.last_hidden"] = h_nodes
        else:
            h_gnn = None

        # --- Slide branch ---
        if self.fusion.uses_slide:
            assert self.slide_encoder is not None  # checked at __init__
            h_slide, cls_attn = self.slide_encoder(tile_embeddings, coordinates)
            attentions["slide_encoder.cls_attention"] = _detach_cpu(cls_attn)
        else:
            h_slide = None

        # --- Fusion + classifier ---
        # The fusion module declares which sub-key it surfaces via
        # ``fusion.attention_key`` (e.g. "alpha", "gate", "cross_attention").
        # Strategies that don't carry a per-strategy attention tensor leave
        # ``attention_key=None``, and the absent-key rule in design §4.3
        # means we simply omit the entry.
        h_fused, attn_value = self.fusion(h_gnn, h_slide)
        if attn_value is not None and self.fusion.attention_key is not None:
            attentions[f"fusion.{self.fusion.attention_key}"] = _detach_cpu(attn_value)
        logits = self.classifier(h_fused)
        return logits, attentions

    # --- factory --------------------------------------------------------- #

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        slide_backbone: Optional[nn.Module] = None,
        slide_backbone_loader=None,
    ) -> "VetGigaGraph":
        """Build a :class:`VetGigaGraph` from the project YAML config.

        Args:
            config: Top-level config dict (``configs/default.yaml`` shape).
            slide_backbone: Optional pre-built slide-encoder backbone
                (passed straight to :class:`GigaPathSlideEncoder`).
                Required for unit tests; production callers will pass
                ``slide_backbone_loader=`` instead.
            slide_backbone_loader: Optional zero-arg callable returning
                the slide backbone. Mutually exclusive with
                ``slide_backbone``.
        """
        gnn = build_gnn_backbone(config)
        fusion = build_fusion_module(config)

        slide_encoder: Optional[GigaPathSlideEncoder] = None
        if fusion.uses_slide:
            slide_cfg = _slide_section(config)
            # The LongNet slide encoder ``gigapath_slide_enc12l768d`` emits a
            # 768-d slide embedding (not 1536-d). The default 1536 matches the
            # tile-encoder output (which is the slide encoder's INPUT). For
            # the wrapper's projection, we need the slide encoder's actual
            # OUTPUT dim. Operators set this via ``model.slide_encoder.embed_dim``.
            slide_encoder = GigaPathSlideEncoder(
                model=slide_backbone,
                model_loader=slide_backbone_loader,
                embed_dim=int(slide_cfg.get("embed_dim", 1536)),
                proj_dim=int(slide_cfg.get("proj_dim", 256)),
                frozen=bool(slide_cfg.get("frozen", False)),
            )

        cls_cfg = _classifier_section(config)
        classifier = MlpClassifier(
            input_dim=int(_proj_dim(config)),
            hidden_dim=int(cls_cfg.get("hidden_dim", 128)),
            num_classes=int(_num_classes(config)),
            dropout=float(cls_cfg.get("dropout", 0.25)),
        )

        # µPDCA #10: optional feature-space adapter.
        feature_adapter = None
        adapter_cfg = config.get("model", {}).get("feature_adapter", {})
        if adapter_cfg and bool(adapter_cfg.get("enabled", False)):
            from src.models.feature_adapter import FeatureAdapter
            feature_adapter = FeatureAdapter(
                embed_dim=int(_embed_dim(config)),
                hidden_dim=int(adapter_cfg.get("hidden_dim", 512)),
                dropout=float(adapter_cfg.get("dropout", 0.1)),
                zero_init=bool(adapter_cfg.get("zero_init", True)),
            )

        return cls(
            gnn=gnn,
            slide_encoder=slide_encoder,
            fusion=fusion,
            classifier=classifier,
            embed_dim=int(_embed_dim(config)),
            feature_adapter=feature_adapter,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _detach_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu")


def _slide_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "model" in config and "slide_encoder" in config["model"]:
        return config["model"]["slide_encoder"]
    if "slide_encoder" in config:
        return config["slide_encoder"]
    return {}


def _classifier_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "model" in config and "classifier" in config["model"]:
        return config["model"]["classifier"]
    if "classifier" in config:
        return config["classifier"]
    return {}


def _proj_dim(config: Mapping[str, Any]) -> int:
    if isinstance(config, Mapping):
        model = config.get("model", {})
        gnn = model.get("gnn") if isinstance(model, Mapping) else None
        if isinstance(gnn, Mapping) and "output_dim" in gnn:
            return int(gnn["output_dim"])
    return 256


def _num_classes(config: Mapping[str, Any]) -> int:
    if isinstance(config, Mapping):
        proj = config.get("project", {})
        if isinstance(proj, Mapping) and "num_classes" in proj:
            return int(proj["num_classes"])
    return 7


def _embed_dim(config: Mapping[str, Any]) -> int:
    if isinstance(config, Mapping):
        fe = config.get("feature_extraction", {})
        if isinstance(fe, Mapping) and "embedding_dim" in fe:
            return int(fe["embedding_dim"])
    return 1536
