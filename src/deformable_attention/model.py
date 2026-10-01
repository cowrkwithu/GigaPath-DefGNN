"""Deformable-attention GNN stack + VetGigaGraph wrapper for v2.

`DeformableAttentionStack` is a drop-in replacement for v1's `GnnBackbone`:
same forward signature (modulo the extra ``coords`` arg), same return type
``(h, attentions_dict)`` so v1's `VetGigaGraph` pipeline (`src/shared/`)
can host it. We provide a thin `VetGigaGraphDeformable` wrapper that
mirrors v1's `VetGigaGraph.from_config` factory while substituting our
stack for the GAT/GCN/GraphSAGE/GIN factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import DeformableAttentionLayer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DeformableAttentionConfig:
    """Hyperparameters for the deformable-attention stack."""

    in_dim: int = 1536
    hidden_dim: int = 128
    output_dim: int = 128
    num_layers: int = 2
    heads: int = 2
    dropout: float = 0.25

    # Deformable-attention specific
    num_offsets: int = 2
    offset_mlp_depth: int = 1
    offset_mlp_hidden: int = 64
    offset_init_scale: float = 0.05
    offset_penalty_weight: float = 1.0e-4
    offset_penalty_epochs: int = 20
    knn_k: int = 8
    knn_temperature: float = 1.0
    knn_chunk_size: int = 64
    fp32_offset_mlp: bool = True
    coord_normalization: str = "per_wsi_unit"   # per_wsi_unit | none

    def __post_init__(self) -> None:
        if self.num_layers < 1 or self.num_layers > 4:
            raise ValueError(f"num_layers in [1, 4]; got {self.num_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout in [0, 1); got {self.dropout}")
        if self.num_offsets < 1:
            raise ValueError("num_offsets must be ≥ 1")
        if self.coord_normalization not in ("per_wsi_unit", "none"):
            raise ValueError(f"unknown coord_normalization {self.coord_normalization!r}")


# --------------------------------------------------------------------------- #
# Stack
# --------------------------------------------------------------------------- #


class DeformableAttentionStack(nn.Module):
    """A stack of `DeformableAttentionLayer` blocks.

    Shape contract matches v1's `GnnBackbone`:
      - Input:  ``x [N, in_dim]``, ``edge_index [2, E]``, plus the v2 extra
        ``coords [N, 2]``.
      - Output: ``h [N, output_dim]`` and an ``attentions`` dict keyed by
        ``layer_<i>.deformable.{attention,offsets}``.
    """

    def __init__(self, config: DeformableAttentionConfig) -> None:
        super().__init__()
        self.config = config
        # Build widths: [in, hidden, hidden, ..., output]
        widths = (
            [config.in_dim]
            + [config.hidden_dim] * (config.num_layers - 1)
            + [config.output_dim]
        )
        self.layers = nn.ModuleList(
            DeformableAttentionLayer(
                in_dim=widths[i],
                out_dim=widths[i + 1],
                heads=config.heads,
                num_offsets=config.num_offsets,
                offset_mlp_hidden=config.offset_mlp_hidden,
                offset_mlp_depth=config.offset_mlp_depth,
                offset_init_scale=config.offset_init_scale,
                knn_k=config.knn_k,
                knn_temperature=config.knn_temperature,
                knn_chunk_size=config.knn_chunk_size,
                fp32_offset_mlp=config.fp32_offset_mlp,
                dropout=config.dropout,
            )
            for i in range(config.num_layers)
        )

    def _normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if self.config.coord_normalization == "per_wsi_unit":
            mn = coords.min(dim=0, keepdim=True).values
            mx = coords.max(dim=0, keepdim=True).values
            span = (mx - mn).clamp_min(1.0)
            return (coords - mn) / span
        return coords

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,            # accepted for API parity, unused
        node_type: torch.Tensor | None = None,            # ditto
        edge_type: torch.Tensor | None = None,            # ditto
        *,
        coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if coords is None:
            raise ValueError(
                "DeformableAttentionStack requires `coords` (tile (x,y))."
            )
        coords = self._normalize_coords(coords)
        attentions: dict[str, torch.Tensor] = {}
        h = x
        last_idx = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h, layer_attn = layer(h, coords, edge_index)
            for k, v in layer_attn.items():
                attentions[f"layer_{i + 1}.{k}"] = v
            if i < last_idx:
                h = F.relu(h)
                h = F.dropout(h, p=self.config.dropout, training=self.training)
        return h, attentions

    # API parity with v1 GnnBackbone (used by VetGigaGraph.from_config).
    @property
    def output_dim(self) -> int:
        return self.config.output_dim


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #


def build_deformable_backbone(config: Mapping[str, Any]) -> DeformableAttentionStack:
    """Construct a `DeformableAttentionStack` from a project YAML config.

    Reads `model.gnn` and `model.gnn.deformable`, mirroring v1's shape so
    the same Hydra config schema works.
    """
    gnn_cfg = config["model"]["gnn"]
    deform_cfg = gnn_cfg.get("deformable", {})
    return DeformableAttentionStack(
        DeformableAttentionConfig(
            in_dim=int(gnn_cfg.get("in_dim", 1536)),
            hidden_dim=int(gnn_cfg.get("hidden_dim", 128)),
            output_dim=int(gnn_cfg.get("output_dim", 128)),
            num_layers=int(gnn_cfg.get("layers", 2)),
            heads=int(gnn_cfg.get("heads", 2)),
            dropout=float(gnn_cfg.get("dropout", 0.25)),
            num_offsets=int(deform_cfg.get("num_offsets", 2)),
            offset_mlp_depth=int(deform_cfg.get("offset_mlp_depth", 1)),
            offset_mlp_hidden=int(deform_cfg.get("offset_mlp_hidden", 64)),
            offset_init_scale=float(deform_cfg.get("offset_init_scale", 0.05)),
            offset_penalty_weight=float(deform_cfg.get("offset_penalty_weight", 1.0e-4)),
            offset_penalty_epochs=int(deform_cfg.get("offset_penalty_epochs", 20)),
            knn_k=int(deform_cfg.get("knn_k", 8)),
            knn_temperature=float(deform_cfg.get("knn_temperature", 1.0)),
            knn_chunk_size=int(deform_cfg.get("knn_chunk_size", 64)),
            fp32_offset_mlp=bool(deform_cfg.get("fp32_offset_mlp", True)),
            coord_normalization=str(deform_cfg.get("coord_normalization", "per_wsi_unit")),
        )
    )


def build_deformable_model(
    config: Mapping[str, Any],
    *,
    slide_backbone=None,
    slide_backbone_loader=None,
):
    """Construct a v1 `VetGigaGraph` model with the GNN slot replaced by
    our `DeformableAttentionStack`. Reuses v1's slide encoder, fusion,
    classifier, and feature-adapter machinery via `src.shared.*`.

    We monkey-patch the GNN slot rather than copy the whole composition
    so that v1 fixes (e.g. fusion-strategy plumbing) flow into v2 for free.
    """
    # v1 modules import from `src.X` absolutely; we put pw-vetGigagraph/ on
    # sys.path in the training script so these imports resolve to v1.
    from src.models.vetgigagraph import VetGigaGraph
    from src.models.fusion import build_fusion_module
    from src.models.classifier import MlpClassifier
    from src.models.gigapath_slide import GigaPathSlideEncoder

    gnn = build_deformable_backbone(config)
    fusion = build_fusion_module(config)

    slide_encoder = None
    if fusion.uses_slide:
        slide_cfg = config["model"]["slide_encoder"]
        slide_encoder = GigaPathSlideEncoder(
            model=slide_backbone,
            model_loader=slide_backbone_loader,
            embed_dim=int(slide_cfg.get("embed_dim", 1536)),
            proj_dim=int(slide_cfg.get("proj_dim", 256)),
            frozen=bool(slide_cfg.get("frozen", False)),
        )

    cls_cfg = config["model"]["classifier"]
    classifier = MlpClassifier(
        input_dim=int(_proj_dim_v2(config)),
        hidden_dim=int(cls_cfg.get("hidden_dim", 128)),
        num_classes=int(config["project"]["num_classes"]),
        dropout=float(cls_cfg.get("dropout", 0.25)),
    )

    model = VetGigaGraph(
        gnn=gnn,                       # duck-typed: needs .config.output_dim + forward
        slide_encoder=slide_encoder,
        fusion=fusion,
        classifier=classifier,
        embed_dim=int(config["feature_extraction"]["embedding_dim"]),
    )
    # Patch VetGigaGraph's `forward` so it threads `coords` into our stack.
    _wrap_gnn_call_with_coords(model)
    return model


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


# A `DeformableAttentionStack` exposes `.config` (a dataclass) so VetGigaGraph
# can read `.gnn.config.output_dim` exactly like with v1's `GnnBackbone`.
# Forward signature is wider (extra `coords`), so we shim it inside VetGigaGraph.

def _wrap_gnn_call_with_coords(model) -> None:
    """Monkey-patch model.forward to pass `coords` into the GNN call.

    v1's `VetGigaGraph.forward(graph, tile_embeddings, coordinates)` calls
    `self.gnn(graph.x, graph.edge_index, graph.edge_attr, **extra)` without
    coords. Our deformable stack needs them. We wrap the original forward so
    when it reaches the GNN call, we inject `coords=graph.pos`.
    """
    original_forward = model.forward
    original_gnn = model.gnn

    class _CoordInjectingGnn(nn.Module):
        """Forwarder that pulls `coords` from the bound graph via a closure."""
        def __init__(self, real: nn.Module) -> None:
            super().__init__()
            self.real = real
            self._current_coords: Optional[torch.Tensor] = None
            # Expose `.config` so VetGigaGraph can introspect output_dim.
            self.config = real.config

        def set_coords(self, coords: torch.Tensor) -> None:
            self._current_coords = coords

        def forward(self, x, edge_index, edge_attr=None, **kwargs):
            return self.real(
                x, edge_index, edge_attr, coords=self._current_coords, **kwargs
            )

    injecting = _CoordInjectingGnn(original_gnn)
    model.gnn = injecting

    def patched_forward(graph, tile_embeddings, coordinates):
        # Use `graph.pos` if present (PyG convention), else `coordinates`.
        coords = getattr(graph, "pos", None)
        if coords is None:
            coords = coordinates
        injecting.set_coords(coords)
        return original_forward(graph, tile_embeddings, coordinates)

    model.forward = patched_forward


def _proj_dim_v2(config: Mapping[str, Any]) -> int:
    """Compute the dim flowing into the classifier (matches v1 `_proj_dim`)."""
    fusion_cfg = config["model"]["fusion"]
    strategy = str(fusion_cfg.get("strategy", "gnn_only"))
    gnn_out = int(config["model"]["gnn"].get("output_dim", 128))
    slide_proj = int(config["model"].get("slide_encoder", {}).get("proj_dim", 256))
    if strategy in ("gnn_only",):
        return gnn_out
    if strategy in ("slide_only",):
        return slide_proj
    if strategy in ("concat",):
        return gnn_out + slide_proj
    # weighted / cross_attention / gated assume both branches project to the
    # same dim — by convention v1 uses gnn_out for these.
    return gnn_out


# --------------------------------------------------------------------------- #
# Public alias
# --------------------------------------------------------------------------- #


#: Convenience name used by training script.
VetGigaGraphDeformable = build_deformable_model
