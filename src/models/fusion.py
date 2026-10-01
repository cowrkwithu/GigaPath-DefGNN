"""Fusion module factory (Phase 5.3) — 6 strategies.

Each strategy combines two same-width vectors:

* ``h_gnn`` — pooled graph representation (``[proj_dim]``).
* ``h_slide`` — projected slide-encoder output (``[proj_dim]``).

and emits a fused vector ``[proj_dim]`` plus an optional ``alpha``
scalar (only the ``learnable_weighted`` strategy reports one). The
``alpha`` scalar surfaces in :class:`VetGigaGraph`'s ``attention_weights``
dict under the locked key ``"fusion.alpha"``.

Strategies (locked names match
``configs/default.yaml model.fusion.strategy``):

* ``gnn_only`` — return ``h_gnn``; ``h_slide`` is ignored.
* ``slide_only`` — return ``h_slide``; ``h_gnn`` is ignored.
* ``concat`` — concat then project to ``proj_dim``.
* ``learnable_weighted`` — ``sigmoid(α) * h_gnn + (1 − sigmoid(α)) * h_slide``.
* ``cross_attention`` — single-head self-attention over the 2-token
  ``[h_gnn; h_slide]`` sequence; output is the mean of the attended tokens.
* ``gated`` — element-wise gating: ``g * h_gnn + (1 − g) * h_slide`` with
  ``g = σ(W [h_gnn ‖ h_slide])``.

References:
    Design: docs/02-design/03-architecture.md §5 (Module D)
    Design: docs/02-design/features/vetgigagraph.design.md §4.3 (attention schema, fusion.alpha)
    Tests:  docs/02-design/03-architecture.md §5.D rows 8–9
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#: Locked strategy names — match ``configs/default.yaml model.fusion.strategy``.
SUPPORTED_FUSIONS = (
    "gnn_only",
    "slide_only",
    "concat",
    "learnable_weighted",
    "cross_attention",
    "gated",
)


@dataclass(frozen=True)
class FusionConfig:
    strategy: str = "learnable_weighted"
    proj_dim: int = 256
    init_alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.strategy not in SUPPORTED_FUSIONS:
            raise ValueError(
                f"fusion strategy must be one of {SUPPORTED_FUSIONS}; "
                f"got {self.strategy!r}"
            )
        if self.proj_dim <= 0:
            raise ValueError(f"proj_dim must be positive; got {self.proj_dim}")


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class _FusionBase(nn.Module):
    """All fusion modules implement ``uses_gnn``/``uses_slide`` so the
    parent :class:`VetGigaGraph` can short-circuit unused branches.

    ``attention_key`` declares which sub-key under ``fusion.<key>`` this
    strategy surfaces in the locked ``attention_weights`` schema
    (`vetgigagraph.design.md` §4.3). Strategies with no per-strategy
    attention tensor leave it as ``None`` — :class:`VetGigaGraph` then
    omits the key (the design's "absent if ..." rule).

    The forward return is ``(fused, attn_value)`` where ``attn_value``'s
    shape depends on ``attention_key`` per the locked schema:
        * ``"alpha"``           → scalar      (learnable_weighted)
        * ``"gate"``            → ``[D]``     (gated, element-wise gate)
        * ``"cross_attention"`` → ``[1, 1]``  (cross_attention, score)
    """

    name: str = ""
    uses_gnn: bool = True
    uses_slide: bool = True
    #: Sub-key under "fusion." in the attention_weights schema (None = no key emitted).
    attention_key: Optional[str] = None

    def forward(  # type: ignore[override]
        self,
        h_gnn: Optional[torch.Tensor],
        h_slide: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #


class GnnOnlyFusion(_FusionBase):
    name = "gnn_only"
    uses_slide = False

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, h_gnn, h_slide):  # type: ignore[override]
        if h_gnn is None:
            raise ValueError("gnn_only fusion requires h_gnn; got None")
        return h_gnn, None


class SlideOnlyFusion(_FusionBase):
    name = "slide_only"
    uses_gnn = False

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, h_gnn, h_slide):  # type: ignore[override]
        if h_slide is None:
            raise ValueError("slide_only fusion requires h_slide; got None")
        return h_slide, None


class ConcatFusion(_FusionBase):
    name = "concat"

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(2 * config.proj_dim, config.proj_dim)

    def forward(self, h_gnn, h_slide):  # type: ignore[override]
        _require_pair("concat", h_gnn, h_slide)
        return self.proj(torch.cat([h_gnn, h_slide], dim=-1)), None


class LearnableWeightedFusion(_FusionBase):
    name = "learnable_weighted"
    attention_key = "alpha"

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        # Initialize alpha so sigmoid(alpha) ≈ init_alpha.
        init_alpha = float(config.init_alpha)
        if not 0.0 < init_alpha < 1.0:
            raise ValueError(
                f"init_alpha must lie in (0, 1) for learnable_weighted; got {init_alpha}"
            )
        logit_init = torch.tensor(
            torch.logit(torch.tensor(init_alpha)).item(), dtype=torch.float32
        )
        self.alpha_logit = nn.Parameter(logit_init)

    def forward(self, h_gnn, h_slide):  # type: ignore[override]
        _require_pair("learnable_weighted", h_gnn, h_slide)
        alpha = torch.sigmoid(self.alpha_logit)
        fused = alpha * h_gnn + (1.0 - alpha) * h_slide
        return fused, alpha


class CrossAttentionFusion(_FusionBase):
    name = "cross_attention"
    attention_key = "cross_attention"

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(
            embed_dim=config.proj_dim,
            num_heads=4,
            batch_first=True,
        )

    def forward(self, h_gnn, h_slide):  # type: ignore[override]
        _require_pair("cross_attention", h_gnn, h_slide)
        # Pack into a 2-token sequence [batch=1, seq=2, dim=proj_dim].
        seq = torch.stack([h_gnn, h_slide], dim=0).unsqueeze(0)  # [1, 2, D]
        # ``need_weights=True`` averages the per-head attention weights so
        # ``attn_weights`` has shape ``[1, seq=2, seq=2]``. The
        # h_gnn → h_slide cross-token score lives at ``[0, 0, 1]``;
        # the design schema asks for a ``[1, 1]`` tensor at that key.
        attended, attn_weights = self.attn(seq, seq, seq, need_weights=True)
        cross_score = attn_weights[0, 0, 1].reshape(1, 1)
        return attended.squeeze(0).mean(dim=0), cross_score


class GatedFusion(_FusionBase):
    name = "gated"
    attention_key = "gate"

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Linear(2 * config.proj_dim, config.proj_dim)

    def forward(self, h_gnn, h_slide):  # type: ignore[override]
        _require_pair("gated", h_gnn, h_slide)
        # The gate vector itself is the per-strategy attention tensor —
        # surface it under ``fusion.gate`` per design §4.3.
        g = torch.sigmoid(self.gate(torch.cat([h_gnn, h_slide], dim=-1)))
        return g * h_gnn + (1.0 - g) * h_slide, g


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


_FUSION_REGISTRY: dict[str, type[_FusionBase]] = {
    "gnn_only": GnnOnlyFusion,
    "slide_only": SlideOnlyFusion,
    "concat": ConcatFusion,
    "learnable_weighted": LearnableWeightedFusion,
    "cross_attention": CrossAttentionFusion,
    "gated": GatedFusion,
}


def build_fusion_module(
    config: Mapping[str, Any] | FusionConfig,
) -> _FusionBase:
    """Build a fusion module from the project YAML config.

    Accepts either a :class:`FusionConfig` directly or any mapping that
    follows the ``configs/default.yaml`` ``model.fusion`` shape.
    """
    if isinstance(config, FusionConfig):
        cfg = config
    else:
        fusion_cfg = _narrow_to_fusion(config)
        proj_dim = _resolve_proj_dim(config)
        cfg = FusionConfig(
            strategy=str(fusion_cfg["strategy"]),
            proj_dim=int(proj_dim),
            init_alpha=float(fusion_cfg.get("init_alpha", 0.5)),
        )
    cls = _FUSION_REGISTRY[cfg.strategy]
    logger.info("Building fusion module: %s (proj_dim=%d)", cfg.strategy, cfg.proj_dim)
    return cls(cfg)


def _narrow_to_fusion(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "strategy" in config:
        return config
    if "fusion" in config:
        return config["fusion"]
    if "model" in config and "fusion" in config["model"]:
        return config["model"]["fusion"]
    raise KeyError("build_fusion_module: could not locate `fusion` section in config")


def _resolve_proj_dim(config: Mapping[str, Any]) -> int:
    """Look up ``model.gnn.output_dim`` (== ``model.slide_encoder.proj_dim``)."""
    if isinstance(config, Mapping):
        model = config.get("model")
        if isinstance(model, Mapping):
            gnn = model.get("gnn")
            if isinstance(gnn, Mapping) and "output_dim" in gnn:
                return int(gnn["output_dim"])
            slide = model.get("slide_encoder")
            if isinstance(slide, Mapping) and "proj_dim" in slide:
                return int(slide["proj_dim"])
    return 256


def _require_pair(
    name: str,
    h_gnn: Optional[torch.Tensor],
    h_slide: Optional[torch.Tensor],
) -> None:
    if h_gnn is None or h_slide is None:
        raise ValueError(
            f"{name} fusion requires both h_gnn and h_slide; got "
            f"{'None' if h_gnn is None else 'tensor'} / "
            f"{'None' if h_slide is None else 'tensor'}"
        )
    if h_gnn.shape != h_slide.shape:
        raise ValueError(
            f"{name} fusion: shape mismatch h_gnn {tuple(h_gnn.shape)} vs "
            f"h_slide {tuple(h_slide.shape)}"
        )
