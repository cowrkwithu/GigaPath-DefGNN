"""Stage 4 — Model components.

Public surface:

* :class:`GnnBackbone` + :func:`build_gnn_backbone` (Phase 5.1)
* :class:`GigaPathSlideEncoder` (Phase 5.2)
* :class:`_FusionBase` subclasses + :func:`build_fusion_module` (Phase 5.3)
* :class:`MlpClassifier` (Phase 5.4)
* :class:`VetGigaGraph` integration (Phase 5.5)

See ``docs/02-design/03-architecture.md`` §5 for the design contract
and ``docs/02-design/features/vetgigagraph.design.md`` §4.3 for the
locked ``attention_weights`` schema.
"""

from src.models.classifier import MlpClassifier
from src.models.fusion import (
    SUPPORTED_FUSIONS,
    ConcatFusion,
    CrossAttentionFusion,
    FusionConfig,
    GatedFusion,
    GnnOnlyFusion,
    LearnableWeightedFusion,
    SlideOnlyFusion,
    build_fusion_module,
)
from src.models.gigapath_slide import (
    DEFAULT_GIGAPATH_SLIDE_MODEL,
    DEFAULT_PROJ_DIM,
    GigaPathSlideEncoder,
)
from src.models.gnn_backbones import (
    SUPPORTED_BACKBONES,
    GnnBackbone,
    GnnBackboneConfig,
    build_gnn_backbone,
)
from src.models.vetgigagraph import GlobalAttentionReadout, VetGigaGraph

__all__ = [
    "ConcatFusion",
    "CrossAttentionFusion",
    "DEFAULT_GIGAPATH_SLIDE_MODEL",
    "DEFAULT_PROJ_DIM",
    "FusionConfig",
    "GatedFusion",
    "GigaPathSlideEncoder",
    "GlobalAttentionReadout",
    "GnnBackbone",
    "GnnBackboneConfig",
    "GnnOnlyFusion",
    "LearnableWeightedFusion",
    "MlpClassifier",
    "SUPPORTED_BACKBONES",
    "SUPPORTED_FUSIONS",
    "SlideOnlyFusion",
    "VetGigaGraph",
    "build_fusion_module",
    "build_gnn_backbone",
]
