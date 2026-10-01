"""Prov-GigaPath tile encoder wrapper (Stage 2).

Wraps the timm-hosted ``hf_hub:prov-gigapath/prov-gigapath`` ViT-Giant
behind a small project-local API:

* All parameters are frozen on construction (``requires_grad=False``).
  An :class:`FrozenEncoderError` is raised immediately if any parameter
  escapes the freeze — this catches a class of refactor bugs where a
  layer is silently re-instantiated and forgets to inherit the frozen
  state.
* ``forward(x)`` always returns ``[B, embedding_dim]`` (default 1536).
  We wrap timm's ``num_classes=0`` mode so callers don't have to know
  whether the model returns CLS-token or pooled features internally.
* Tests can inject a small stand-in via ``model=``, avoiding the 1.13B-
  parameter HF download in CI. The default constructor still loads the
  real GigaPath when ``model`` is omitted.

References:
    Design: docs/02-design/03-architecture.md §3 (Module B — Feature Extraction)
    Design: docs/02-design/03-architecture.md §3.B (Verification, 8 cases)
    Plan:   docs/02-design/02-data-spec.md §6.4 (post-extraction acceptance gate)
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch
import torch.nn as nn

from src.utils.errors import FrozenEncoderError

logger = logging.getLogger(__name__)

#: Locked timm hub spec for the published GigaPath checkpoint.
DEFAULT_GIGAPATH_MODEL = "hf_hub:prov-gigapath/prov-gigapath"

#: Locked embedding dimensionality (matches HDF5 schema in design §3.B).
GIGAPATH_EMBEDDING_DIM = 1536

#: Locked input resolution. Prov-GigaPath's tile encoder is a ViT trained
#: at 224×224. The tile-grade pipeline upstream produces 256×256 tiles,
#: so a per-tile resize happens in :class:`TileDirectoryDataset`.
GIGAPATH_INPUT_SIZE = 224


class GigaPathTileEncoder(nn.Module):
    """Frozen ViT tile encoder; ``forward`` returns ``[B, 1536]`` embeddings.

    Args:
        model: An optional pre-built backbone. When provided, takes
            precedence over ``model_name`` (used for unit tests with a
            small stand-in network). Must accept ``[B, 3, H, W]`` and
            return ``[B, embedding_dim]``.
        model_name: timm model spec; default is the locked GigaPath HF
            hub path.
        embedding_dim: Expected output dimensionality. Used to validate
            the backbone matches the design contract.
        device: Optional device to move the encoder to immediately. If
            ``None``, the model stays on whatever device ``model`` was
            constructed on (CPU by default for timm).
        pretrained: Forwarded to ``timm.create_model``. Default ``True``
            so the production path always loads the real weights.
    """

    EMBEDDING_DIM = GIGAPATH_EMBEDDING_DIM
    DEFAULT_MODEL = DEFAULT_GIGAPATH_MODEL
    INPUT_SIZE = GIGAPATH_INPUT_SIZE

    def __init__(
        self,
        *,
        model: Optional[nn.Module] = None,
        model_name: str = DEFAULT_GIGAPATH_MODEL,
        embedding_dim: int = GIGAPATH_EMBEDDING_DIM,
        device: Optional[Union[str, torch.device]] = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = int(embedding_dim)

        if model is None:
            model = self._build_default_backbone(model_name, pretrained=pretrained)
        self.model = model

        # Freeze BEFORE moving to device so the assertion below sees the
        # initial parameter set.
        self._freeze()
        self.eval()

        if device is not None:
            self.to(device)

    # --- public API ------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of tiles into ``[B, embedding_dim]`` embeddings.

        ``torch.no_grad()`` is **not** used here: the caller is expected
        to wrap the inference loop appropriately. We rely on the frozen
        parameters to short-circuit autograd through this module.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"GigaPathTileEncoder expects [B, 3, H, W]; got {tuple(x.shape)}"
            )
        out = self.model(x)
        if out.ndim != 2 or out.shape[-1] != self.embedding_dim:
            raise FrozenEncoderError(
                f"Backbone returned {tuple(out.shape)} but expected "
                f"[B, {self.embedding_dim}]. Did you forget num_classes=0?"
            )
        return out

    def assert_frozen(self) -> None:
        """Raise :class:`FrozenEncoderError` if any parameter is trainable.

        Call this before every training run as a defence-in-depth check
        against accidental ``.requires_grad_(True)`` calls in user code.
        """
        leaks = [n for n, p in self.named_parameters() if p.requires_grad]
        if leaks:
            raise FrozenEncoderError(
                f"GigaPath tile encoder must stay frozen; found "
                f"{len(leaks)} trainable params: {leaks[:3]}{'…' if len(leaks) > 3 else ''}"
            )

    # --- internals -------------------------------------------------------- #

    @staticmethod
    def _build_default_backbone(
        model_name: str,
        *,
        pretrained: bool,
    ) -> nn.Module:
        """Construct the GigaPath backbone via timm.

        ``num_classes=0`` removes the classification head so the model
        returns pooled features directly.
        """
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "timm is required to build GigaPathTileEncoder. "
                "Install via requirements.txt or pip install timm>=1.0.0"
            ) from e

        logger.info("Loading tile encoder backbone %s (pretrained=%s)", model_name, pretrained)
        return timm.create_model(model_name, pretrained=pretrained, num_classes=0)

    def _freeze(self) -> None:
        """Set ``requires_grad=False`` on all parameters and verify."""
        for p in self.parameters():
            p.requires_grad = False
        self.assert_frozen()
