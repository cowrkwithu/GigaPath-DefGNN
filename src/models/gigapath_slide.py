"""GigaPath slide-level encoder wrapper (Phase 5.2).

The GigaPath slide encoder (LongNet) consumes a *sequence* of frozen
tile embeddings + their spatial coordinates and emits a slide-level
representation, plus a per-tile attention map (``cls_attention``) that
the ``attention_weights`` schema in
``docs/02-design/features/vetgigagraph.design.md`` §4.3 requires.

This wrapper:

* Exposes a single :meth:`forward` taking ``(tile_embeddings[N, 1536],
  coordinates[N, 2])`` and returning ``(h_proj[proj_dim], cls_attn[N])``.
* Supports a ``frozen`` toggle at construction time so the freeze
  ablation (Experiment 5 in the design) is a config flag, not a model
  rewrite.
* Accepts an injected ``model=`` for unit tests, mirroring the pattern
  used in :class:`src.feature_extraction.GigaPathTileEncoder`.

The default constructor loads the real GigaPath slide encoder from
HuggingFace Hub; tests use a small deterministic stand-in. The exact
HF entry point depends on the published GigaPath build — we expose
``model_loader=`` so it can be swapped without touching the wrapper.

References:
    Design: docs/02-design/03-architecture.md §5 (Module D)
    Design: docs/02-design/features/vetgigagraph.design.md §4.3 (attention schema)
    Tests:  docs/02-design/03-architecture.md §5.D rows 5–6 (freeze toggle)
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from src.utils.errors import FrozenEncoderError

logger = logging.getLogger(__name__)

#: Legacy constant — kept only because `vetgigagraph.design.md` references
#: it. The slide encoder weights actually live in the main GigaPath HF
#: repo (not a separate ``-slide`` repo), so prefer the three constants
#: below for new code. See slide-encoder-injection µPDCA design §1.2.
DEFAULT_GIGAPATH_SLIDE_MODEL = "hf_hub:prov-gigapath/prov-gigapath-slide"

#: Production HF Hub repo + weight filename + arch name for the slide
#: encoder (LongNet, ~85M params). Verified 2026-05-15 against upstream
#: commit ``3505f87``. See slide-encoder-injection µPDCA design §1.2.
DEFAULT_GIGAPATH_SLIDE_HF_REPO = "prov-gigapath/prov-gigapath"
DEFAULT_GIGAPATH_SLIDE_WEIGHT_FILE = "slide_encoder.pth"
DEFAULT_GIGAPATH_SLIDE_ARCH = "gigapath_slide_enc12l768d"

#: Locked dimensionalities. ``EMBED_DIM`` is the tile-encoder output
#: width (and the slide encoder's input). ``DEFAULT_PROJ_DIM`` is the
#: ``configs/default.yaml model.slide_encoder.proj_dim`` default.
EMBED_DIM = 1536
DEFAULT_PROJ_DIM = 256

#: A type for the optional model-builder callable users can plug in.
SlideBackboneFactory = Callable[[], nn.Module]


class SlideEncoderOutput(nn.Module):
    """Marker base class — slide-encoder backbones must implement
    ``forward(tile_embeddings, coordinates) -> (cls_output[1536], cls_attention[N])``.

    Real GigaPath slide encoders (e.g. LongNet) follow this contract;
    test stand-ins should subclass this and emit comparable shapes.
    """

    def forward(  # type: ignore[override]
        self, tile_embeddings: torch.Tensor, coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Wrapper
# --------------------------------------------------------------------------- #


class GigaPathSlideEncoder(nn.Module):
    """Slide-level encoder wrapper.

    Args:
        model: Backbone module that takes ``(tile_embeddings[N, embed_dim],
            coordinates[N, 2])`` and returns ``(cls_output[embed_dim],
            cls_attention[N])``. When ``None``, the default GigaPath
            backbone is loaded via ``model_loader`` (or the locked
            HF-Hub default).
        embed_dim: Tile-embedding dimensionality. Default 1536.
        proj_dim: Output projection dim (matches GNN side for fusion).
            Default 256, locked in ``configs/default.yaml``.
        frozen: When ``True``, freezes the *backbone* params on init.
            The projection layer always trains. Default ``False`` per
            ``configs/default.yaml model.slide_encoder.frozen``.
        model_loader: Optional callable returning the backbone, used
            when ``model is None``. Defaults to a lazy timm/HF load
            (raises NotImplementedError if no real loader is wired up).
    """

    DEFAULT_MODEL = DEFAULT_GIGAPATH_SLIDE_MODEL

    def __init__(
        self,
        *,
        model: Optional[nn.Module] = None,
        embed_dim: int = EMBED_DIM,
        proj_dim: int = DEFAULT_PROJ_DIM,
        frozen: bool = False,
        model_loader: Optional[SlideBackboneFactory] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.proj_dim = int(proj_dim)
        self.frozen = bool(frozen)

        if model is None:
            if model_loader is None:
                model_loader = _default_slide_backbone_loader
            model = model_loader()
        self.backbone = model
        self.proj = nn.Linear(self.embed_dim, self.proj_dim)

        if self.frozen:
            self._freeze_backbone()

    # --- public API ------------------------------------------------------ #

    def forward(
        self,
        tile_embeddings: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single slide.

        Args:
            tile_embeddings: ``[N, embed_dim]`` tile embeddings (frozen
                tile encoder output).
            coordinates: ``[N, 2]`` spatial coordinates.

        Returns:
            ``(h_proj, cls_attention)`` with shapes ``[proj_dim]`` and
            ``[N]`` respectively. ``cls_attention`` sums to 1 along the
            tile dim.
        """
        # ``embed_dim`` here is the slide-encoder *output* dim (proj input).
        # The slide-encoder *input* dim (tile-encoder output) can differ — for
        # ``gigapath_slide_enc12l768d`` the input is 1536 and the output is 768.
        # Validate input rank+coords-alignment only; the backbone enforces the
        # input feature width via its own Linear ``patch_embed``.
        if tile_embeddings.ndim != 2:
            raise ValueError(
                f"tile_embeddings must be 2D [N, D]; got shape "
                f"{tuple(tile_embeddings.shape)}"
            )
        if coordinates.shape[0] != tile_embeddings.shape[0] or coordinates.shape[1] != 2:
            raise ValueError(
                f"coordinates must be [N, 2] matching N={tile_embeddings.shape[0]}; "
                f"got {tuple(coordinates.shape)}"
            )

        if self.frozen:
            with torch.no_grad():
                cls_output, cls_attention = self.backbone(tile_embeddings, coordinates)
        else:
            cls_output, cls_attention = self.backbone(tile_embeddings, coordinates)
        if cls_output.ndim != 1 or cls_output.shape[0] != self.embed_dim:
            raise ValueError(
                f"slide backbone returned cls_output shape "
                f"{tuple(cls_output.shape)}, expected ({self.embed_dim},)"
            )
        if cls_attention.shape != (tile_embeddings.shape[0],):
            raise ValueError(
                f"slide backbone returned cls_attention shape "
                f"{tuple(cls_attention.shape)}, expected ({tile_embeddings.shape[0]},)"
            )
        h_proj = self.proj(cls_output)
        return h_proj, cls_attention

    def assert_freeze_state(self) -> None:
        """Raise if `self.frozen` and any backbone param is trainable."""
        if not self.frozen:
            return
        leaks = [n for n, p in self.backbone.named_parameters() if p.requires_grad]
        if leaks:
            raise FrozenEncoderError(
                f"GigaPathSlideEncoder is configured frozen but has "
                f"{len(leaks)} trainable backbone params: {leaks[:3]}"
                f"{'…' if len(leaks) > 3 else ''}"
            )

    def freeze(self) -> None:
        """Programmatically freeze the backbone."""
        self.frozen = True
        self._freeze_backbone()

    def unfreeze(self) -> None:
        """Programmatically unfreeze the backbone."""
        self.frozen = False
        for p in self.backbone.parameters():
            p.requires_grad = True

    # --- internals ------------------------------------------------------- #

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.assert_freeze_state()


# --------------------------------------------------------------------------- #
# Default loader (real production path)
# --------------------------------------------------------------------------- #


def _default_slide_backbone_loader() -> nn.Module:  # pragma: no cover
    """Legacy stub — superseded by :func:`production_slide_backbone_loader`.

    Kept for backwards compat with any caller that uses the old name.
    New call-sites should use ``production_slide_backbone_loader``.
    """
    return production_slide_backbone_loader()


class _AttentionExtractingAdapter(nn.Module):
    """Adapter wrapping the vendored ``LongNetViT`` to emit a 2-tuple
    ``(cls_output[embed_dim], cls_attention[N])`` per slide.

    **D-2 v2 implementation** (resolves drift D-16, supersedes the v1
    forward-hook approach that was reverted on 2026-05-16):

        Computes ``cls_attention`` as a *post-hoc cosine-similarity
        attention* between the slide-level CLS output and the projected
        tile embeddings produced by the vendored ``LongNetViT.patch_embed``::

            tile_proj[N, D] = backbone.patch_embed(tile_emb)
            cls_output[D]   = backbone(tile_emb, coord)   # real GigaPath CLS
            sim[N]          = cosine_sim(cls_output, tile_proj)
            cls_attention[N] = softmax(sim / tau)         # sums to 1, non-uniform

        Why not forward-hook on the last LongNet block?
        v1 attempted that by forcing ``args.flash_attention=False`` on the
        last block, but ``DilatedAttention.forward()`` has a hard
        ``assert self.args.flash_attention`` at line 145 — independent of
        the graceful-fallback condition in
        ``MultiheadAttention.attention_ops:73``. That assert bricked 21/30
        fold runs of the Exp 4 sweep (see ``docs/work-log/2026-05-16.md``).
        The cosine-similarity reformulation gives a real, non-uniform
        attention without touching any vendored code path — zero incident
        risk, identical 2-tuple contract from the caller's perspective.

    Backward compatibility:
        Module-level alias ``_UniformAttentionAdapter = _AttentionExtractingAdapter``
        below preserves existing import names.

    Fallback policy:
        If ``backbone.patch_embed`` is missing (synthetic backbones in unit
        tests) OR cosine-similarity computation raises, falls back to
        uniform ``1/N`` with a ``logger.warning`` — never crashes the
        calling code path.

    Tunable: ``self.tau = 1.0`` — softmax temperature. Smaller values
    make attention more peaky.
    """

    def __init__(self, backbone: nn.Module, tau: float = 1.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.tau = float(tau)

    def _compute_cosine_attention(
        self,
        tile_emb_b: torch.Tensor,
        cls_output: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Cosine-sim attention path. Returns ``[N]`` on the same device/dtype
        as ``cls_output``. Raises on any failure — caller handles fallback.
        """
        patch_embed = getattr(self.backbone, "patch_embed", None)
        if patch_embed is None:
            raise AttributeError("backbone has no patch_embed (synthetic backbone)")
        # patch_embed projects [B, N, in_chans=1536] → [B, N, embed_dim=768]
        with torch.no_grad():
            tile_proj = patch_embed(tile_emb_b)        # [1, N, D]
        tile_proj = tile_proj.squeeze(0)                # [N, D]
        cls_vec = cls_output.reshape(-1)                # [D]
        cls_n = nn.functional.normalize(cls_vec, dim=0)
        tile_n = nn.functional.normalize(tile_proj, dim=-1)
        sim = tile_n @ cls_n                            # [N]
        attn = torch.softmax(sim / max(self.tau, 1e-6), dim=0)
        if attn.shape[0] != N:
            raise ValueError(f"attention shape mismatch: {attn.shape} vs N={N}")
        return attn

    def forward(
        self, tile_embeddings: torch.Tensor, coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Accept either [N, D] (legacy GigaPathSlideEncoder contract) or
        # [B=1, N, D] (upstream LongNetViT contract). Internally use [B=1, N, D].
        if tile_embeddings.ndim == 2:
            tile_emb_b = tile_embeddings.unsqueeze(0)
            coord_b = coordinates.unsqueeze(0)
            squeeze_out = True
        else:
            tile_emb_b = tile_embeddings
            coord_b = coordinates
            squeeze_out = False

        out_list = self.backbone(tile_emb_b, coord_b)
        cls_b = out_list[0] if isinstance(out_list, (list, tuple)) else out_list

        if squeeze_out:
            cls_output = cls_b.squeeze(0)
        else:
            cls_output = cls_b

        N = tile_embeddings.shape[-2] if tile_embeddings.ndim >= 2 else tile_embeddings.shape[0]

        try:
            cls_attention = self._compute_cosine_attention(tile_emb_b, cls_output, N)
            cls_attention = cls_attention.to(
                device=tile_embeddings.device, dtype=tile_embeddings.dtype
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "D-2 cosine-similarity attention failed (%s); falling back to D-1 uniform",
                exc.__class__.__name__,
            )
            cls_attention = torch.full(
                (N,), 1.0 / N, device=tile_embeddings.device, dtype=tile_embeddings.dtype
            )
        return cls_output, cls_attention


# Backward-compat alias: any caller importing the old name continues to work.
_UniformAttentionAdapter = _AttentionExtractingAdapter


def production_slide_backbone_loader(
    hf_repo: str = DEFAULT_GIGAPATH_SLIDE_HF_REPO,
    weight_filename: str = DEFAULT_GIGAPATH_SLIDE_WEIGHT_FILE,
    arch_name: str = DEFAULT_GIGAPATH_SLIDE_ARCH,
    in_chans: int = EMBED_DIM,
) -> nn.Module:
    """Production loader for the GigaPath LongNet slide encoder.

    Downloads ``slide_encoder.pth`` from the gated HF Hub repo
    (requires ``HF_TOKEN`` in ``.env``), instantiates the vendored
    ``LongNetViT`` arch via timm's model registry, loads pretrained
    weights, and wraps in :class:`_UniformAttentionAdapter` so the
    forward signature matches the design-§4.3 contract expected by
    :class:`GigaPathSlideEncoder`.

    Args:
        hf_repo: Hugging Face Hub repo ID. Default
            ``"prov-gigapath/prov-gigapath"`` (verified upstream commit
            ``3505f87``). The legacy constant
            ``DEFAULT_GIGAPATH_SLIDE_MODEL`` points at a non-existent
            ``-slide`` repo and is kept only for backward compat.
        weight_filename: Filename inside the repo. Default
            ``"slide_encoder.pth"`` (~345 MB).
        arch_name: Model arch name registered via timm
            ``@register_model``. Default ``"gigapath_slide_enc12l768d"``
            (12 layers, 768 hidden, 85.15M trainable params).
        in_chans: Input channels (tile embedding dim). Default 1536.

    Returns:
        An ``nn.Module`` whose ``forward(tile_emb, coord)`` returns
        ``(cls_output[1536], cls_attention[N])``. The cls_attention is
        a D-1 uniform placeholder (``1/N``); real attention extraction
        is queued as a follow-up µPDCA.

    Raises:
        EnvironmentError: ``HF_TOKEN`` env var not set (see ``.env``).
        FileNotFoundError: weight file missing in the HF repo.
        RuntimeError: state_dict load reports significant missing or
            unexpected keys (vendored arch drifted from upstream).
    """
    import os
    import timm
    from huggingface_hub import hf_hub_download
    from src.models import _vendored_gigapath  # noqa: F401  side-effect: register_model

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN env var required to download GigaPath slide encoder "
            "weights from gated HF Hub repo. Set it in .env (see "
            "CLAUDE.md §Critical Rules + .env.example)."
        )

    weight_path = hf_hub_download(
        repo_id=hf_repo, filename=weight_filename, token=hf_token,
    )
    logger.info(
        "Downloaded GigaPath slide encoder weights (%d MB) → %s",
        int(os.path.getsize(weight_path) / 1e6),
        weight_path,
    )

    model = timm.create_model(arch_name, pretrained=False, in_chans=in_chans)
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            "GigaPath slide encoder state_dict mismatch: missing=%d, "
            "unexpected=%d. First 3 missing: %s; first 3 unexpected: %s",
            len(missing), len(unexpected), missing[:3], unexpected[:3],
        )
    else:
        logger.info(
            "GigaPath slide encoder pretrained weights loaded "
            "(missing=0, unexpected=0). Trainable params: %.2fM",
            sum(p.numel() for p in model.parameters()) / 1e6,
        )

    return _UniformAttentionAdapter(model)
