"""Feature-space adapter (µPDCA #10 D-18 saturation falsification test).

A small trainable MLP placed between the frozen GigaPath tile encoder
output and the GNN input. Lets the slide-classification task warp the
1536-d feature space without touching the (frozen) tile encoder.

Architecture (residual two-layer MLP):
    x [N, 1536] → Linear(1536, hidden) → GELU → Dropout
              → Linear(hidden, 1536) → +x (residual) → output [N, 1536]

The residual connection means ``adapter(x) ≈ x`` at initialization (the
second linear is zero-init), so a model with the adapter inserted reduces
to the no-adapter baseline at epoch 0. Whether the adapter learns a
useful warp is the experimental question.

Memory cost: ~6M trainable params at hidden=512 (vs the 1.13B frozen tile
encoder). Falls comfortably under the 24 GiB D-18 budget.

Per CLAUDE.md §11 ("GigaPath의 수의 조직 일반화 실패 → Domain-adaptive
fine-tuning 또는 tile encoder 일부 레이어 학습"), this is the
domain-adaptation-on-features variant of that recommendation.

References:
    Plan: inline in µPDCA #10 (small surface, no separate plan doc)
    Test: docs/04-report/upd9-heterogat-d23-fix.report.md §4.3
        ("Three plausible mechanisms" — adapter tests mechanism #1
         "frozen tile features are the ceiling")
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """Two-layer residual MLP that warps frozen tile features.

    Parameters
    ----------
    embed_dim:
        Feature dimension. Must match the tile encoder's output (1536 for
        GigaPath ViT-G).
    hidden_dim:
        Bottleneck width of the MLP. Default ``512`` for ~1.5M params
        (modest capacity, safe under 24 GiB).
    dropout:
        Dropout rate between the two Linear layers. Default ``0.1``.
    zero_init:
        When ``True`` (default), the second Linear is zero-initialized so
        ``adapter(x) == x`` at the start of training (residual identity).
        Lets the model fall back to the frozen-features baseline cleanly
        if no useful warp exists; the adapter "earns" its representation
        capacity through gradient descent.
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim < 1 or hidden_dim < 1:
            raise ValueError(
                f"embed_dim and hidden_dim must be positive; got "
                f"embed_dim={embed_dim}, hidden_dim={hidden_dim}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.proj_in = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, embed_dim)

        if zero_init:
            # Zero-init the output projection so adapter is initially
            # an identity (via residual). Bias also zero.
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual MLP. ``x`` shape ``[N, embed_dim]`` → same."""
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"FeatureAdapter expects last-dim {self.embed_dim}; "
                f"got {tuple(x.shape)}"
            )
        h = self.proj_in(x)
        h = self.act(h)
        h = self.drop(h)
        h = self.proj_out(h)
        return x + h  # residual connection

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}, "
            f"dropout={self.dropout}, residual=True"
        )
