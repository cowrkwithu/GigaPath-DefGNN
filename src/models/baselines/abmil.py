"""ABMIL baseline — Attention-based Deep MIL (Ilse et al., ICML 2018).

Reference:
    Ilse, Tomczak, Welling. "Attention-based Deep Multiple Instance
    Learning." ICML 2018. https://arxiv.org/abs/1802.04712
    Reference implementation: https://github.com/AMLab-Amsterdam/AttentionDeepMIL

Architecture (gated attention variant — eq. 9 in the paper):

    feat_proj : Linear(embed_dim → hidden_dim) → ReLU → Dropout
    attn_V    : Linear(hidden_dim → attn_dim) → tanh
    attn_U    : Linear(hidden_dim → attn_dim) → sigmoid
    attn_w    : Linear(attn_dim → 1)
    a         : softmax_N( attn_w(attn_V(h) ⊙ attn_U(h)) )    [N, 1]
    bag       : sum_N( a · h )                                [hidden_dim]
    logits    : Linear(hidden_dim → num_classes)              [num_classes]

Returns raw logits (no softmax) so the same class-weighted
CrossEntropy loss used for VetGigaGraph applies unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """Gated attention-based MIL (Ilse et al., 2018)."""

    def __init__(
        self,
        *,
        embed_dim: int = 1536,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.attn_dim = int(attn_dim)
        self.num_classes = int(num_classes)

        self.feat = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attn_dim), nn.Tanh()
        )
        self.attn_U = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attn_dim), nn.Sigmoid()
        )
        self.attn_w = nn.Linear(self.attn_dim, 1)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward a single bag.

        Args:
            tile_embeddings: ``[N, embed_dim]`` per-tile features.

        Returns:
            ``[num_classes]`` raw logits.
        """
        if tile_embeddings.ndim != 2 or tile_embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"ABMIL expects [N, {self.embed_dim}]; got {tuple(tile_embeddings.shape)}"
            )
        h = self.feat(tile_embeddings)  # [N, hidden]
        a = self.attn_w(self.attn_V(h) * self.attn_U(h))  # [N, 1]
        a = F.softmax(a, dim=0)
        bag = (a * h).sum(dim=0)  # [hidden]
        return self.classifier(bag)
