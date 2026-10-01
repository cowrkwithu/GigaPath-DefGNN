"""TransMIL baseline — Transformer-based correlated MIL.

Reference:
    Shao, Bian, Chen, Wang, Zhang, Ji, Zhang. "TransMIL: Transformer
    based Correlated Multiple Instance Learning for Whole Slide Image
    Classification." NeurIPS 2021.
    https://arxiv.org/abs/2106.00908
    Reference implementation: https://github.com/szc19990412/TransMIL

Simplified port (faithful to the bag-classifier path; we omit the
PPEG — Pyramid Position Encoding Generator — because the upstream
:class:`src.graph_construction` already encodes spatial structure and
the published PPEG depends on a 2D tile-grid layout that VetGigaGraph
does not enforce). The transformer encoder + CLS-token aggregation +
linear head are kept verbatim:

    proj         : Linear(embed_dim → hidden_dim)
    cls_token    : learnable [1, 1, hidden_dim]
    transformer  : 2 × TransformerEncoderLayer(d_model=hidden, heads=8)
    classifier   : Linear(hidden_dim → num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransMIL(nn.Module):
    """Transformer-based MIL (Shao et al., 2021)."""

    def __init__(
        self,
        *,
        embed_dim: int = 1536,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 7,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        self.proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        if tile_embeddings.ndim != 2 or tile_embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"TransMIL expects [N, {self.embed_dim}]; got {tuple(tile_embeddings.shape)}"
            )
        h = self.proj(tile_embeddings).unsqueeze(0)         # [1, N, hidden]
        cls = self.cls_token.expand(1, -1, -1)              # [1, 1, hidden]
        seq = torch.cat([cls, h], dim=1)                    # [1, N+1, hidden]
        out = self.transformer(seq)                          # [1, N+1, hidden]
        cls_out = out[:, 0]                                  # [1, hidden]
        return self.classifier(cls_out).squeeze(0)           # [num_classes]
