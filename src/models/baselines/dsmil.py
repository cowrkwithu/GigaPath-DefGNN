"""DSMIL baseline — Dual-Stream MIL with non-local attention.

Reference:
    Li, Li, Eliceiri. "Dual-stream Multiple Instance Learning Network
    for Whole Slide Image Classification with Self-supervised
    Contrastive Learning." CVPR 2021.
    https://arxiv.org/abs/2011.08939
    Reference implementation: https://github.com/binli123/dsmil-wsi

Architecture (faithful port of the published `dsmil.py`):

    feat       : Linear(embed_dim → hidden_dim) → ReLU → Dropout
    inst_clf   : Linear(hidden_dim → num_classes)         # per-tile scores
    critical   : argmax over N for each class             # [num_classes]
    q_proj     : Linear(hidden_dim → q_dim)               # queries
    v_proj     : Linear(hidden_dim → hidden_dim)          # values
    attn       : softmax( q(critical) @ q(all)ᵀ / √q_dim ) [num_classes, N]
    bag_feats  : attn @ v(all)                             [num_classes, hidden]
    bag_logits : per-class linear over bag_feats           [num_classes]

The DSMIL paper averages instance-max logits with bag-attention logits
to produce the final prediction; we follow that convention so the
return is a single ``[num_classes]`` tensor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSMIL(nn.Module):
    """Dual-stream MIL (Li et al., 2021)."""

    def __init__(
        self,
        *,
        embed_dim: int = 1536,
        hidden_dim: int = 512,
        q_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.q_dim = int(q_dim)
        self.num_classes = int(num_classes)

        self.feat = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Per-tile classifier for the "instance stream".
        self.inst_clf = nn.Linear(self.hidden_dim, self.num_classes)
        # Non-local attention projections.
        self.q_proj = nn.Linear(self.hidden_dim, self.q_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Per-class bag scoring head: each class gets its own linear over
        # the attended bag feature.
        self.bag_clf = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(self.num_classes)]
        )

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        if tile_embeddings.ndim != 2 or tile_embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"DSMIL expects [N, {self.embed_dim}]; got {tuple(tile_embeddings.shape)}"
            )
        h = self.feat(tile_embeddings)  # [N, hidden]

        # --- Instance stream: per-tile per-class scores.
        inst_logits = self.inst_clf(h)  # [N, C]
        max_inst_logits, max_inst_idx = inst_logits.max(dim=0)  # [C]

        # --- Bag stream: non-local attention from each class's critical instance.
        critical = h[max_inst_idx]  # [C, hidden]
        q_critical = self.q_proj(critical)  # [C, q_dim]
        q_all = self.q_proj(h)              # [N, q_dim]
        v_all = self.v_proj(h)              # [N, hidden]

        attn_logits = q_critical @ q_all.t() / math.sqrt(self.q_dim)  # [C, N]
        attn = F.softmax(attn_logits, dim=-1)                         # [C, N]
        bag_feats = attn @ v_all                                       # [C, hidden]

        # Per-class bag scoring.
        bag_logits = torch.stack(
            [clf(bag_feats[c]).squeeze(-1) for c, clf in enumerate(self.bag_clf)],
            dim=0,
        )  # [C]

        # DSMIL final: average the two streams.
        return 0.5 * (max_inst_logits + bag_logits)
