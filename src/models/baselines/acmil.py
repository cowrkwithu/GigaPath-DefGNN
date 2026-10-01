"""ACMIL baseline — Attention-Challenging MIL (Zhang et al., ECCV 2024).

Reference:
    Zhang, Y., et al. "Attention-Challenging Multiple Instance Learning for
    Whole Slide Image Classification." ECCV 2024.
    Reference implementation: https://github.com/dazhangyu123/ACMIL
    (``architecture/transformer.py::ACMIL_GA`` and the loss in
    ``Step3_WSI_classification_ACMIL.py``; ported at commit e53d19c).

Architecture (gated-attention variant, ``ACMIL_GA``):

    feat     : Linear(embed_dim → hidden_dim, no bias) → ReLU
    A        : gated attention with ``n_token`` branches          [n_token, N]
    STKIM    : in training, randomly mask ``mask_drop`` of each branch's
               top-``n_masked_patch`` instances (A → -1e9)
    branch_c : one Linear(hidden_dim → C) per branch on softmax(A_k)·h
    slide    : Linear(hidden_dim → C) on mean_k softmax(A_k)·h     [C]

``forward`` returns the slide logits, so evaluation is identical to the
other MIL baselines. The multi-branch terms of the official objective are
exposed through :meth:`auxiliary_loss`, which the Lightning module adds to
the slide loss during training:

    L = L_slide + CE(branch logits, y) + mean pairwise cos(softmax A_i, softmax A_j)

Defaults follow the official README (n_token=5, n_masked_patch=10,
mask_drop=0.6, attention dim 128).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACMIL(nn.Module):
    """Multi-branch gated attention MIL with stochastic top-K masking."""

    def __init__(
        self,
        *,
        embed_dim: int = 1536,
        hidden_dim: int = 256,
        attn_dim: int = 128,
        num_classes: int = 7,
        n_token: int = 5,
        n_masked_patch: int = 10,
        mask_drop: float = 0.6,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n_token = int(n_token)
        self.n_masked_patch = int(n_masked_patch)
        self.mask_drop = float(mask_drop)

        self.feat = nn.Sequential(nn.Linear(self.embed_dim, hidden_dim, bias=False), nn.ReLU())
        self.attn_V = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, self.n_token)
        self.branch_classifiers = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes) for _ in range(self.n_token)
        )
        self.slide_classifier = nn.Linear(hidden_dim, num_classes)
        self._branch_logits: torch.Tensor | None = None
        self._attn: torch.Tensor | None = None

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        if tile_embeddings.ndim != 2 or tile_embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"ACMIL expects [N, {self.embed_dim}]; got {tuple(tile_embeddings.shape)}"
            )
        h = self.feat(tile_embeddings)                                   # [N, hidden]
        A = self.attn_w(self.attn_V(h) * self.attn_U(h)).transpose(0, 1)  # [K, N]

        if self.training and self.n_masked_patch > 0:
            k, n = A.shape
            n_masked = min(self.n_masked_patch, n)
            _, top = torch.topk(A, n_masked, dim=-1)
            pick = torch.argsort(torch.rand(top.shape, device=A.device), dim=-1)
            pick = pick[:, : int(n_masked * self.mask_drop)]
            masked = top[torch.arange(k, device=A.device).unsqueeze(-1), pick]
            keep = torch.ones_like(A, dtype=torch.bool).scatter_(-1, masked, False)
            A = A.masked_fill(~keep, -1e9 if A.dtype == torch.float32 else -1e4)

        A_soft = F.softmax(A.float(), dim=1).to(h.dtype)                 # [K, N]
        branch_feat = A_soft @ h                                          # [K, hidden]
        self._branch_logits = torch.stack(
            [clf(branch_feat[i]) for i, clf in enumerate(self.branch_classifiers)]
        )                                                                 # [K, C]
        self._attn = A
        bag = A_soft.mean(dim=0) @ h                                      # [hidden]
        return self.slide_classifier(bag)

    def auxiliary_loss(self, target: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        """Branch CE + attention-diversity terms of the official objective."""
        assert self._branch_logits is not None and self._attn is not None
        loss = self._branch_logits.new_zeros(())
        if self.n_token > 1:
            loss = loss + loss_fn(self._branch_logits, target.reshape(-1).repeat(self.n_token))
            attn = F.softmax(self._attn.float(), dim=-1)
            n_pairs = self.n_token * (self.n_token - 1) / 2
            for i in range(self.n_token):
                for j in range(i + 1, self.n_token):
                    loss = loss + F.cosine_similarity(attn[i], attn[j], dim=-1) / n_pairs
        return loss
