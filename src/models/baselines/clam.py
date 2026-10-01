"""CLAM-SB and CLAM-MB baselines — clustering-constrained attention MIL.

Reference:
    Lu, Williamson, Chen, Chen, Barbieri, Mahmood. "Data Efficient and
    Weakly Supervised Computational Pathology on Whole Slide Images."
    Nature Biomedical Engineering, 2021.
    https://arxiv.org/abs/2004.09666
    Reference implementation: https://github.com/mahmoodlab/CLAM

Two variants:

* **CLAM-SB** — single attention branch, single classifier. Identical
  in spirit to ABMIL but with the published projection sizes.
* **CLAM-MB** — multi-branch: one attention branch per class, plus a
  dedicated linear head per class operating on the corresponding
  bag-feature. Returns ``[num_classes]`` raw logits; the per-class
  scores naturally support multi-label settings the original paper
  benchmarks against.

Both variants are simplified ports — we omit the optional instance-
level clustering loss (introduced as a regularizer in the CLAM paper)
because the design's training spec uses a single class-weighted
CrossEntropy loss across all baselines for fair comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLAM_SB(nn.Module):
    """Single-branch CLAM (Lu et al., 2021)."""

    def __init__(
        self,
        *,
        embed_dim: int = 1536,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        self.feat = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Sequential(
            nn.Linear(self.hidden_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attn_U = nn.Sequential(
            nn.Linear(self.hidden_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        if tile_embeddings.ndim != 2 or tile_embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"CLAM_SB expects [N, {self.embed_dim}]; got {tuple(tile_embeddings.shape)}"
            )
        h = self.feat(tile_embeddings)
        a = self.attn_w(self.attn_V(h) * self.attn_U(h))  # [N, 1]
        a = F.softmax(a, dim=0)
        bag = (a * h).sum(dim=0)
        return self.classifier(bag)


class CLAM_MB(nn.Module):
    """Multi-branch CLAM (Lu et al., 2021).

    One attention branch per class, one linear head per class. The
    final logits are a length-``num_classes`` vector composed of each
    class's own bag score.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 1536,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        self.feat = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Sequential(
            nn.Linear(self.hidden_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout)
        )
        self.attn_U = nn.Sequential(
            nn.Linear(self.hidden_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )
        # One scalar gate per class.
        self.attn_w = nn.Linear(attn_dim, self.num_classes)
        # One classifier head per class (each head sees its own bag rep).
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(self.num_classes)]
        )

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        if tile_embeddings.ndim != 2 or tile_embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"CLAM_MB expects [N, {self.embed_dim}]; got {tuple(tile_embeddings.shape)}"
            )
        h = self.feat(tile_embeddings)                          # [N, hidden]
        a = self.attn_w(self.attn_V(h) * self.attn_U(h))         # [N, C]
        a = F.softmax(a, dim=0)                                  # [N, C]
        # Per-class bag rep = weighted sum over N (one weight per class).
        # Shape calc: a [N, C, 1] * h [N, 1, hidden] → sum_N → [C, hidden].
        bag_per_class = (a.unsqueeze(-1) * h.unsqueeze(1)).sum(dim=0)  # [C, hidden]
        # Per-class scalar logit.
        logits = torch.stack(
            [clf(bag_per_class[c]).squeeze(-1) for c, clf in enumerate(self.classifiers)],
            dim=0,
        )
        return logits  # [C]


# --------------------------------------------------------------------------- #
# CLAM with the instance-level clustering loss (1st revision)
# --------------------------------------------------------------------------- #


class _InstanceClusteringMixin:
    """Instance-level clustering loss of the official CLAM (``inst_eval`` /
    ``inst_eval_out`` in ``models/model_clam.py``, mahmoodlab/CLAM @ 53e2409),
    in the subtyping setting used for multi-class tumor subtyping:

    * true-class branch: the ``k_sample`` highest-attention tiles are positives
      and the ``k_sample`` lowest are negatives for that class's instance head;
    * every other class's branch: its ``k_sample`` highest-attention tiles
      are negatives;
    * the summed instance CE is divided by the number of classes.

    Training objective (official defaults): 0.7 · slide CE + 0.3 · instance CE,
    via ``bag_loss_weight`` and :meth:`auxiliary_loss`.
    """

    k_sample: int = 8
    bag_loss_weight: float = 0.7

    def _init_instance_heads(self) -> None:
        self.instance_classifiers = nn.ModuleList(
            nn.Linear(self.hidden_dim, 2) for _ in range(self.num_classes)
        )
        self._h: torch.Tensor | None = None
        self._A: torch.Tensor | None = None  # [branches, N], softmax over N

    def _instance_loss(self, A: torch.Tensor, h: torch.Tensor, clf: nn.Module,
                       in_class: bool) -> torch.Tensor:
        k = min(self.k_sample, A.shape[0])
        top = torch.topk(A, k).indices
        if not in_class:
            logits = clf(h[top])
            return F.cross_entropy(logits.float(), torch.zeros(k, dtype=torch.long, device=h.device))
        bottom = torch.topk(-A, k).indices
        logits = clf(torch.cat([h[top], h[bottom]]))
        targets = torch.cat([torch.ones(k), torch.zeros(k)]).long().to(h.device)
        return F.cross_entropy(logits.float(), targets)

    def auxiliary_loss(self, target: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        assert self._h is not None and self._A is not None
        label = int(target.reshape(-1)[0])
        total = self._h.new_zeros((), dtype=torch.float32)
        for c, clf in enumerate(self.instance_classifiers):
            A = self._A[c] if self._A.shape[0] > 1 else self._A[0]
            total = total + self._instance_loss(A, self._h, clf, in_class=(c == label))
        return (1.0 - self.bag_loss_weight) * total / self.num_classes


class CLAM_SB_Inst(_InstanceClusteringMixin, CLAM_SB):
    """CLAM-SB trained with the instance-level clustering loss."""

    def __init__(self, **kwargs) -> None:
        CLAM_SB.__init__(self, **kwargs)
        self._init_instance_heads()

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        h = self.feat(tile_embeddings)
        a = F.softmax(self.attn_w(self.attn_V(h) * self.attn_U(h)), dim=0)  # [N, 1]
        self._h, self._A = h, a.transpose(0, 1)
        return self.classifier((a * h).sum(dim=0))


class CLAM_MB_Inst(_InstanceClusteringMixin, CLAM_MB):
    """CLAM-MB trained with the instance-level clustering loss."""

    def __init__(self, **kwargs) -> None:
        CLAM_MB.__init__(self, **kwargs)
        self._init_instance_heads()

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        h = self.feat(tile_embeddings)
        a = F.softmax(self.attn_w(self.attn_V(h) * self.attn_U(h)), dim=0)  # [N, C]
        self._h, self._A = h, a.transpose(0, 1)
        bag_per_class = (a.unsqueeze(-1) * h.unsqueeze(1)).sum(dim=0)
        return torch.stack(
            [clf(bag_per_class[c]).squeeze(-1) for c, clf in enumerate(self.classifiers)], dim=0
        )
