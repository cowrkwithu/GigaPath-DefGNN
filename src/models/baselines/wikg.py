"""WiKG baseline — dynamic knowledge-graph MIL (Li et al., CVPR 2024).

Reference:
    Li, J., et al. "Dynamic Graph Representation with Knowledge-aware
    Attention for Histopathology Whole Slide Image Analysis." CVPR 2024.
    Reference implementation: https://github.com/WonderLandxD/WiKG
    (``model.py::WiKG``; ported at commit eb3144f).

Each tile is a node; its neighbours are the top-k tiles by learned
head/tail similarity (e_h · e_t / sqrt(d)), rebuilt on every forward pass,
so the graph is learned rather than fixed. Knowledge-aware gated attention
aggregates the neighbours, a bi-interaction layer fuses node and neighbour
embeddings, and a global attention readout produces the slide vector.

Port change (memory only, same math): the official code materialises the
full [N, N] similarity matrix before ``topk``, which needs ~35 GB at the
largest CATCH slide (~94k tiles). Here the top-k search runs over query
chunks of ``chunk_size`` rows, giving identical neighbours and weights.

Defaults follow the official code (topk=6, bi-interaction, dropout=0.3,
attention readout); ``hidden_dim`` is set by the registry to the common MIL
width used for every baseline in this study.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WiKG(nn.Module):
    """Dynamic top-k knowledge graph with gated knowledge-aware attention."""

    def __init__(
        self,
        *,
        embed_dim: int = 1536,
        hidden_dim: int = 256,
        num_classes: int = 7,
        topk: int = 6,
        dropout: float = 0.3,
        chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.topk = int(topk)
        self.chunk_size = int(chunk_size)
        self.scale = hidden_dim ** -0.5

        self.fc1 = nn.Sequential(nn.Linear(self.embed_dim, hidden_dim), nn.LeakyReLU())
        self.W_head = nn.Linear(hidden_dim, hidden_dim)
        self.W_tail = nn.Linear(hidden_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)
        self.readout_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def _topk_neighbours(self, e_h: torch.Tensor, e_t: torch.Tensor):
        """Row-chunked equivalent of ``topk((e_h*scale) @ e_t.T, k)``."""
        k = min(self.topk, e_t.shape[0])
        weights, index = [], []
        for start in range(0, e_h.shape[0], self.chunk_size):
            logit = (e_h[start:start + self.chunk_size] * self.scale) @ e_t.transpose(0, 1)
            w, i = torch.topk(logit, k=k, dim=-1)
            weights.append(w)
            index.append(i)
        return torch.cat(weights), torch.cat(index)

    def forward(self, tile_embeddings: torch.Tensor) -> torch.Tensor:
        if tile_embeddings.ndim != 2 or tile_embeddings.shape[1] != self.embed_dim:
            raise ValueError(
                f"WiKG expects [N, {self.embed_dim}]; got {tuple(tile_embeddings.shape)}"
            )
        x = self.fc1(tile_embeddings)                          # [N, C]
        x = (x + x.mean(dim=0, keepdim=True)) * 0.5

        e_h = self.W_head(x)
        e_t = self.W_tail(x)
        topk_weight, topk_index = self._topk_neighbours(e_h, e_t)   # [N, k]
        Nb_h = e_t[topk_index]                                  # [N, k, C]

        topk_prob = F.softmax(topk_weight.float(), dim=1).to(e_h.dtype)
        eh_r = topk_prob.unsqueeze(-1) * Nb_h + (1 - topk_prob).unsqueeze(-1) * e_h.unsqueeze(1)

        gate = torch.tanh(e_h.unsqueeze(1) + eh_r)              # [N, k, C]
        # Official einsum('ijkl,ijkm->ijk') sums the two operands over separate
        # indices, i.e. (Σ Nb_h)·(Σ gate) rather than a dot product; kept as is.
        ka_weight = Nb_h.sum(dim=-1) * gate.sum(dim=-1)         # [N, k]
        ka_prob = F.softmax(ka_weight.float(), dim=1).to(e_h.dtype)
        e_Nh = (ka_prob.unsqueeze(-1) * Nb_h).sum(dim=1)        # [N, C]

        embedding = self.activation(self.linear1(e_h + e_Nh)) + \
            self.activation(self.linear2(e_h * e_Nh))
        h = self.message_dropout(embedding)

        gate_logit = self.readout_gate(h)                       # [N, 1]
        a = F.softmax(gate_logit.float(), dim=0).to(h.dtype)
        slide = (a * h).sum(dim=0)
        return self.fc(self.norm(slide))
