"""MLP classifier head (Phase 5.4).

Locked architecture (per `configs/default.yaml model.classifier` and
`docs/02-design/03-architecture.md` §5):

    Linear(input_dim → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → num_classes)

Returns **raw logits** (no softmax) — the design's
``test_no_softmax_in_logits`` and downstream class-weighted CrossEntropy
loss both depend on this.

References:
    Design: docs/02-design/03-architecture.md §5 (classifier row)
    Tests:  docs/02-design/03-architecture.md §5.D row 3 (no_softmax_in_logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MlpClassifier(nn.Module):
    """2-layer MLP classifier producing raw logits.

    Args:
        input_dim: Input feature width (default 256, matches GNN output
            and slide-encoder projection).
        hidden_dim: Hidden width (default 128 per locked config).
        num_classes: Output classes (default 7 — locked).
        dropout: Dropout between the two linear layers (default 0.25).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"MlpClassifier expects input_dim={self.input_dim}; "
                f"got {tuple(x.shape)}"
            )
        return self.net(x)
