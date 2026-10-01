"""Single deformable-attention layer for VetGigaGraph v2.

Adapts Fu et al. 2025 to irregular WSI tile graphs. Each node:
    1. Predicts K coordinate offsets from its (h, x, y).
    2. Soft-KNN-samples features at the offset positions.
    3. Runs multi-head attention over (static graph neighbors ∪ K sampled feats).
    4. Residual + LayerNorm, à la GAT.

Designed to slot into a GnnBackbone-compatible stack so the v1 VetGigaGraph
training/eval pipeline (under `src/shared/`) can drive it unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import scatter, softmax as pyg_softmax

from .kernel import coord_knn_sample


class _OffsetMLP(nn.Module):
    """Predicts K 2-D offsets per node from (feature, coord)."""

    def __init__(
        self,
        in_dim: int,
        num_offsets: int,
        hidden_dim: int = 64,
        depth: int = 1,
        init_scale: float = 0.05,
        fp32: bool = True,
    ) -> None:
        super().__init__()
        self.num_offsets = int(num_offsets)
        self.fp32 = bool(fp32)
        layers: list[nn.Module] = []
        d_in = in_dim + 2                  # concat with (x, y)
        d_h = int(hidden_dim)
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(d_in, d_h), nn.ReLU()]
            d_in = d_h
        layers += [nn.Linear(d_in, num_offsets * 2)]
        self.net = nn.Sequential(*layers)
        # Initialize the final layer toward small offsets so the model starts
        # close to a vanilla attention layer (zero-offset = no deformation).
        with torch.no_grad():
            last = self.net[-1]
            assert isinstance(last, nn.Linear)
            last.weight.mul_(0.01)
            last.bias.zero_()
            # Add tiny random offsets so K samples don't collapse to identical points
            last.bias.add_(torch.randn_like(last.bias) * float(init_scale) * 0.1)

    def forward(self, h: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Return ``[N, K, 2]`` offset tensor."""
        if self.fp32:
            h_in = h.float()
            coord_in = coords.float()
            inp = torch.cat([h_in, coord_in], dim=-1)
            raw = self.net(inp)
        else:
            inp = torch.cat([h, coords.to(h.dtype)], dim=-1)
            raw = self.net(inp)
        return raw.view(-1, self.num_offsets, 2).to(h.dtype)


class DeformableAttentionLayer(nn.Module):
    """One deformable-attention message-passing block.

    Forward signature mirrors what ``GnnBackbone`` calls each layer with,
    plus the extra ``coords`` argument carried alongside ``x``.

    Args:
        in_dim:          Node-feature width on input.
        out_dim:         Output width (matches ``hidden_dim`` for non-last layers,
                         ``output_dim`` for the last).
        heads:           Number of attention heads.
        num_offsets:     K — sampled positions per node (Fu 2025).
        offset_mlp_hidden: Hidden width of the offset MLP.
        offset_mlp_depth:  Depth of the offset MLP (1 = single linear).
        offset_init_scale: Initial magnitude of offsets (normalized coords).
        knn_k:           Number of nearest tiles used in soft-KNN sampling.
        knn_temperature: Softmax temperature on the distances.
        fp32_offset_mlp: Promote the offset MLP to fp32 for stability under fp16.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 2,
        *,
        num_offsets: int = 2,
        offset_mlp_hidden: int = 64,
        offset_mlp_depth: int = 1,
        offset_init_scale: float = 0.05,
        knn_k: int = 8,
        knn_temperature: float = 1.0,
        knn_chunk_size: int = 64,
        fp32_offset_mlp: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError(
                f"out_dim={out_dim} must be divisible by heads={heads}."
            )
        self.heads = int(heads)
        self.head_dim = int(out_dim) // int(heads)
        self.num_offsets = int(num_offsets)
        self.knn_k = int(knn_k)
        self.knn_temperature = float(knn_temperature)
        self.knn_chunk_size = int(knn_chunk_size)

        # QKV projections — Q from each node, K/V from both static neighbors
        # and sampled features. We share K/V projections across both sources.
        self.q_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.kv_proj_static = nn.Linear(in_dim, 2 * out_dim, bias=False)
        self.kv_proj_sampled = nn.Linear(in_dim, 2 * out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)

        # Offset prediction
        self.offset_mlp = _OffsetMLP(
            in_dim=in_dim,
            num_offsets=num_offsets,
            hidden_dim=offset_mlp_hidden,
            depth=offset_mlp_depth,
            init_scale=offset_init_scale,
            fp32=fp32_offset_mlp,
        )

        # Residual + norm. If in_dim != out_dim, add a projection on the skip.
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = float(dropout)

    def _compute_h(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute h_out only (no attention bookkeeping). Cheap to checkpoint."""
        n_nodes = x.shape[0]
        H, D_head = self.heads, self.head_dim

        # 1. Offsets + soft-KNN sample
        offsets = self.offset_mlp(x, coords)
        query_coords = coords.unsqueeze(1) + offsets
        sampled = coord_knn_sample(
            query_coords=query_coords,
            key_coords=coords,
            key_features=x,
            knn_k=self.knn_k,
            temperature=self.knn_temperature,
            chunk_size=self.knn_chunk_size,
        )

        # 2. Q, K, V projections
        q = self.q_proj(x).view(n_nodes, H, D_head)
        kv_static = self.kv_proj_static(x).view(n_nodes, 2, H, D_head)
        k_static = kv_static[:, 0]
        v_static = kv_static[:, 1]
        kv_sampled = self.kv_proj_sampled(sampled).view(
            n_nodes, self.num_offsets, 2, H, D_head
        )
        k_sampled = kv_sampled[:, :, 0]
        v_sampled = kv_sampled[:, :, 1]

        # 3. Vectorized scatter-softmax attention over (static ∪ sampled)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        scale = D_head ** 0.5
        score_static = (q[dst_idx] * k_static[src_idx]).sum(dim=-1) / scale
        v_static_at_src = v_static[src_idx]
        score_sampled = (q.unsqueeze(1) * k_sampled).sum(dim=-1) / scale
        K = self.num_offsets
        dst_sampled = torch.arange(n_nodes, device=x.device).repeat_interleave(K)
        score_sampled_flat = score_sampled.reshape(n_nodes * K, H)
        v_sampled_flat = v_sampled.reshape(n_nodes * K, H, D_head)

        dst_all = torch.cat([dst_idx, dst_sampled], dim=0)
        score_all = torch.cat([score_static, score_sampled_flat], dim=0)
        v_all = torch.cat([v_static_at_src, v_sampled_flat], dim=0)

        attn = pyg_softmax(score_all, dst_all, num_nodes=n_nodes, dim=0)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        weighted = attn.unsqueeze(-1) * v_all
        h_attn = scatter(weighted, dst_all, dim=0, dim_size=n_nodes, reduce="sum")
        h_attn = h_attn.reshape(n_nodes, H * D_head)
        h_out = self.out_proj(h_attn)
        h_out = self.norm(h_out + self.skip(x))
        return h_out

    def forward(
        self,
        x: torch.Tensor,                       # [N, in_dim]
        coords: torch.Tensor,                  # [N, 2]
        edge_index: torch.Tensor,              # [2, E]  static dual-edge graph
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns ``(h_out, attentions)``.

        Memory note: training-time forward is wrapped in `checkpoint` to
        recompute the O(N·K) sampling + scatter-softmax intermediates during
        backward instead of retaining them. Without this, 30k-tile WSIs OOM
        on a 24 GiB GPU.
        """
        if self.training and x.requires_grad:
            h_out = checkpoint(self._compute_h, x, coords, edge_index,
                               use_reentrant=False)
            # During training we don't pay the cost of cloning attention/offset
            # tensors — they're only useful for post-hoc analysis.
            attentions: dict[str, torch.Tensor] = {}
        else:
            # Eval path: compute h and detach attention/offsets for logging.
            with torch.no_grad():
                offsets = self.offset_mlp(x, coords)
            h_out = self._compute_h(x, coords, edge_index)
            attentions = {
                "deformable.offsets": offsets.detach().to("cpu"),
            }
        return h_out, attentions
