"""Feature-similarity graph (variant 2 of 5).

Edges link tiles whose embedding cosine similarity exceeds ``tau``
(default ``0.8`` per the locked default in
``docs/02-design/04-experiment-design.md``). Self-loops are dropped.
The edge weight is the cosine similarity itself.

When ``tau`` is set too high (e.g. ``0.999``), the graph can have zero
edges; :class:`BaseGraphConstructor.build_graph` will then raise
:class:`GraphDisconnectedError` so the caller can lower the threshold.

Implementation note: the similarity matrix is computed in **row chunks**
rather than as a single ``[N, N]`` matmul. CATCH WSIs can reach
N≈94k tiles where the full float32 matrix would be ~35 GB and OOM
even on a 24 GB GPU; ``chunk=2048`` keeps peak GPU memory bounded
at a few hundred MB per chunk.

**D-19 fix (2026-05-20)**: dense slides (N≈94k tiles, ~5% sim>tau density)
were producing ~440M edges per slide, exceeding 24 GiB at ``torch.cat`` time.
The new ``max_edges_per_node`` parameter caps each row's outgoing edges at
top-K by cosine similarity (default ``None`` = no cap, preserves legacy
behavior). Recommended ``max_edges_per_node=64`` for CATCH-scale data —
sparse slides remain unaffected (their natural edge count is far below K),
while dense slides are bounded so .pt files stay tractable.

References:
    Design: docs/02-design/03-architecture.md §4 (Graph 2)
    Tests:  docs/02-design/03-architecture.md §4.C (Feature-similarity rows)
    Drift:  D-19 in docs/03-analysis/vetgigagraph.analysis.md v0.9
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.graph_construction.base_graph import (
    DEFAULT_CHUNK_SIZE,
    BaseGraphConstructor,
    DeviceLike,
)


class FeatureSimGraph(BaseGraphConstructor):
    """Cosine-similarity-thresholded graph with optional top-K-per-node cap.

    Parameters
    ----------
    tau:
        Cosine similarity threshold. Edge (i, j) added only if sim(i, j) >= tau.
        Locked default is ``0.8`` per planning doc §3.2.3.
    max_edges_per_node:
        If set, each row keeps only its top-K incident edges (by similarity)
        from the thresholded set. ``None`` (default) preserves legacy unbounded
        behavior. Recommended K=64 for CATCH-scale data to avoid OOM on dense
        slides where N > 50k tiles can produce 100M+ edges (see D-19).
    device:
        Build device (``"auto"`` / ``"cuda"`` / ``"cpu"``).
    chunk_size:
        Row chunk size for the [N, N] cosine matrix (memory cap).
    """

    def __init__(
        self,
        tau: float = 0.8,
        *,
        max_edges_per_node: int | None = None,
        device: DeviceLike = "auto",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        if not -1.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [-1, 1]; got {tau}")
        if max_edges_per_node is not None and max_edges_per_node < 1:
            raise ValueError(
                f"max_edges_per_node must be a positive int or None; got {max_edges_per_node}"
            )
        self.tau = float(tau)
        self.max_edges_per_node = (
            int(max_edges_per_node) if max_edges_per_node is not None else None
        )
        self.chunk_size = int(chunk_size)
        self.device = device

    def get_config(self) -> dict[str, float | int | None]:
        # chunk_size is an execution detail (same role as device); excluded
        # from the round-trip config so saved graphs don't bake host tuning
        # into the spec. max_edges_per_node IS part of the spec (it changes
        # the resulting edge set on dense slides), so it IS round-tripped.
        return {"tau": self.tau, "max_edges_per_node": self.max_edges_per_node}

    def _build_edges(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        n = x.shape[0]
        x_n = F.normalize(x, dim=-1)

        u_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []
        attr_chunks: list[torch.Tensor] = []

        K = self.max_edges_per_node

        for i in range(0, n, self.chunk_size):
            j = min(i + self.chunk_size, n)
            # [B, N] cosine similarity of rows [i:j] against all tiles.
            sim = x_n[i:j] @ x_n.t()

            rows = torch.arange(i, j, device=x.device).unsqueeze(-1)
            cols = torch.arange(n, device=x.device).unsqueeze(0)
            # Drop self-similarities (always == 1) before thresholding.
            sim.masked_fill_(rows == cols, float("-inf"))

            if K is None:
                # Legacy path — return all (i, j) with sim >= tau.
                mask = sim >= self.tau
                local_u, local_v = mask.nonzero(as_tuple=True)
                if local_u.numel() == 0:
                    continue
                u_chunks.append(local_u + i)
                v_chunks.append(local_v)
                attr_chunks.append(sim[local_u, local_v])
            else:
                # D-19 fix: top-K per row, then apply tau threshold so a sparse
                # row (fewer than K neighbours above tau) still respects the
                # threshold instead of being padded to K with low-sim edges.
                k_eff = min(K, n - 1)
                topk_vals, topk_idx = sim.topk(k_eff, dim=-1)
                tau_mask = topk_vals >= self.tau
                # Convert to (row_in_chunk, col) coordinates of surviving entries.
                local_rows, local_pos = tau_mask.nonzero(as_tuple=True)
                if local_rows.numel() == 0:
                    continue
                local_u = local_rows + i
                local_v = topk_idx[local_rows, local_pos]
                local_attr = topk_vals[local_rows, local_pos]
                u_chunks.append(local_u)
                v_chunks.append(local_v)
                attr_chunks.append(local_attr)

        if not u_chunks:
            # Returning a zero-edge dict lets BaseGraphConstructor surface
            # the standard GraphDisconnectedError with the right message.
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            edge_attr = torch.empty((0, 1), dtype=torch.float32, device=x.device)
            return {"edge_index": edge_index, "edge_attr": edge_attr}

        u = torch.cat(u_chunks)
        v = torch.cat(v_chunks)
        # The cosine matrix is symmetric. Without top-K, that means the edge
        # set is already bidirectional. WITH top-K, row i's top-K neighbours
        # need not include row j even if j's top-K includes i, so the set
        # may be asymmetric. We symmetrize explicitly so downstream GAT layers
        # see consistent neighbourhoods on both endpoints.
        if K is not None:
            edge_index = torch.stack([u, v], dim=0).to(torch.long)
            edge_attr = torch.cat(attr_chunks).unsqueeze(-1)
            # Add reverse edges; dedupe via unique on packed key.
            rev_u = v
            rev_v = u
            all_u = torch.cat([u, rev_u])
            all_v = torch.cat([v, rev_v])
            all_attr = torch.cat([edge_attr.squeeze(-1), edge_attr.squeeze(-1)])
            # Dedupe by packing (u, v) into a single int64 key.
            key = all_u.to(torch.long) * n + all_v.to(torch.long)
            unique_key, inverse = torch.unique(key, return_inverse=True)
            uniq_u = unique_key // n
            uniq_v = unique_key % n
            # When the same (u, v) pair appears twice, prefer the max sim
            # (they should match anyway since cosine is symmetric, but float
            # precision can disagree on the last bit).
            uniq_attr = torch.full(
                (unique_key.numel(),), float("-inf"), device=x.device, dtype=all_attr.dtype
            )
            uniq_attr = uniq_attr.scatter_reduce(
                0, inverse, all_attr, reduce="amax", include_self=True
            )
            edge_index = torch.stack([uniq_u, uniq_v], dim=0).to(torch.long)
            edge_attr = uniq_attr.unsqueeze(-1)
        else:
            edge_index = torch.stack([u, v], dim=0).to(torch.long)
            edge_attr = torch.cat(attr_chunks).unsqueeze(-1)
        return {"edge_index": edge_index, "edge_attr": edge_attr}
