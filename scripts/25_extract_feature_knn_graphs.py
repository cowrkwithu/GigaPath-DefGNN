#!/usr/bin/env python3
"""Feature-only k-NN graphs (k = 5) for the graph-construction ablation (R1-4).

The dual-edge graphs store an edge_type per edge (0 = spatial k = 8,
1 = feature k = 5 cosine k-NN, both symmetrized). Keeping only the
edge_type == 1 edges gives exactly the feature branch of the dual-edge graph,
with identical neighbours and attributes, so no k-NN is recomputed.

Run:  python3 scripts/25_extract_feature_knn_graphs.py
Out:  /data/cia_outputs/graphs/feature_knn/<slide_id>.pt (skips existing)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import output_dir  # noqa: E402
from src.utils.io_utils import save_pyg_data  # noqa: E402  (atomic write)

SRC = output_dir() / "graphs/dual_edge"
DST = output_dir() / "graphs/feature_knn"
FEATURE_TYPE = 1


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.pt"))
    done = 0
    for f in files:
        out = DST / f.name
        if out.exists():
            continue
        g = torch.load(f, map_location="cpu", weights_only=False)
        keep = g.edge_type == FEATURE_TYPE
        g.edge_index = g.edge_index[:, keep].contiguous()
        g.edge_attr = g.edge_attr[keep].contiguous()
        g.edge_type = g.edge_type[keep].contiguous()
        save_pyg_data(g, out)
        done += 1
        if done % 25 == 0:
            print(f"{done} written", flush=True)
    print(f"done: {done} written, {len(list(DST.glob('*.pt')))} / {len(files)} present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
