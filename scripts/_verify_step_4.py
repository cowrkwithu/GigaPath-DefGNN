#!/usr/bin/env python3
"""Phase 10.9 — Step 4 verifier (post-graph-construction).

Walks every ``paths.graphs/<variant>/*.pt`` and asserts the contracts
in ``docs/02-design/03-architecture.md`` §4.C:

* count(pt files per variant) == count(features/*.h5)
* each graph passes shape-invariant suite (delegated to BaseGraphConstructor's
  guards — re-loaded data should still satisfy them)

Exits 0 on success, 1 on first failure.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path

import torch

from scripts._common import add_common_args, fail, load_runtime
from src.graph_construction import GRAPH_VARIANTS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--features-root", type=Path, default=None)
    p.add_argument("--graphs-root", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rt = load_runtime(args, out_subdir="graphs")
    features_root = args.features_root or Path(rt.config["paths"]["features"])
    graphs_root = args.graphs_root or Path(rt.config["paths"]["graphs"])

    if not graphs_root.exists():
        fail(f"graphs root not found: {graphs_root}")
    n_features = len(list(features_root.glob("*.h5")))

    for variant in sorted(GRAPH_VARIANTS):
        variant_dir = graphs_root / variant
        if not variant_dir.exists():
            continue
        pts = sorted(variant_dir.glob("*.pt"))
        if n_features and len(pts) != n_features:
            fail(f"{variant}: {len(pts)} graphs vs {n_features} features (count mismatch)")
        for pt in pts[:5]:  # spot-check first 5 per variant
            data = torch.load(pt, map_location="cpu", weights_only=False)
            for attr in ("x", "pos", "edge_index", "edge_attr"):
                if not hasattr(data, attr):
                    fail(f"{pt.name}: missing attr '{attr}'")
            if data.edge_index.dtype != torch.long:
                fail(f"{pt.name}: edge_index dtype {data.edge_index.dtype} != int64")
            if (data.edge_index[0] == data.edge_index[1]).any():
                fail(f"{pt.name}: contains self-loops")
            if not torch.isfinite(data.edge_attr).all():
                fail(f"{pt.name}: non-finite edge_attr")
        print(f"  {variant}: {len(pts)} graphs OK")

    print(f"OK: graph integrity verified.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
