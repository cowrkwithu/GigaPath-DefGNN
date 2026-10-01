#!/usr/bin/env python3
"""Phase 10.9 — Step 3 verifier (post-feature-extraction).

Walks ``paths.features`` and asserts the contracts in
``docs/02-design/03-architecture.md`` §3.B:

* every HDF5 has the locked schema (embeddings/coordinates/tile_indices/metadata)
* embedding dim = 1536, dtype float32
* no NaN/Inf
* mean L2 norm ∈ [0.5, 50]

Exits 0 on success, 1 on first failure.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from scripts._common import add_common_args, fail, load_runtime


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--features-root", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rt = load_runtime(args, out_subdir="features")
    features_root = args.features_root or Path(rt.config["paths"]["features"])
    expected_dim = int(rt.config["feature_extraction"]["embedding_dim"])

    if not features_root.exists():
        fail(f"features root not found: {features_root}")
    h5_files = sorted(features_root.glob("*.h5"))
    if not h5_files:
        fail(f"no .h5 files under {features_root}")

    for h5 in h5_files:
        with h5py.File(h5, "r") as f:
            for k in ("embeddings", "coordinates", "tile_indices", "metadata"):
                if k not in f:
                    fail(f"{h5.name}: missing dataset '{k}'")
            emb = f["embeddings"][...]
            if emb.ndim != 2 or emb.shape[1] != expected_dim:
                fail(f"{h5.name}: embeddings shape {emb.shape} != [N, {expected_dim}]")
            if emb.dtype != np.float32:
                fail(f"{h5.name}: embeddings dtype {emb.dtype} != float32")
            if not np.isfinite(emb).all():
                fail(f"{h5.name}: non-finite values in embeddings")
            mean_norm = float(np.linalg.norm(emb, axis=1).mean())
            if not 0.5 < mean_norm < 50.0:
                fail(f"{h5.name}: mean L2 norm {mean_norm:.3f} outside [0.5, 50]")

    print(f"OK: {len(h5_files)} HDF5 files pass Step 3 verification.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
