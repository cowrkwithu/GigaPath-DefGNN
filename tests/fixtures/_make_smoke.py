"""Smoke-fixture generator (Phase 11.1, 11.3).

Synthesizes a tiny end-to-end-runnable graph dataset + matching split
CSV in ``tmp_path``. Used by:

* ``tests/integration/test_smoke_pipeline.py``
* ``scripts/smoke_test.sh`` (when invoked with ``--generate``).

Why synthesize at test time instead of committing fixtures: real WSI
files are 100s of MB each (binary medical images), and even a
"5-WSI subset" exceeds reasonable git-repo sizes. Committing the
*output* of the pipeline (~8KB graph .pt files) is workable, but
generating them on demand keeps the repo lean and exercises the
graph-construction code path in addition to the trainer.

The synthesised data:

* 5 slides per class × 7 classes = 35 slides total (so the
  StratifiedGroupKFold split has enough samples per class per fold).
* Each slide: 16–32 tiles, embed_dim=32 (matches ``smoke.yaml``).
* Each class's embeddings cluster around a class-specific centroid
  so a trivial classifier can learn a non-trivial signal in 1 epoch.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from src.evaluation import make_5fold_splits, write_split_csv
from src.evaluation.metrics import CLASS_ORDER
from src.graph_construction import SpatialKnnGraph
from src.utils.io_utils import save_pyg_data

#: Locked at 1536 to match the design contract enforced in
#: ``src.utils.config`` (LOCKED_INVARIANTS). Smoke fixtures are still
#: tiny because we have 70 slides × ~25 tiles each ≈ 1750 vectors.
EMBED_DIM = 1536
#: Need ≥ INNER_K (=7) patients per class in trainval so the inner
#: stratified split inside ``make_5fold_splits`` finds enough samples.
#: With 10 slides/class × 5 outer folds, each fold has ≥ 8 trainval
#: patients per class — comfortably above the inner-k=7 threshold.
SLIDES_PER_CLASS = 10
N_TILES_RANGE = (16, 33)


def make_smoke_fixtures(
    tmp_root: Path,
    *,
    seed: int = 0,
    embed_dim: int = EMBED_DIM,
    slides_per_class: int = SLIDES_PER_CLASS,
) -> Tuple[Path, Path]:
    """Build a synthetic smoke dataset under ``tmp_root``.

    Returns:
        ``(splits_csv, graphs_root)`` — the two paths the trainer needs.

    Layout produced::

        tmp_root/
            splits/cv5fold.csv
            graphs/spatial_knn/<slide_id>.pt   # one per slide
    """
    tmp_root = Path(tmp_root)
    splits_dir = tmp_root / "splits"
    graphs_root = tmp_root / "graphs" / "spatial_knn"
    splits_dir.mkdir(parents=True, exist_ok=True)
    graphs_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Per-class centroid in embedding space — drives the toy signal so
    # a trivial classifier learns above chance in 1 epoch.
    centroids = rng.normal(scale=2.0, size=(len(CLASS_ORDER), embed_dim))

    # Build slide manifest (one row per slide; patient_id mirrors slide_id
    # so each is a singleton group — fits the StratifiedGroupKFold
    # invariant without needing slot multiplicity here).
    rows: list[dict] = []
    for class_idx, label in enumerate(CLASS_ORDER):
        for i in range(slides_per_class):
            slide_id = f"{label}_{i:02d}_1"
            rows.append(
                {
                    "slide_id": slide_id,
                    "patient_id": f"{label}_{i:02d}",
                    "tumor_class": label,
                }
            )
    manifest = pd.DataFrame(rows)

    # 5-fold split CSV. Use inner_k=4 so each val partition has enough
    # samples per class with the small smoke fixture (10 slides/class).
    splits_df = make_5fold_splits(manifest, seed=seed, inner_k=4)
    splits_csv = write_split_csv(splits_df, splits_dir / "cv5fold.csv")

    # Per-slide PyG graph (spatial k-NN over toy coords).
    constructor = SpatialKnnGraph(k=4)
    for _, row in manifest.iterrows():
        n_tiles = int(rng.integers(*N_TILES_RANGE))
        # Embeddings: class centroid + jitter.
        cls_idx = CLASS_ORDER.index(str(row["tumor_class"]))
        emb = centroids[cls_idx] + rng.normal(scale=0.3, size=(n_tiles, embed_dim))
        # Coordinates: small grid jitter.
        coords = rng.uniform(0, 1024, size=(n_tiles, 2))
        data = constructor.build_graph(
            embeddings=torch.from_numpy(emb).to(torch.float32),
            coordinates=torch.from_numpy(coords).to(torch.float32),
            slide_id=str(row["slide_id"]),
            y=cls_idx,
        )
        save_pyg_data(data, graphs_root / f"{row['slide_id']}.pt")

    return splits_csv, graphs_root


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Output root for smoke fixtures.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    splits_csv, graphs_root = make_smoke_fixtures(args.out, seed=args.seed)
    print(f"Wrote splits → {splits_csv}")
    print(f"Wrote graphs → {graphs_root}")
