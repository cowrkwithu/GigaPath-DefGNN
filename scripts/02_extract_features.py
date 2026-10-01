#!/usr/bin/env python3
"""Phase 10.3 — Run frozen GigaPath tile encoder over preprocessed tiles.

Walks each per-slide tile directory under ``paths.tiles`` and writes a
locked-schema HDF5 file under ``paths.features`` via
:func:`src.feature_extraction.encode_slide`.

Usage:
    python scripts/02_extract_features.py --config configs/default.yaml
    python scripts/02_extract_features.py --slides MEL_01_1            # subset
    python scripts/02_extract_features.py --batch-size 128             # override

References:
    Design: docs/02-design/03-architecture.md §3 (Module B)
    Design: docs/02-design/06-execution-pipeline.md Step 3
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import logging
import sys
from pathlib import Path

from scripts._common import add_common_args, load_runtime
from src.feature_extraction import (
    GigaPathTileEncoder,
    encode_slide,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument(
        "--tiles-root",
        type=Path,
        default=None,
        help="Override per-slide tile-dir parent (default: paths.tiles).",
    )
    p.add_argument(
        "--features-root",
        type=Path,
        default=None,
        help="Override HDF5 output dir (default: paths.features).",
    )
    p.add_argument(
        "--slides",
        nargs="+",
        default=None,
        help="Restrict to specific slide IDs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override DataLoader batch size (default: feature_extraction.batch_size).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers (default: feature_extraction.num_workers).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rt = load_runtime(args, out_subdir="features")

    paths_cfg = rt.config["paths"]
    fe_cfg = rt.config["feature_extraction"]
    tiles_root = args.tiles_root or Path(paths_cfg["tiles"])
    features_root = args.features_root or Path(paths_cfg["features"])
    features_root.mkdir(parents=True, exist_ok=True)

    encoder = GigaPathTileEncoder(
        model_name=str(fe_cfg["model"]),
        embedding_dim=int(fe_cfg["embedding_dim"]),
        pretrained=True,
    )

    targets: list[Path]
    if args.slides:
        targets = [tiles_root / sid for sid in args.slides]
        missing = [p for p in targets if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing tile dirs: {[str(p) for p in missing]}")
    else:
        targets = sorted(p for p in tiles_root.iterdir() if p.is_dir() and not p.name.startswith("_"))

    logger.info("Encoding %d slides → %s", len(targets), features_root)
    bs = int(args.batch_size if args.batch_size is not None else fe_cfg["batch_size"])
    nw = int(args.num_workers if args.num_workers is not None else fe_cfg["num_workers"])

    for tile_dir in targets:
        out_h5 = features_root / f"{tile_dir.name}.h5"
        encode_slide(
            encoder=encoder,
            tiles_dir=tile_dir,
            out_h5=out_h5,
            batch_size=bs,
            num_workers=nw,
            slide_id=tile_dir.name,
        )
    logger.info("Done: %d HDF5 files written.", len(targets))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
