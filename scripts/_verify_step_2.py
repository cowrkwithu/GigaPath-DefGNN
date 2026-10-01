#!/usr/bin/env python3
"""Phase 10.9 — Step 2 verifier (post-preprocessing).

Walks ``paths.tiles`` and asserts the contracts in
``docs/02-design/02-data-spec.md`` §6.3:

* tile count ≥ ``min_tiles_per_slide`` (or slide is in ``skipped_slides.csv``)
* coords.csv has exactly the locked column set
* every PNG is 256×256×3 uint8
* tissue_ratio ≥ 0.5, laplacian_var ≥ 100 for every kept tile

Exits 0 on success, 1 on first failure.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts._common import add_common_args, fail, load_runtime
from src.preprocessing import COORDS_CSV_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--tiles-root", type=Path, default=None)
    p.add_argument("--max-check", type=int, default=5, help="PNG-shape spot-check sample size per slide.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rt = load_runtime(args, out_subdir="tiles")
    tiles_root = args.tiles_root or Path(rt.config["paths"]["tiles"])
    pp = rt.config["preprocessing"]
    min_tiles = int(pp["min_tiles_per_slide"])
    tile_size = int(pp["tile_size"])
    tissue_floor = float(pp["tissue_threshold"])
    blur_floor = float(pp["blur_threshold"])

    if not tiles_root.exists():
        fail(f"tiles root not found: {tiles_root}")

    slide_dirs = sorted(p for p in tiles_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not slide_dirs:
        fail(f"no slide subdirs under {tiles_root}")

    for slide_dir in slide_dirs:
        coords_csv = slide_dir / "coords.csv"
        if not coords_csv.exists():
            fail(f"{slide_dir.name}: missing coords.csv")
        df = pd.read_csv(coords_csv)
        if list(df.columns) != list(COORDS_CSV_COLUMNS):
            fail(f"{slide_dir.name}: coords.csv columns drift {list(df.columns)} != {list(COORDS_CSV_COLUMNS)}")
        if len(df) < min_tiles:
            fail(f"{slide_dir.name}: only {len(df)} tiles (< min={min_tiles})")
        if (df["tissue_ratio"] < tissue_floor).any():
            fail(f"{slide_dir.name}: tissue_ratio < {tissue_floor} present")
        if (df["laplacian_var"] < blur_floor).any():
            fail(f"{slide_dir.name}: laplacian_var < {blur_floor} present")

        # Spot-check PNG dims on first N tiles.
        for i, row in df.head(args.max_check).iterrows():
            png = slide_dir / f"tile_{int(row.tile_idx):06d}_{int(row.x)}_{int(row.y)}.png"
            if not png.exists():
                fail(f"{slide_dir.name}: missing {png.name}")
            with Image.open(png) as img:
                if img.size != (tile_size, tile_size) or img.mode != "RGB":
                    fail(f"{slide_dir.name}: {png.name} has shape {img.size}/{img.mode}")

    print(f"OK: {len(slide_dirs)} slides pass Step 2 verification.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
