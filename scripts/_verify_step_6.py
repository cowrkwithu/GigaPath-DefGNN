#!/usr/bin/env python3
"""Phase 10.9 — Step 6 verifier (post-deliverable-generation).

Asserts that every figure + table in
``docs/02-design/09-deliverables.md`` §5.1, §5.2 exists and passes
its structural check.

Exits 0 on success, 1 on first failure.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from scripts._common import add_common_args, fail, load_runtime

REQUIRED_FIGURES = {
    "fig03_baseline_bars.svg": "svg",
    "fig04_graph_radar.svg": "svg",
    "fig05_confusion.png": "png",
    "fig06_roc.svg": "svg",
    "fig08_tsne.png": "png",
    "fig09_transfer.svg": "svg",
    "figS1_curves.svg": "svg",
}

REQUIRED_TABLES = (
    "table01_dataset",
    "table02_baseline",
    "table03_graphs",
    "table04_backbones",
    "table05_fusion",
    "table06_transfer",
    "table07_per_class",
    "table08_cost",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--figures-dir", type=Path, default=None)
    p.add_argument("--tables-dir", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rt = load_runtime(args, out_subdir="figures")
    figures_dir = args.figures_dir or Path(rt.config["paths"]["figures"])
    tables_dir = args.tables_dir or Path(rt.config["paths"]["tables"])

    for fname, kind in REQUIRED_FIGURES.items():
        path = figures_dir / fname
        if not path.exists():
            fail(f"missing {path}")
        if kind == "svg":
            try:
                ET.parse(path)
            except ET.ParseError as e:
                fail(f"{path}: invalid XML ({e})")
        elif kind == "png":
            with Image.open(path) as img:
                if img.format != "PNG":
                    fail(f"{path}: not a PNG")
        print(f"  ✅ {path}")

    for tname in REQUIRED_TABLES:
        for ext in ("csv", "tex"):
            path = tables_dir / f"{tname}.{ext}"
            if not path.exists():
                fail(f"missing {path}")
            if ext == "tex" and "\\caption" not in path.read_text(encoding="utf-8"):
                fail(f"{path}: missing \\caption")
            print(f"  ✅ {path}")

    print(f"\nOK: all {len(REQUIRED_FIGURES)} figures + {len(REQUIRED_TABLES) * 2} table files verified.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
