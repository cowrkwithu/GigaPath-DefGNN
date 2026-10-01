#!/usr/bin/env python3
"""Phase 10.7 — Generate paper figures + tables on demand.

Routes ``--figure <name>`` and ``--table <name>`` to
:data:`src.visualization.FIGURE_REGISTRY` /
:data:`src.visualization.TABLE_REGISTRY`. The actual data inputs (CSV
paths, embeddings, etc.) are loaded by per-figure adapter functions
defined here.

Usage:
    python scripts/06_visualize.py --figure fig03_baseline_bars \\
        --baseline-csv results/tables/table02_baseline.csv

    python scripts/06_visualize.py --table table02_baseline \\
        --baseline-csv results/eval/aggregated_baseline.csv

    python scripts/06_visualize.py --all   # generate every available deliverable

References:
    Design: docs/02-design/09-deliverables.md
    Design: docs/02-design/features/vetgigagraph.do.md Phase 9 + 10.7
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
from src.visualization import (
    DEFERRED_FIGURES,
    FIGURE_REGISTRY,
    TABLE_REGISTRY,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument(
        "--figure",
        type=str,
        default=None,
        choices=list(FIGURE_REGISTRY),
        help="Figure name to generate (see --list).",
    )
    p.add_argument(
        "--table",
        type=str,
        default=None,
        choices=list(TABLE_REGISTRY),
        help="Table name to generate (see --list).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Generate every figure (skipping deferred ones) and table.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available figure + table names and exit.",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Output dir for figures (default: paths.figures).",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Output dir for tables (default: paths.tables).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print("Figures:")
        for name in sorted(FIGURE_REGISTRY):
            tag = " (deferred)" if name in DEFERRED_FIGURES else ""
            print(f"  - {name}{tag}")
        print("\nTables:")
        for name in sorted(TABLE_REGISTRY):
            print(f"  - {name}")
        return 0

    rt = load_runtime(args, out_subdir="figures")
    cfg = rt.config
    figures_dir = args.figures_dir or Path(cfg["paths"]["figures"])
    tables_dir = args.tables_dir or Path(cfg["paths"]["tables"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not (args.figure or args.table or args.all):
        logger.error("Specify --figure, --table, --all, or --list.")
        return 2

    # Most generators need real data inputs (CSVs, NPZ embeddings, etc.).
    # Phase 11 (smoke fixture) wires those adapters; this script is the
    # public CLI entry that the smoke + full pipeline both call.
    logger.info(
        "Phase 11 (smoke fixture) wires per-figure data adapters. This "
        "entry point validates the registry lookup + output path "
        "resolution; actual figure rendering needs --figure/--table data "
        "kwargs once the smoke or full results are available. Use "
        "src.visualization functions directly from a notebook for now."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
