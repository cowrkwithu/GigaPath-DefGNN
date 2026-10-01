#!/usr/bin/env python3
"""Phase 10.2 — Preprocess CATCH WSIs into per-slide tile directories.

Wraps :class:`src.preprocessing.WSITiler` in a per-slide loop. Reads
:func:`src.utils.io_utils.discover_slides` to enumerate the CATCH
archive and writes the locked output layout under
``configs/default.yaml paths.tiles``.

Skipped slides (insufficient tiles) are logged to
``logs/skipped_slides.csv`` per the design contract — never silently
dropped.

Usage:
    python scripts/01_preprocess.py --config configs/default.yaml
    python scripts/01_preprocess.py --slides MEL_001 MEL_002      # subset
    python scripts/01_preprocess.py --raw-dir /alt/path           # override
    python scripts/01_preprocess.py --force                       # re-tile finished slides

By default, slides whose ``<tiles>/<slide_id>/metadata.json`` already
exists are skipped (resume-friendly). A directory without
``metadata.json`` is treated as a partial run and re-processed
(``_write_outputs`` overwrites in place). Pass ``--force`` to re-tile
even completed slides.

References:
    Design: docs/02-design/03-architecture.md §2 (Module A)
    Design: docs/02-design/06-execution-pipeline.md Step 2
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import csv
import logging
import sys
from pathlib import Path

from scripts._common import REPO_ROOT, add_common_args, load_runtime
from src.preprocessing import WSITiler
from src.utils.errors import InsufficientTilesError, VetGigaGraphError
from src.utils.io_utils import SlideMetadata, discover_slides, parse_slide_filename

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override CATCH raw root (default: paths.raw from config).",
    )
    p.add_argument(
        "--slides",
        nargs="+",
        default=None,
        help="Restrict to specific slide IDs (e.g. MEL_01_1). Default: all 350.",
    )
    p.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Read (tile_size*d)^2 at level 0 and box-filter to tile_size^2 "
             "(d=2 gives 20x tiles from 40x scans). Default 1 (native 40x).",
    )
    p.add_argument(
        "--prefilter-tissue-fraction",
        type=float,
        default=0.0,
        help="Skip grid cells whose thumbnail tissue fraction is below this "
             "(speed-up only; 0 disables). See WSITiler.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-tile slides whose metadata.json already exists. "
            "Default: skip completed slides (partial directories are re-tiled regardless)."
        ),
    )
    return p


def _resolve_targets(
    raw_dir: Path,
    selected: list[str] | None,
) -> list[SlideMetadata]:
    targets = discover_slides(raw_dir)
    if selected:
        wanted = set(selected)
        targets = [s for s in targets if s.slide_id in wanted]
        missing = wanted - {s.slide_id for s in targets}
        if missing:
            raise FileNotFoundError(f"requested slide IDs not found: {sorted(missing)}")
    return targets


def _partition_completed(
    targets: list[SlideMetadata],
    out_dir: Path,
) -> tuple[list[SlideMetadata], list[SlideMetadata]]:
    """Split ``targets`` into (pending, already_done) by metadata.json existence.

    The "completion marker" is ``<out_dir>/<slide_id>/metadata.json`` —
    written last by :meth:`WSITiler._write_outputs`, so its presence is
    an atomic signal that PNGs + coords.csv finished too. Partial
    directories (PNGs only, no metadata.json) are returned in
    ``pending`` and will be overwritten on re-tile.
    """
    pending: list[SlideMetadata] = []
    completed: list[SlideMetadata] = []
    for meta in targets:
        if (out_dir / meta.slide_id / "metadata.json").exists():
            completed.append(meta)
        else:
            pending.append(meta)
    return pending, completed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rt = load_runtime(args, out_subdir="tiles")

    paths_cfg = rt.config["paths"]
    raw_dir = args.raw_dir or Path(paths_cfg["raw"])
    targets = _resolve_targets(raw_dir, args.slides)

    if args.force:
        pending, already_done = targets, []
    else:
        pending, already_done = _partition_completed(targets, rt.out_dir)

    if already_done:
        logger.info(
            "Skipping %d slide(s) with existing metadata.json (use --force to re-tile).",
            len(already_done),
        )
    logger.info("Tiling %d slide(s) → %s", len(pending), rt.out_dir)

    pp_cfg = rt.config["preprocessing"]
    tiler = WSITiler(
        tile_size=int(pp_cfg["tile_size"]),
        stride=int(pp_cfg["stride"]),
        tissue_threshold=float(pp_cfg["tissue_threshold"]),
        blur_threshold=float(pp_cfg["blur_threshold"]),
        min_tiles_per_slide=int(pp_cfg["min_tiles_per_slide"]),
        max_tiles_per_slide=pp_cfg.get("max_tiles_per_slide"),
        stain_normalization=str(pp_cfg["stain_normalization"]),
        downsample=args.downsample,
        prefilter_tissue_fraction=args.prefilter_tissue_fraction,
    )

    logs_dir = Path(paths_cfg.get("logs", REPO_ROOT / "results" / "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = logs_dir / "skipped_slides.csv"

    skipped: list[tuple[str, str, str]] = []
    for meta in pending:
        try:
            tiler.tile(meta.source_path, out_dir=rt.out_dir, slide_id=meta.slide_id)
        except InsufficientTilesError as e:
            logger.warning("Skipping %s: %s", meta.slide_id, e)
            skipped.append((meta.slide_id, "InsufficientTilesError", str(e)))
        except VetGigaGraphError as e:  # pragma: no cover — surface, do not silently drop
            logger.error("Slide %s failed: %s", meta.slide_id, e)
            skipped.append((meta.slide_id, type(e).__name__, str(e)))

    if skipped:
        with skipped_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(("slide_id", "reason", "detail"))
            w.writerows(skipped)
        logger.warning("Wrote %d skipped slides → %s", len(skipped), skipped_path)
    logger.info(
        "Done: %d tiled, %d failed, %d already done",
        len(pending) - len(skipped),
        len(skipped),
        len(already_done),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
