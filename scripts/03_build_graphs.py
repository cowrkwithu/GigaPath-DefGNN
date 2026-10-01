#!/usr/bin/env python3
"""Phase 10.4 — Build per-slide PyG graphs from frozen embeddings.

Looks up the variant in :data:`src.graph_construction.GRAPH_VARIANTS`
and runs the constructor over every HDF5 under ``paths.features``.
Outputs land at ``paths.graphs/<graph_type>/<slide_id>.pt``.

Usage:
    python scripts/03_build_graphs.py --graph-type dual_edge
    python scripts/03_build_graphs.py --graph-type spatial_knn --slides MEL_01_1
    python scripts/03_build_graphs.py --watch                   # stream alongside extraction
    python scripts/03_build_graphs.py --force                   # rebuild already-done graphs

By default, slides whose ``<graphs_root>/<graph_type>/<slide_id>.pt``
already exists are skipped (resume-friendly). HDF5 files whose mtime
is within ``--min-age-seconds`` are also skipped — that grace window
prevents reading a feature file mid-write while extraction is still
running. Pass ``--force`` to rebuild even completed graphs.

In ``--watch`` mode the script polls ``paths.features`` every
``--poll-interval`` seconds, builds whatever new + ready ``.h5`` files
have appeared, and exits once ``.pt`` count reaches ``--expected-total``
(default 350). This lets graph construction run in parallel with
``02_extract_features.py`` instead of waiting for it to finish.

References:
    Design: docs/02-design/03-architecture.md §4 (Module C)
    Design: docs/02-design/06-execution-pipeline.md Step 4
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from scripts._common import add_common_args, load_runtime
from src.graph_construction import GRAPH_VARIANTS
from src.utils.errors import DataIntegrityError
from src.utils.io_utils import (
    LABEL_TO_INT,
    read_features_h5,
    save_pyg_data,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument(
        "--graph-type",
        type=str,
        choices=sorted(GRAPH_VARIANTS.keys()),
        default=None,
        help="Variant to build (default: graph.type from config).",
    )
    p.add_argument(
        "--features-root",
        type=Path,
        default=None,
        help="Override HDF5 input dir (default: paths.features).",
    )
    p.add_argument(
        "--graphs-root",
        type=Path,
        default=None,
        help="Override .pt output dir (default: paths.graphs).",
    )
    p.add_argument(
        "--slides",
        nargs="+",
        default=None,
        help="Restrict to specific slide IDs.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild slides whose .pt already exists. "
            "Default: skip completed graphs (resume-friendly)."
        ),
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help=(
            "Poll the features dir until --expected-total .pt files exist. "
            "Lets graph construction run alongside 02_extract_features.py."
        ),
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=60.0,
        help="Seconds between polls in --watch mode (default: 60).",
    )
    p.add_argument(
        "--min-age-seconds",
        type=float,
        default=10.0,
        help=(
            "Skip .h5 files whose mtime is younger than this many seconds. "
            "Guards against reading a feature file mid-write (default: 10)."
        ),
    )
    p.add_argument(
        "--expected-total",
        type=int,
        default=350,
        help="In --watch mode, exit when .pt count reaches this (default: 350).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Device for graph construction ('auto', 'cuda', 'cuda:0', 'cpu'). "
            "Default 'auto' picks CUDA when available. Constructors run k-NN, "
            "cdist and similarity matmuls on this device; saved .pt files are "
            "always on CPU."
        ),
    )
    return p


def _partition_completed(
    h5_files: list[Path],
    graphs_root: Path,
) -> tuple[list[Path], list[Path]]:
    """Split ``h5_files`` into (pending, already_done) by ``.pt`` existence.

    The completion marker is ``<graphs_root>/<slide_id>.pt`` — written
    by :func:`save_pyg_data` (a single ``torch.save`` call) at the end
    of per-slide processing. Its presence signals the graph is built.
    """
    pending: list[Path] = []
    completed: list[Path] = []
    for h5 in h5_files:
        if (graphs_root / f"{h5.stem}.pt").exists():
            completed.append(h5)
        else:
            pending.append(h5)
    return pending, completed


def _is_h5_ready(h5_path: Path, min_age_seconds: float) -> bool:
    """File exists, is non-empty, and last modified ≥ min_age_seconds ago.

    The mtime grace prevents racing :func:`write_features_h5`, which
    opens-truncates-writes-closes the destination directly (no tmp +
    rename). For typical 10–300 MB outputs the close-to-fsync window
    is well under 1 s, so a 10 s default has comfortable headroom.
    """
    try:
        st = h5_path.stat()
    except FileNotFoundError:
        return False
    if st.st_size == 0:
        return False
    return (time.time() - st.st_mtime) >= min_age_seconds


def _resolve_h5_files(
    features_root: Path,
    selected: list[str] | None,
) -> list[Path]:
    if selected:
        h5_files = [features_root / f"{sid}.h5" for sid in selected]
        # Watch / resume modes tolerate not-yet-existing inputs; the
        # readiness gate filters them.
        return h5_files
    return sorted(features_root.glob("*.h5"))


def _label_from_slide_id(slide_id: str) -> int | None:
    """Return the integer class label for a slide_id, or None if unknown.

    ``slide_id`` is in the locked abbreviated form ``<ABBREV>_<PATIENT>_<SLOT>``
    (e.g. ``HIS_01_1`` for a Histiocytoma slide). The first token is the
    class abbreviation; we look it up in ``LABEL_TO_INT``. We deliberately
    do NOT call :func:`parse_slide_filename`, which expects the *raw*
    class-directory form (e.g. ``Histiocytoma_01_1.svs``) and would raise
    ``DataIntegrityError`` for every slide_id ever passed here.
    """
    abbrev = slide_id.split("_", 1)[0]
    return LABEL_TO_INT.get(abbrev)


def _build_one(
    constructor: Any,
    h5_path: Path,
    out_path: Path,
) -> bool:
    """Build one graph; return True on success, False on read error."""
    slide_id = h5_path.stem
    label = _label_from_slide_id(slide_id)
    if label is None:
        # Don't silently train without a label — the design contract requires
        # supervised CV. Surface the bad slide_id loudly.
        logger.error(
            "Unknown class abbreviation in slide_id %r — skipping. "
            "Expected one of %s.",
            slide_id,
            sorted(LABEL_TO_INT),
        )
        return False
    try:
        payload = read_features_h5(h5_path)
    except (OSError, DataIntegrityError) as e:
        # File appeared on disk but was not (yet) a complete HDF5 — try again
        # next poll. In --watch mode this is expected when min-age-seconds is
        # too aggressive vs. the writer's flush cadence.
        logger.warning("Skipping %s — feature HDF5 not ready (%s).", slide_id, e)
        return False
    # µPDCA #8 M3: for HeterogeneousGraph, thread in per-WSI tile labels so
    # the constructor can use GT CATCH categories instead of k-means.
    build_kwargs = dict(
        embeddings=payload["embeddings"],
        coordinates=payload["coordinates"],
        slide_id=slide_id,
        y=label,
    )
    from src.graph_construction.heterogeneous import HeterogeneousGraph
    if isinstance(constructor, HeterogeneousGraph) and constructor.use_gt_labels:
        from src.utils.io_utils import load_tile_labels
        tile_labels = load_tile_labels(slide_id)
        if tile_labels is not None:
            build_kwargs["tile_labels"] = tile_labels
    data = constructor.build_graph(**build_kwargs)
    save_pyg_data(data, out_path)
    return True


def _run_pass(
    constructor: Any,
    features_root: Path,
    graphs_root: Path,
    selected: list[str] | None,
    *,
    force: bool,
    min_age_seconds: float,
) -> tuple[int, int, int]:
    """One pass over the features dir.

    Returns ``(built, skipped_not_ready, already_done)`` so the watch
    loop can decide whether to keep polling.
    """
    h5_files = _resolve_h5_files(features_root, selected)
    if force:
        pending, already_done = h5_files, []
    else:
        pending, already_done = _partition_completed(h5_files, graphs_root)

    built = 0
    not_ready = 0
    for h5 in pending:
        if not _is_h5_ready(h5, min_age_seconds):
            not_ready += 1
            continue
        out_path = graphs_root / f"{h5.stem}.pt"
        if _build_one(constructor, h5, out_path):
            built += 1
        else:
            not_ready += 1  # transient read failure → effectively not ready
    return built, not_ready, len(already_done)


def _build_constructor(graph_type: str, g_cfg: dict, *, device: str = "auto") -> Any:
    constructor_cls = GRAPH_VARIANTS[graph_type]
    kwargs: dict = {"device": device}
    if graph_type == "spatial_knn":
        kwargs["k"] = int(g_cfg["spatial"]["k"])
    elif graph_type == "feature_sim":
        kwargs["tau"] = float(g_cfg["feature"]["threshold"])
        # D-19 fix: optional per-node top-K cap to bound memory on dense slides
        # (N > ~50k tiles). Pulled from `graph.feature.max_edges_per_node`
        # if present; default None preserves legacy unbounded behavior.
        max_k = g_cfg.get("feature", {}).get("max_edges_per_node")
        if max_k is not None:
            kwargs["max_edges_per_node"] = int(max_k)
    elif graph_type == "heterogeneous":
        # µPDCA #8 M2: use GT CATCH labels when available (default True).
        # The constructor will fall back to k-means for unmapped tiles.
        kwargs["use_gt_labels"] = bool(
            g_cfg.get("heterogeneous", {}).get("use_gt_labels", True)
        )
    elif graph_type == "dual_edge":
        kwargs["spatial_k"] = int(g_cfg["spatial"]["k"])
        kwargs["feature_k"] = int(g_cfg["feature"]["k"])
    elif graph_type == "hierarchical":
        kwargs["k_level1"] = int(g_cfg["spatial"]["k"])
    # heterogeneous: default k=8 + k-means fallback (no extra kwargs from config)
    return constructor_cls(**kwargs)


def _log_device_choice(requested: str) -> torch.device:
    """Resolve the requested device and log what we got.

    Falls back to CPU with a warning when the user asked for CUDA but
    the host has no GPU — surfacing this loudly avoids silently running
    a 24-hour CPU build that the user thought was on GPU.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info(
                "Graph build device: cuda (auto-detected, %s).",
                torch.cuda.get_device_name(0),
            )
        else:
            dev = torch.device("cpu")
            logger.info("Graph build device: cpu (no CUDA available).")
        return dev

    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "Requested device %r but CUDA is not available — falling back to CPU.",
            requested,
        )
        return torch.device("cpu")
    logger.info("Graph build device: %s (explicit).", dev)
    return dev


def _write_build_manifest(
    graphs_root: Path,
    *,
    graph_type: str,
    device: torch.device,
) -> None:
    """Append the current build pass to ``<graphs_root>/_build_manifest.json``.

    Closes finding F6 of the 2026-05-15 GPU graph-build gap analysis:
    surfaces mixed-device builds across folds so the Phase-12 CV
    protocol can audit whether all 350 ``.pt`` files in a given variant
    came from one device. ``cdist + topk`` resolves float32 cosine ties
    differently from scipy KDTree (~5% feature-KNN edge sym-diff), so
    mixing devices within a variant directory is a reproducibility risk.

    The manifest is append-only history; readers should treat the
    last entry as authoritative for the current state on disk.
    """
    manifest_path = graphs_root / "_build_manifest.json"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "graph_type": graph_type,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if manifest_path.exists():
        try:
            history = json.loads(manifest_path.read_text())
            if not isinstance(history, list):
                history = [history]
        except json.JSONDecodeError:
            # Corrupted manifest — start fresh rather than crashing the build.
            logger.warning("Resetting unreadable %s.", manifest_path)
            history = []
    else:
        history = []
    history.append(entry)
    manifest_path.write_text(json.dumps(history, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rt = load_runtime(args, out_subdir="graphs")
    cfg = rt.config

    graph_type = args.graph_type or str(cfg["graph"]["type"])
    if graph_type not in GRAPH_VARIANTS:
        raise ValueError(f"unknown graph type {graph_type!r}; expected one of {sorted(GRAPH_VARIANTS)}")
    device = _log_device_choice(str(args.device))
    constructor = _build_constructor(graph_type, cfg["graph"], device=str(device))

    features_root = args.features_root or Path(cfg["paths"]["features"])
    graphs_root = (args.graphs_root or Path(cfg["paths"]["graphs"])) / graph_type
    graphs_root.mkdir(parents=True, exist_ok=True)
    _write_build_manifest(graphs_root, graph_type=graph_type, device=device)

    if args.watch:
        return _watch_loop(
            constructor=constructor,
            features_root=features_root,
            graphs_root=graphs_root,
            selected=args.slides,
            graph_type=graph_type,
            poll_interval=float(args.poll_interval),
            min_age_seconds=float(args.min_age_seconds),
            expected_total=int(args.expected_total),
            force=bool(args.force),
        )

    # Single-shot mode: behaviour matches the original script + new --force flag.
    if args.slides:
        # In single-shot mode, all requested .h5 must exist.
        h5_files = [features_root / f"{sid}.h5" for sid in args.slides]
        missing = [p for p in h5_files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing feature HDF5: {[str(p) for p in missing]}")

    built, not_ready, already_done = _run_pass(
        constructor=constructor,
        features_root=features_root,
        graphs_root=graphs_root,
        selected=args.slides,
        force=bool(args.force),
        min_age_seconds=float(args.min_age_seconds),
    )

    if already_done:
        logger.info(
            "Skipping %d slide(s) with existing .pt (use --force to rebuild).",
            already_done,
        )
    if not_ready:
        logger.warning(
            "%d slide(s) skipped — feature HDF5 not ready (mtime within --min-age-seconds, "
            "or read failed). Re-run after extraction settles.",
            not_ready,
        )
    logger.info(
        "Done: %d %s graphs built, %d already done, %d not ready.",
        built,
        graph_type,
        already_done,
        not_ready,
    )
    return 0


def _watch_loop(
    *,
    constructor: Any,
    features_root: Path,
    graphs_root: Path,
    selected: list[str] | None,
    graph_type: str,
    poll_interval: float,
    min_age_seconds: float,
    expected_total: int,
    force: bool,
) -> int:
    """Poll features_root and build new graphs until expected_total .pt exist."""
    logger.info(
        "Watch mode: graph_type=%s expected=%d poll=%.1fs min_age=%.1fs",
        graph_type,
        expected_total,
        poll_interval,
        min_age_seconds,
    )
    poll = 0
    while True:
        poll += 1
        built, not_ready, already_done = _run_pass(
            constructor=constructor,
            features_root=features_root,
            graphs_root=graphs_root,
            selected=selected,
            force=force,
            min_age_seconds=min_age_seconds,
        )
        pt_count = sum(1 for _ in graphs_root.glob("*.pt"))
        logger.info(
            "poll=%d built=%d not_ready=%d already_done=%d pt_total=%d/%d",
            poll,
            built,
            not_ready,
            already_done,
            pt_count,
            expected_total,
        )
        if pt_count >= expected_total:
            logger.info("Reached expected total %d — exiting watch loop.", expected_total)
            return 0
        time.sleep(poll_interval)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
