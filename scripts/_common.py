"""Shared CLI helpers for the ``scripts/`` pipeline (Phase 10).

Every pipeline script (``01_preprocess.py`` … ``06_visualize.py``) is a
thin wrapper that:

1. Calls :func:`add_common_args` to register the locked flag set
   (``--config``, ``--seed``, ``--gpus``, ``--wandb_project``, ``--out``).
2. Parses + loads ``configs/<file>.yaml`` via :func:`load_runtime`.
3. Seeds RNGs deterministically via :func:`src.utils.seed.set_global_seed`.
4. Routes the parsed args into the appropriate ``src/`` module.

No actual pipeline logic lives here — this is just glue. That keeps the
unit tests for the scripts fast (just an `--help` parse) and means
breaking changes to the locked flag set produce a single-file diff.

References:
    Design: docs/02-design/features/vetgigagraph.do.md Phase 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.utils.seed import set_global_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class Runtime:
    """Resolved runtime context after argparse + config load."""

    args: argparse.Namespace
    config: Any
    out_dir: Path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register the locked common flag set.

    Locked names per `vetgigagraph.do.md` Phase 10:
        ``--config`` (default: ``configs/default.yaml``)
        ``--seed``   (default: 42, the canonical seed)
        ``--gpus``   (default: ``"auto"``, passed to Lightning)
        ``--wandb_project`` (default: ``vetgigagraph``)
        ``--out``    (default: derived from config; per-script subdir)
    """
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for Python random / numpy / torch (default: 42).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="Lightning device spec ('auto', integer, or comma list).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vetgigagraph",
        help="WandB project name (set WANDB_API_KEY in .env).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override the per-script output directory (default: from config).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )


def load_runtime(args: argparse.Namespace, *, out_subdir: Optional[str] = None) -> Runtime:
    """Resolve the runtime context: load config, seed RNGs, set up logging.

    Args:
        args: Parsed argparse namespace from a script that registered
            :func:`add_common_args`.
        out_subdir: Default subdirectory under ``config.paths.output_root``
            when ``--out`` is not given. Each script passes its own
            (e.g. ``"tiles"``, ``"features"``).
    """
    setup_logging(level=args.log_level)
    config = load_config(args.config)
    set_global_seed(args.seed)

    out_dir = args.out
    if out_dir is None and out_subdir is not None:
        # Pull the matching path from config.paths if it exists; else fallback
        # to <output_root>/<out_subdir>.
        paths = config.get("paths") if hasattr(config, "get") else getattr(config, "paths", {})
        if paths is not None and out_subdir in paths:
            out_dir = Path(paths[out_subdir])
        elif paths is not None and "output_root" in paths:
            out_dir = Path(paths["output_root"]) / out_subdir
    if out_dir is None:
        out_dir = REPO_ROOT / "results" / (out_subdir or "output")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return Runtime(args=args, config=config, out_dir=Path(out_dir))


def fail(msg: str, *, exit_code: int = 1) -> None:
    """Log an error and exit non-zero. Used by verifier scripts."""
    logging.getLogger("scripts").error(msg)
    sys.exit(exit_code)
