"""YAML configuration loader (OmegaConf-based).

Every entry-point script accepts ``--config <path>`` and calls
:func:`load_config`. CLI overrides (``key=value``) are merged on top.

References:
    Design: docs/02-design/04-experiment-design.md (Locked Default Hyperparameters)
    Design: docs/02-design/05-software-architecture.md §3 (Coding Conventions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf

# Side-effect: loads .env into os.environ on first import.
# Keeps secrets (HF_TOKEN, WANDB_API_KEY) out of YAML configs.
from src.utils import env as _env  # noqa: F401  (import for side-effect only)
from src.utils.errors import ConfigError

logger = logging.getLogger(__name__)

#: Default config relative to repo root.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

#: Required top-level sections in every config (validated on load).
REQUIRED_SECTIONS: tuple[str, ...] = (
    "project",
    "paths",
    "preprocessing",
    "feature_extraction",
    "graph",
    "model",
    "training",
    "cross_validation",
    "evaluation",
)

#: Locked values that no override may change (verified post-merge).
LOCKED_INVARIANTS: dict[str, Any] = {
    "project.num_classes": 7,
    "feature_extraction.embedding_dim": 1536,
    "preprocessing.tile_size": 256,
    "preprocessing.stride": 256,
    "preprocessing.magnification": 40,
}


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Iterable[str] | None = None,
    validate: bool = True,
) -> DictConfig:
    """Load a YAML config and merge CLI-style overrides.

    Args:
        path: Path to YAML config. Defaults to ``configs/default.yaml``.
        overrides: Dotlist overrides (e.g. ``["training.max_epochs=10",
            "model.gnn.backbone=gcn"]``). Applied on top of the loaded YAML.
        validate: Run :func:`_validate_config` after merge.

    Returns:
        The merged ``DictConfig``.

    Raises:
        ConfigError: If the file is missing, malformed, or violates
            :data:`LOCKED_INVARIANTS`.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"Config not found: {cfg_path}")

    try:
        cfg = OmegaConf.load(cfg_path)
    except Exception as e:
        raise ConfigError(f"Failed to parse {cfg_path}: {e}") from e

    if not isinstance(cfg, DictConfig):
        raise ConfigError(f"Config root must be a mapping, got {type(cfg).__name__}")

    if overrides:
        try:
            override_cfg = OmegaConf.from_dotlist(list(overrides))
            cfg = OmegaConf.merge(cfg, override_cfg)
        except Exception as e:
            raise ConfigError(f"Failed to apply overrides {overrides!r}: {e}") from e

    assert isinstance(cfg, DictConfig)  # mypy: merge result type
    if validate:
        _validate_config(cfg)
    logger.info("Loaded config from %s (overrides=%s)", cfg_path, list(overrides or []))
    return cfg


def _validate_config(cfg: DictConfig) -> None:
    """Verify required sections exist and locked invariants hold."""
    for section in REQUIRED_SECTIONS:
        if section not in cfg:
            raise ConfigError(f"Missing required section: '{section}'")

    for dotted_key, expected in LOCKED_INVARIANTS.items():
        actual = OmegaConf.select(cfg, dotted_key)
        if actual != expected:
            raise ConfigError(
                f"Locked invariant violated: '{dotted_key}' must be {expected!r}, "
                f"got {actual!r}. These values are fixed by design and cannot be overridden."
            )

    # Cross-validation seeds match the locked list
    from src.utils.seed import LOCKED_FOLD_SEEDS

    seeds = OmegaConf.to_object(cfg.cross_validation.seeds)
    if tuple(seeds) != LOCKED_FOLD_SEEDS:
        raise ConfigError(
            f"cross_validation.seeds must equal {LOCKED_FOLD_SEEDS}, got {tuple(seeds)}. "
            "These are locked by design (08-statistics-reproducibility.md §2)."
        )


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save a config to YAML (used to record exactly-what-ran per checkpoint)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out)
    logger.info("Saved config to %s", out)


def resolve_paths(cfg: DictConfig) -> DictConfig:
    """Resolve ``paths.*`` interpolations and create output directories.

    Returns the same cfg (mutated). Output dirs (everything under
    ``cfg.paths.output_root``) are created with ``parents=True, exist_ok=True``.
    The raw dataset directory is **not** created (must pre-exist, read-only).
    """
    paths = cfg.paths
    raw = Path(paths.raw)
    if not raw.exists():
        raise ConfigError(f"Raw data directory does not exist: {raw}")

    # Create output subdirectories
    for key in ("tiles", "features", "graphs", "splits", "checkpoints", "logs", "figures", "tables", "wandb", "smoke"):
        if key in paths:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

    return cfg
