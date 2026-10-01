"""Unit tests for src/utils/config.py."""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.utils.config import DEFAULT_CONFIG_PATH, load_config
from src.utils.errors import ConfigError


def test_default_config_loads():
    cfg = load_config()
    assert cfg.project.num_classes == 7
    assert cfg.feature_extraction.embedding_dim == 1536
    assert tuple(cfg.cross_validation.seeds) == (42, 123, 456, 789, 1024)


def test_required_sections_present():
    cfg = load_config()
    for section in ("project", "paths", "preprocessing", "feature_extraction",
                    "graph", "model", "training", "cross_validation", "evaluation"):
        assert section in cfg, f"section '{section}' missing"


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="Config not found"):
        load_config("/nonexistent/path/to/config.yaml")


def test_missing_required_section(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("project:\n  num_classes: 7\n")  # missing many sections
    with pytest.raises(ConfigError, match="Missing required section"):
        load_config(bad)


def test_locked_invariant_violation(tmp_path: Path):
    """Cannot override num_classes."""
    with pytest.raises(ConfigError, match="Locked invariant violated"):
        load_config(overrides=["project.num_classes=10"])


def test_seeds_must_match_locked_list(tmp_path: Path):
    with pytest.raises(ConfigError, match="cross_validation.seeds"):
        load_config(overrides=["cross_validation.seeds=[1,2,3,4,5]"])


def test_overrides_apply():
    cfg = load_config(overrides=["training.max_epochs=10"])
    assert cfg.training.max_epochs == 10


def test_paths_use_external_dirs():
    """Confirm the locked external paths are pointing where expected."""
    cfg = load_config()
    assert cfg.paths.raw == "/data/cancerImagingArchive"
    assert cfg.paths.output_root == "/data/cia_outputs"
    assert cfg.paths.tiles == "/data/cia_outputs/tiles"
    assert cfg.paths.features == "/data/cia_outputs/features"
