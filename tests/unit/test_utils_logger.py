"""Unit tests for src/utils/logger.py."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.utils.logger import _NoOpRun, setup_logging, wandb_run


def test_setup_logging_console_only():
    log_file = setup_logging(level="DEBUG")
    assert log_file is None
    logger = logging.getLogger("test_console")
    logger.info("hello")
    logger.warning("warn")


def test_setup_logging_with_file(tmp_path: Path):
    log_file = setup_logging(level="INFO", log_dir=tmp_path, run_id="unit_test", jsonl=True)
    assert log_file is not None
    assert log_file.exists()
    logger = logging.getLogger("test_file")
    logger.info("structured-log-line")
    # Force flush
    for h in logging.getLogger().handlers:
        h.flush()
    content = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert any("structured-log-line" in line for line in content)
    # Each line is valid JSON
    for line in content:
        parsed = json.loads(line)
        assert "ts" in parsed
        assert "level" in parsed
        assert "msg" in parsed


def test_wandb_disabled_via_env(monkeypatch):
    monkeypatch.setenv("WANDB_DISABLED", "true")
    with wandb_run(project="vetgigagraph-test") as run:
        assert isinstance(run, _NoOpRun)
        run.log({"x": 1.0})  # no-op should not raise
        assert run.id == "noop"


def test_wandb_disabled_via_mode():
    with wandb_run(project="vetgigagraph-test", mode="disabled") as run:
        assert isinstance(run, _NoOpRun)
