"""Structured logging setup.

The pipeline uses two channels:

* **Diagnostics** (``logging`` stdlib) — code-level info, warnings, errors.
* **Experiments** (``wandb.log()``) — per-step metrics, artifacts, config snapshots.

This module configures the diagnostics channel and provides a thin WandB
context manager that no-ops when ``wandb`` is unavailable or disabled.

References:
    Design: docs/02-design/05-software-architecture.md §3
    Design: docs/02-design/04-experiment-design.md §6.1 U1 (config logging)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class _JsonlFormatter(logging.Formatter):
    """One JSON object per line — friendly for downstream log parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Allow callers to attach structured context via `extra={...}`
        for k, v in record.__dict__.items():
            if k.startswith("ctx_"):
                payload[k[4:]] = v
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    *,
    level: str | int = "INFO",
    log_dir: str | Path | None = None,
    run_id: str | None = None,
    jsonl: bool = True,
) -> Path | None:
    """Configure the root logger.

    Args:
        level: Logging level (string or numeric). Default ``INFO``.
        log_dir: If provided, append a JSONL log file there (one per run).
        run_id: Run identifier for the log filename. Defaults to UTC timestamp.
        jsonl: If True (default) the file handler writes JSONL; if False uses
            human-readable single-line format.

    Returns:
        Path to the JSONL log file, or None if ``log_dir`` was not provided.
    """
    root = logging.getLogger()
    # Clear any existing handlers (e.g. set by 3rd-party libs at import time)
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level)

    # Console handler — concise, level-tagged
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
                          datefmt="%Y-%m-%dT%H:%M:%S")
    )
    root.addHandler(console)

    log_file: Path | None = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        log_file = log_dir / f"{run_id}.log.jsonl"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=64 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(_JsonlFormatter() if jsonl else
                                  logging.Formatter("%(asctime)s [%(levelname)s] %(name)s :: %(message)s"))
        root.addHandler(file_handler)

    # Quiet known noisy libraries unless DEBUG
    if logging.getLevelName(level) != "DEBUG":
        for noisy in ("urllib3", "matplotlib", "PIL", "huggingface_hub", "openslide"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return log_file


@contextmanager
def wandb_run(
    *,
    project: str,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    mode: str | None = None,
) -> Iterator[Any]:
    """Context manager that yields a WandB run, or None if WandB is disabled.

    Disabled when:

    * ``wandb`` is not importable, or
    * ``WANDB_DISABLED=true`` env var is set, or
    * ``mode == "disabled"``.

    The yielded object exposes ``log(dict)`` and ``finish()`` semantics; when
    disabled, all calls become no-ops.

    Args:
        project: WandB project name.
        config: Run configuration to log.
        name: Run display name.
        tags: Searchable tags.
        mode: ``online`` / ``offline`` / ``disabled``. Default reads
            ``WANDB_MODE`` env var, else ``online``.
    """
    disabled = (
        os.environ.get("WANDB_DISABLED", "").lower() == "true"
        or mode == "disabled"
    )

    if disabled:
        logger.info("WandB disabled (env or arg) — using no-op logger")
        yield _NoOpRun()
        return

    try:
        import wandb  # type: ignore
    except ImportError:
        logger.warning("wandb not installed — using no-op logger")
        yield _NoOpRun()
        return

    run = wandb.init(project=project, config=config, name=name, tags=tags or [], mode=mode)
    try:
        yield run
    finally:
        run.finish()


class _NoOpRun:
    """Stand-in for a WandB run when logging is disabled."""

    def log(self, *_: Any, **__: Any) -> None:
        return None

    def finish(self, *_: Any, **__: Any) -> None:
        return None

    @property
    def id(self) -> str:
        return "noop"
