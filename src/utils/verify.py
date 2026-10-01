"""Common verifier skeleton used by ``scripts/_verify_step_*.py``.

Each step's verifier collects independent checks and emits a structured JSON
report. Exit code 0 = all green, non-zero = at least one check failed.

References:
    Design: docs/02-design/06-execution-pipeline.md (Step-Verifier Helper Scripts)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Outcome of a single verification check."""

    name: str
    ok: bool
    detail: str | None = None
    error: str | None = None
    extras: dict = field(default_factory=dict)


def run_check(name: str, fn: Callable[[], bool | tuple[bool, str]]) -> CheckResult:
    """Run one check, capturing any exception as a failure.

    The check function returns either:

    * ``bool`` — True for pass, False for fail.
    * ``tuple[bool, str]`` — (pass/fail, detail message).
    """
    try:
        result = fn()
        if isinstance(result, tuple):
            ok, detail = result
        else:
            ok, detail = bool(result), None
        return CheckResult(name=name, ok=ok, detail=detail)
    except Exception as e:
        logger.exception("Check '%s' raised", name)
        return CheckResult(name=name, ok=False, error=f"{type(e).__name__}: {e}")


def emit_report(checks: list[CheckResult], *, header: str = "verification") -> int:
    """Print a structured JSON report and return an exit code.

    Args:
        checks: All collected check results.
        header: Top-level label (e.g. ``"step_2"``).

    Returns:
        ``0`` if every check passed, ``1`` otherwise.
    """
    failed = [c for c in checks if not c.ok]
    payload = {
        "header": header,
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": [asdict(c) for c in failed],
        "results": [asdict(c) for c in checks],
    }
    json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()
    if failed:
        logger.error("%d/%d checks failed", len(failed), len(checks))
        return 1
    logger.info("All %d checks passed", len(checks))
    return 0
