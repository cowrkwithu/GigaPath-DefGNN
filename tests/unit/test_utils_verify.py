"""Unit tests for src/utils/verify.py."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from src.utils.verify import CheckResult, emit_report, run_check


def test_run_check_pass():
    result = run_check("trivial", lambda: True)
    assert result.ok is True
    assert result.error is None


def test_run_check_fail_bool():
    result = run_check("trivial-fail", lambda: False)
    assert result.ok is False


def test_run_check_with_detail():
    result = run_check("with-detail", lambda: (True, "found 350 slides"))
    assert result.ok is True
    assert result.detail == "found 350 slides"


def test_run_check_exception_captured():
    def boom() -> bool:
        raise ValueError("synthetic")
    result = run_check("boom", boom)
    assert result.ok is False
    assert "ValueError" in (result.error or "")
    assert "synthetic" in (result.error or "")


def test_emit_report_all_pass():
    checks = [CheckResult(name=f"c{i}", ok=True) for i in range(3)]
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = emit_report(checks, header="test")
    assert code == 0
    parsed = json.loads(buf.getvalue())
    assert parsed["total"] == 3
    assert parsed["passed"] == 3
    assert parsed["failed"] == []


def test_emit_report_some_fail():
    checks = [
        CheckResult(name="ok", ok=True),
        CheckResult(name="bad", ok=False, error="oops"),
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = emit_report(checks, header="test")
    assert code == 1
    parsed = json.loads(buf.getvalue())
    assert parsed["passed"] == 1
    assert len(parsed["failed"]) == 1
    assert parsed["failed"][0]["name"] == "bad"
