"""End-to-end smoke (Phase 11.4).

Generates a synthetic 7-class × 10-slide graph dataset under
``tmp_path`` (bypasses the WSI/feature-extraction stages, which need
real CATCH data + GigaPath weights), then drives ``scripts/04_train.py``
end-to-end via subprocess on the smallest possible model+config.

Phase 11 → Phase 12 gate per ``vetgigagraph.do.md``: this test must
pass in < 5 min and produce a valid checkpoint + metrics JSON.

The full WSI → tiles → features → graphs pipeline is exercised by the
unit tests for each module; this integration test verifies the
end-to-end *training* pipeline ends in a coherent state given valid
graph inputs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def smoke_workspace(tmp_path: Path):
    """Generate the synthetic smoke dataset; yield the resolved paths."""
    sys.path.insert(0, str(REPO_ROOT))
    from tests.fixtures._make_smoke import make_smoke_fixtures

    splits_csv, graphs_root = make_smoke_fixtures(tmp_path / "data", seed=0)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "splits_csv": splits_csv,
        "graphs_root": graphs_root,
        "out_dir": out_dir,
        "config": REPO_ROOT / "tests" / "fixtures" / "configs" / "smoke.yaml",
    }


@pytest.mark.integration
def test_smoke_pipeline_train_abmil(smoke_workspace) -> None:
    """End-to-end: 1-epoch ABMIL on synthetic graphs → checkpoint + metrics JSON."""
    out_dir = smoke_workspace["out_dir"]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "04_train.py"),
        "--config", str(smoke_workspace["config"]),
        "--model", "abmil",
        "--fold", "0",
        "--splits-csv", str(smoke_workspace["splits_csv"]),
        "--graphs-root", str(smoke_workspace["graphs_root"]),
        "--max-epochs", "1",
        "--mixed-precision", "false",
        "--gpus", "cpu",
        "--out", str(out_dir),
        "--num-workers", "0",
        "--metrics-out", str(out_dir / "abmil_fold0.json"),
        "--checkpoint-dir", str(out_dir / "ckpt"),
        "--log-level", "WARNING",
    ]
    res = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300
    )
    assert res.returncode == 0, (
        f"04_train.py exited {res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    metrics_path = out_dir / "abmil_fold0.json"
    assert metrics_path.exists(), f"metrics JSON not written: {metrics_path}"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "train_loss" in str(metrics) or "train_loss_epoch" in metrics, (
        f"missing expected metric keys in {metrics_path}: {list(metrics.keys())}"
    )
    ckpt = out_dir / "ckpt" / "fold_0.ckpt"
    assert ckpt.exists() and ckpt.stat().st_size > 0, f"checkpoint missing: {ckpt}"


@pytest.mark.integration
def test_smoke_pipeline_train_vetgigagraph(smoke_workspace) -> None:
    """Same but with VetGigaGraph (gnn_only fusion path so no slide encoder)."""
    out_dir = smoke_workspace["out_dir"]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "04_train.py"),
        "--config", str(smoke_workspace["config"]),
        "--model", "vetgigagraph",
        "--fold", "0",
        "--splits-csv", str(smoke_workspace["splits_csv"]),
        "--graphs-root", str(smoke_workspace["graphs_root"]),
        "--max-epochs", "1",
        "--mixed-precision", "false",
        "--gpus", "cpu",
        "--out", str(out_dir),
        "--num-workers", "0",
        "--metrics-out", str(out_dir / "vetgigagraph_fold0.json"),
        "--checkpoint-dir", str(out_dir / "ckpt_vgg"),
        "--log-level", "WARNING",
    ]
    res = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300
    )
    assert res.returncode == 0, (
        f"vetgigagraph smoke exited {res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    assert (out_dir / "vetgigagraph_fold0.json").exists()
    assert (out_dir / "ckpt_vgg" / "fold_0.ckpt").exists()


@pytest.mark.integration
def test_smoke_pipeline_split_csv_well_formed(smoke_workspace) -> None:
    """Generated split CSV has the locked schema and 5 folds.

    Note: full :func:`src.evaluation.verify_splits` (which asserts every
    class is present in every fold's val/test) is unit-tested separately
    in ``test_evaluation.py`` with a 30-slide/class fixture. The smoke
    fixture (10 slides/class) is too small for that invariant to hold
    reliably under stratification rounding — but the *schema* must.
    """
    import pandas as pd

    df = pd.read_csv(smoke_workspace["splits_csv"])
    required = {"slide_id", "patient_id", "tumor_class", "fold", "split"}
    assert set(df.columns) >= required
    assert sorted(df["fold"].unique().tolist()) == [0, 1, 2, 3, 4]
    assert set(df["split"].unique()).issubset({"train", "val", "test"})
