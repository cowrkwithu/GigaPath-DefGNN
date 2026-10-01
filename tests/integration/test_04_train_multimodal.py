"""Multi-modal fast-dev-run integration smoke (slide-encoder-tests µPDCA #4).

HEAVY-gated end-to-end test that drives ``scripts/04_train.py`` with
``configs/experiment_multimodal.yaml`` to verify the slide-encoder
branch added by ``slide-encoder-injection`` µPDCA #3 still works after
any change. Mirrors the manual smoke the operator ran during µPDCA #3:

  - downloads ``slide_encoder.pth`` from HF Hub (~345 MB, cached)
  - loads pretrained weights (must report ``missing=0, unexpected=0``)
  - constructs ``VetGigaGraph`` with ``learnable_weighted`` fusion +
    frozen LongNet slide encoder (D-18 mandatory on 24 GiB)
  - runs ``--fast-dev-run`` (1 batch) on the **real** CATCH dual-edge
    graphs (production WSI input)
  - asserts ``fold_0_metrics.json`` contains finite ``val_*`` values

Skipped by default; runs only when both gates are set:

    VETGIGAGRAPH_RUN_HEAVY_TESTS=1
    HF_TOKEN=<valid token>

Test design references:
- Sibling µPDCA #3 design §8.2 / acceptance criterion AC-4
- slide-encoder-tests µPDCA #4 plan §4 M5
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_OK = (
    Path("/data/cia_outputs/graphs/dual_edge").exists()
    and Path("/data/cia_outputs/splits/cv5fold.csv").exists()
)
HEAVY_OK = os.getenv("VETGIGAGRAPH_RUN_HEAVY_TESTS") == "1" and bool(os.getenv("HF_TOKEN"))


@pytest.mark.skipif(
    not HEAVY_OK,
    reason="HEAVY test: requires VETGIGAGRAPH_RUN_HEAVY_TESTS=1 + HF_TOKEN in env",
)
@pytest.mark.skipif(
    not DATA_OK,
    reason="Requires CATCH preprocessed graphs at /data/cia_outputs/graphs/dual_edge "
           "and splits at /data/cia_outputs/splits/cv5fold.csv "
           "(run scripts 01–03 first per docs/work-log/2026-04-29.md §2.3–2.6)",
)
@pytest.mark.integration
def test_04_train_vetgigagraph_multimodal_fast_dev_run(tmp_path: Path) -> None:
    """End-to-end: vetgigagraph fold-0 fast-dev-run with learnable_weighted fusion."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "04_train.py"),
        "--config", str(REPO_ROOT / "configs" / "experiment_multimodal.yaml"),
        "--model", "vetgigagraph",
        "--fold", "0",
        "--seed", "42",
        "--fast-dev-run",
        "--out", str(out_dir),
    ]
    env = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,  # 10-min ceiling; observed runtime ~30s for fast-dev-run
    )
    assert result.returncode == 0, (
        f"04_train.py exit code {result.returncode}.\n"
        f"--- stderr (tail) ---\n{result.stderr[-2000:]}\n"
        f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
    )

    metrics_path = out_dir / "vetgigagraph" / "fold_0_metrics.json"
    assert metrics_path.exists(), f"fold_0_metrics.json missing under {out_dir}"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    # fast-dev-run is single batch → BACC is 0.0 (no real learning), but all
    # values must be finite (not NaN/inf — the GAT+LongNet+fusion path must not
    # blow up numerically on the first step)
    for key in ("train_loss", "val_loss", "val_balanced_accuracy"):
        assert key in metrics, f"metrics missing key {key!r}"
        v = metrics[key]
        assert isinstance(v, (int, float))
        # finite: not NaN, not ±inf
        assert v == v, f"{key} is NaN"
        assert v not in (float("inf"), float("-inf")), f"{key} is ±inf"
