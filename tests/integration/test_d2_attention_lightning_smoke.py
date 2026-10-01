"""D-2 v2 cosine-similarity attention — Lightning deep-path smoke (G1 closure).

Closes G1 from `docs/03-analysis/slide-encoder-d2-real-attention-v2.analysis.md`:
the v2 µPDCA's M6 milestone called for a Lightning ``Trainer.fit()`` deep-path
smoke as a checked-in pytest, but only a manual run was performed. This test
provides the CI regression guard.

Why this matters:
    The D-2 v1 attempt (commits f346697 + fc42645, reverted 2026-05-16) shipped
    with file-level invariants verified but no Lightning deep-path test. A hard
    ``assert self.args.flash_attention`` in
    ``src/models/_vendored_gigapath/torchscale/component/dilated_attention.py:145``
    tripped during ``Trainer._run_sanity_check`` and bricked 21/30 fold runs of
    the Exp 4 sweep. See ``docs/work-log/2026-05-16.md``.

    The v2 cosine-similarity reformulation eliminates the v1 incident class
    structurally (no vendored args mutation), but the *contract* — "the
    D-2 path runs end-to-end through Lightning fit() without exception" —
    must remain mechanically verified going forward. This test is that guard.

Behaviour:
    - Drives ``scripts/04_train.py`` with ``/tmp/exp4_configs/concat.yaml``
      (the same config used in the manual M6 verification) at
      ``--fast-dev-run`` (1 sanity batch + 1 train batch).
    - Asserts exit code 0 (the v1 incident would fail at the sanity-check
      step with AssertionError).
    - Asserts ``fold_0_metrics.json`` is written with finite values.
    - Skipped without ``VETGIGAGRAPH_RUN_HEAVY_TESTS=1`` + ``HF_TOKEN``
      (needs HF download of ~345 MB ``slide_encoder.pth``).

References:
- µPDCA #5 v2 plan §3 I4 (Lightning Trainer.fit smoke)
- µPDCA #5 v2 plan §M6 exit criterion
- µPDCA #5 v2 analysis §3 G1 (this test closes the gap)
- Incident record: docs/work-log/2026-05-16.md
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
CONCAT_CFG = Path("/tmp/exp4_configs/concat.yaml")
HEAVY_OK = (
    os.getenv("VETGIGAGRAPH_RUN_HEAVY_TESTS") == "1"
    and bool(os.getenv("HF_TOKEN"))
    and CONCAT_CFG.exists()
)


@pytest.mark.skipif(
    not HEAVY_OK,
    reason=(
        "HEAVY test: requires VETGIGAGRAPH_RUN_HEAVY_TESTS=1 + HF_TOKEN "
        f"+ {CONCAT_CFG} present"
    ),
)
@pytest.mark.skipif(
    not DATA_OK,
    reason=(
        "Requires CATCH preprocessed graphs at /data/cia_outputs/graphs/dual_edge "
        "and splits at /data/cia_outputs/splits/cv5fold.csv "
        "(see docs/work-log/2026-04-29.md §2.3–2.6)"
    ),
)
@pytest.mark.integration
def test_d2_v2_lightning_deep_path_smoke(tmp_path: Path) -> None:
    """End-to-end Lightning Trainer.fit() smoke on the D-2 v2 cosine path.

    This is the regression guard that v1 lacked. If a future change to
    ``src/models/gigapath_slide.py`` (or any of its dependencies) breaks
    the deep code path again, this test fails — *before* the sweep does.
    """
    out_dir = tmp_path / "d2v2_smoke_concat"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "04_train.py"),
        "--config", str(CONCAT_CFG),
        "--model", "vetgigagraph",
        "--fold", "0",
        "--seed", "42",
        "--gpus", "auto",
        "--out", str(out_dir),
        "--fast-dev-run",
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
        timeout=600,  # 10-min ceiling; observed ~30s for fast-dev-run
    )
    # The v1 incident would fail here with AssertionError from
    # dilated_attention.py:145. v2 cosine must pass cleanly.
    assert result.returncode == 0, (
        f"D-2 v2 Lightning deep-path smoke failed (exit {result.returncode}). "
        f"If you see 'AssertionError' in stderr referencing "
        f"dilated_attention.py:145, the v1 incident has been reintroduced — "
        f"see docs/work-log/2026-05-16.md.\n"
        f"--- stderr (tail) ---\n{result.stderr[-2000:]}\n"
        f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
    )

    metrics_path = out_dir / "vetgigagraph" / "fold_0_metrics.json"
    assert metrics_path.exists(), (
        f"fold_0_metrics.json missing under {out_dir} — Trainer.fit() may have "
        f"exited cleanly without running the verifier hook."
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("train_loss", "val_loss", "val_balanced_accuracy"):
        assert key in metrics, f"metrics missing key {key!r}"
        v = metrics[key]
        assert isinstance(v, (int, float))
        # finite check — D-2 cosine path must not blow up numerically
        assert v == v, f"{key} is NaN"
        assert v not in (float("inf"), float("-inf")), f"{key} is ±inf"
