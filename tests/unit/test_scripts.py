"""Unit tests for ``scripts/`` (Phase 10).

Scripts are thin argparse wrappers around ``src/`` modules — there's
no logic of their own to test in isolation. The unit-level contract is:

1. Every script imports cleanly (no syntax / import-cycle errors).
2. Every script's ``--help`` exits 0 (argparse spec is well-formed).
3. The locked common flag set ``{--config, --seed, --gpus,
   --wandb_project, --out, --log-level}`` is present on every Python
   pipeline script (per ``vetgigagraph.do.md`` Phase 10).

End-to-end smoke runs (with real WSI input) belong to Phase 11.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: Python pipeline scripts that must implement the locked common flag set.
PIPELINE_SCRIPTS = (
    "01_preprocess.py",
    "02_extract_features.py",
    "03_build_graphs.py",
    "04_train.py",
    "05_evaluate.py",
    "06_visualize.py",
)

#: Per-step verifier scripts.
VERIFIER_SCRIPTS = (
    "_verify_step_2.py",
    "_verify_step_3.py",
    "_verify_step_4.py",
    "_verify_step_5.py",
    "_verify_step_6.py",
)

#: Locked common flags from `scripts/_common.py::add_common_args`.
COMMON_FLAGS = ("--config", "--seed", "--gpus", "--wandb_project", "--out", "--log-level")


def _run_help(script_name: str) -> subprocess.CompletedProcess:
    """Run ``python scripts/<name> --help`` from the repo root."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script_name), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


# --------------------------------------------------------------------------- #
# Import sanity
# --------------------------------------------------------------------------- #


def test_common_module_imports() -> None:
    """`scripts._common` must import as a package member without side effects."""
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts._common import add_common_args, load_runtime, fail, Runtime; "
            "assert callable(add_common_args) and callable(load_runtime)",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, f"_common import failed:\n{res.stderr}"


# --------------------------------------------------------------------------- #
# --help on every pipeline script
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("script", PIPELINE_SCRIPTS)
def test_pipeline_script_help_exits_zero(script: str) -> None:
    res = _run_help(script)
    assert res.returncode == 0, (
        f"{script} --help exited {res.returncode}\nstderr:\n{res.stderr}"
    )
    assert res.stdout, f"{script} --help printed nothing"


@pytest.mark.parametrize("script", PIPELINE_SCRIPTS)
def test_pipeline_script_has_locked_common_flags(script: str) -> None:
    res = _run_help(script)
    for flag in COMMON_FLAGS:
        assert flag in res.stdout, (
            f"{script}: locked flag {flag} not in --help output\n{res.stdout}"
        )


# --------------------------------------------------------------------------- #
# Verifiers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("script", VERIFIER_SCRIPTS)
def test_verifier_script_help_exits_zero(script: str) -> None:
    res = _run_help(script)
    assert res.returncode == 0, (
        f"{script} --help exited {res.returncode}\nstderr:\n{res.stderr}"
    )


# --------------------------------------------------------------------------- #
# Listing without args
# --------------------------------------------------------------------------- #


def test_visualize_list_flag() -> None:
    """`06_visualize.py --list` enumerates the registries without erroring."""
    res = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "06_visualize.py"), "--list"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0
    assert "fig03_baseline_bars" in res.stdout
    assert "table01_dataset" in res.stdout
    assert "(deferred)" in res.stdout  # marks fig01/fig02/fig07


def test_train_supports_baseline_models() -> None:
    """`04_train.py --help` lists every BASELINE_REGISTRY name + vetgigagraph."""
    res = _run_help("04_train.py")
    for name in ("abmil", "dsmil", "transmil", "clam_sb", "clam_mb", "vetgigagraph"):
        assert name in res.stdout, f"04_train.py missing model option: {name}"


def test_build_graphs_supports_all_variants() -> None:
    res = _run_help("03_build_graphs.py")
    for variant in ("spatial_knn", "feature_sim", "dual_edge", "hierarchical", "heterogeneous"):
        assert variant in res.stdout, f"03_build_graphs.py missing variant: {variant}"


# --------------------------------------------------------------------------- #
# Bash drivers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("script", ("00_preflight.sh", "run_all_experiments.sh", "verify_all.sh"))
def test_bash_script_is_executable(script: str) -> None:
    path = SCRIPTS_DIR / script
    assert path.exists()
    assert (path.stat().st_mode & 0o111) != 0, f"{script} is not executable"
    # Sanity: must start with a shebang.
    assert path.read_text().startswith("#!"), f"{script} missing shebang"


# --------------------------------------------------------------------------- #
# 01_preprocess: resume-friendly skip of completed slides
# --------------------------------------------------------------------------- #


def test_partition_completed_uses_metadata_json_marker(tmp_path: Path) -> None:
    """`_partition_completed` skips slides whose metadata.json already exists.

    The marker is the *last* file ``WSITiler._write_outputs`` writes, so
    its presence is an atomic completion signal. A directory that has
    PNGs but no metadata.json (partial / interrupted run) must remain in
    ``pending`` so it gets overwritten on re-tile.
    """
    spec = importlib.util.spec_from_file_location(
        "_preprocess_module", SCRIPTS_DIR / "01_preprocess.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from src.utils.io_utils import SlideMetadata  # local import — needs sys.path setup

    def _meta(sid: str) -> SlideMetadata:
        return SlideMetadata(
            slide_id=sid,
            patient_id=sid.rsplit("_", 1)[0],
            tumor_class=sid.split("_", 1)[0],
            label=0,
            source_path=Path(f"/fake/{sid}.svs"),
            raw_class_dir="Fake",
            patient_num="01",
            slot=sid.rsplit("_", 1)[1],
        )

    out_dir = tmp_path
    targets = [_meta("MEL_01_1"), _meta("MEL_02_1"), _meta("MCT_06_1")]

    # MEL_01_1: fully done (metadata.json present).
    (out_dir / "MEL_01_1").mkdir()
    (out_dir / "MEL_01_1" / "metadata.json").write_text("{}")
    # MEL_02_1: partial — directory and a PNG, but no metadata.json.
    (out_dir / "MEL_02_1").mkdir()
    (out_dir / "MEL_02_1" / "tile_000000_0_0.png").write_bytes(b"fake")
    # MCT_06_1: not started.

    pending, completed = mod._partition_completed(targets, out_dir)

    pending_ids = {m.slide_id for m in pending}
    completed_ids = {m.slide_id for m in completed}
    assert completed_ids == {"MEL_01_1"}
    assert pending_ids == {"MEL_02_1", "MCT_06_1"}, (
        "partial directory must re-tile; not-started must re-tile"
    )


def test_preprocess_help_advertises_force_flag() -> None:
    """`01_preprocess.py --help` must document the new --force flag."""
    res = _run_help("01_preprocess.py")
    assert res.returncode == 0
    assert "--force" in res.stdout, (
        "01_preprocess.py: --force flag missing from --help output"
    )


# --------------------------------------------------------------------------- #
# 03_build_graphs: resume-friendly skip + watch-mode streaming
# --------------------------------------------------------------------------- #


def _load_build_graphs_module():
    spec = importlib.util.spec_from_file_location(
        "_build_graphs_module", SCRIPTS_DIR / "03_build_graphs.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_graphs_partition_uses_pt_marker(tmp_path: Path) -> None:
    """`_partition_completed` skips slides whose ``<slide_id>.pt`` exists.

    The marker is the ``.pt`` written by :func:`save_pyg_data` (single
    ``torch.save``) at the end of per-slide processing — its presence
    signals the graph is built. An ``.h5`` whose ``.pt`` is missing
    must remain in ``pending`` (resume).
    """
    mod = _load_build_graphs_module()

    features_root = tmp_path / "features"
    graphs_root = tmp_path / "graphs"
    features_root.mkdir()
    graphs_root.mkdir()

    # Three feature files; only one has a corresponding .pt.
    h5_files = []
    for sid in ("HIS_01_1", "MEL_05_1", "MCT_02_1"):
        h5 = features_root / f"{sid}.h5"
        h5.write_bytes(b"")  # contents irrelevant to partitioning
        h5_files.append(h5)
    (graphs_root / "HIS_01_1.pt").write_bytes(b"fake")

    pending, completed = mod._partition_completed(h5_files, graphs_root)

    assert {p.stem for p in completed} == {"HIS_01_1"}
    assert {p.stem for p in pending} == {"MEL_05_1", "MCT_02_1"}


def test_build_graphs_is_h5_ready_respects_min_age(tmp_path: Path) -> None:
    """`_is_h5_ready` blocks recent writes and rejects empty / missing files.

    Mid-write ``.h5`` (just-touched) → False. Old + non-empty → True.
    Missing / zero-byte → False.
    """
    import os

    mod = _load_build_graphs_module()

    # Missing file → False.
    missing = tmp_path / "nope.h5"
    assert mod._is_h5_ready(missing, min_age_seconds=1.0) is False

    # Zero-byte file → False (writer just created it but hasn't flushed).
    empty = tmp_path / "empty.h5"
    empty.touch()
    # Force mtime to look old to ensure the rejection is on size, not age.
    old = time.time() - 60
    os.utime(empty, (old, old))
    assert mod._is_h5_ready(empty, min_age_seconds=1.0) is False

    # Recent + non-empty → False (mtime gate).
    recent = tmp_path / "recent.h5"
    recent.write_bytes(b"x" * 16)
    # mtime is "now", min_age=10s → not ready
    assert mod._is_h5_ready(recent, min_age_seconds=10.0) is False

    # Old + non-empty → True.
    old_path = tmp_path / "old.h5"
    old_path.write_bytes(b"x" * 16)
    os.utime(old_path, (old, old))
    assert mod._is_h5_ready(old_path, min_age_seconds=10.0) is True


def test_build_graphs_label_from_slide_id() -> None:
    """`_label_from_slide_id` maps abbreviated slide_ids → integer labels.

    Regression test: previous code called ``parse_slide_filename(sid + ".svs")``
    which expected the *raw class-directory* form (Histiocytoma_01_1.svs)
    and raised DataIntegrityError on every abbreviated slide_id (HIS_01_1).
    The bare except swallowed it and silently produced graphs with
    ``y=None`` — destroying the training supervision signal. The fix
    looks up the abbreviation directly in LABEL_TO_INT.
    """
    mod = _load_build_graphs_module()
    # All seven CATCH classes — slide_ids are <ABBREV>_<PATIENT>_<SLOT>.
    cases = {
        "MEL_01_1": 0,
        "MCT_05_1": 1,
        "SCC_03_2": 2,
        "PNST_07_1": 3,
        "PLC_04_1": 4,
        "TRB_15_1": 5,
        "HIS_01_1": 6,
    }
    for sid, expected in cases.items():
        assert mod._label_from_slide_id(sid) == expected, sid
    # Unknown abbreviations → None (handled by caller as an error).
    assert mod._label_from_slide_id("XYZ_01_1") is None


def test_build_graphs_help_advertises_streaming_flags() -> None:
    """`03_build_graphs.py --help` must document --watch / --force / --min-age-seconds."""
    res = _run_help("03_build_graphs.py")
    assert res.returncode == 0
    for flag in ("--watch", "--force", "--poll-interval", "--min-age-seconds", "--expected-total"):
        assert flag in res.stdout, (
            f"03_build_graphs.py: {flag} missing from --help output"
        )
