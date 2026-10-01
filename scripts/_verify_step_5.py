#!/usr/bin/env python3
"""Phase 10.9 — Step 5 verifier (post-training).

Reads per-fold checkpoint + metrics under ``paths.checkpoints`` and
``results/eval/`` and runs the U1–U8 + S1–S6 validators
(:func:`src.evaluation.run_validators`) on each.

Exits 0 if all runs pass at "investigate" or "pass" severity; non-zero
on any "invalidate" failure.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import sys
from pathlib import Path

from scripts._common import add_common_args, fail, load_runtime
from src.evaluation import RunMetadata, aggregate_severity, run_validators


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--eval-root", type=Path, default=None)
    p.add_argument("--checkpoints-root", type=Path, default=None)
    p.add_argument("--splits-csv", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rt = load_runtime(args, out_subdir="eval")
    eval_root = args.eval_root or Path(rt.config["paths"].get("logs", rt.out_dir.parent)) / "eval"
    checkpoints_root = args.checkpoints_root or Path(rt.config["paths"]["checkpoints"])
    splits_csv = args.splits_csv or Path(rt.config["paths"]["splits"]) / "cv5fold.csv"

    if not eval_root.exists():
        fail(f"eval root not found: {eval_root} (run scripts/04_train.py first)")

    overall_status = "pass"
    n_runs = 0
    for metric_file in sorted(eval_root.rglob("fold_*_metrics.json")):
        n_runs += 1
        run_dir = metric_file.parent
        fold = int(metric_file.stem.split("_")[1])
        metrics = json.loads(metric_file.read_text(encoding="utf-8"))
        ckpt = checkpoints_root / run_dir.name / f"fold_{fold}.ckpt"
        meta = RunMetadata(
            config_logged=True,
            seed_was_set=True,
            fold=fold,
            splits_csv=splits_csv if splits_csv.exists() else None,
            metrics=metrics,
            checkpoint_path=ckpt if ckpt.exists() else None,
            training_finished=True,
            train_bacc=metrics.get("train_balanced_accuracy"),
            predictions=metrics.get("predictions"),
        )
        results = run_validators(meta)
        sev = aggregate_severity(results)
        print(f"{run_dir.name}/fold_{fold}: {sev}")
        if sev == "invalidate":
            overall_status = "invalidate"
            for r in results:
                if not r.passed and r.severity == "invalidate":
                    print(f"  ❌ {r.name}: {r.detail}")
        elif sev == "investigate" and overall_status == "pass":
            overall_status = "investigate"

    print(f"\nOK: {n_runs} runs verified — overall status: {overall_status}")
    return 0 if overall_status != "invalidate" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
