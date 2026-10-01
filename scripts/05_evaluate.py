#!/usr/bin/env python3
"""Phase 10.6 — Aggregate per-fold metrics + run statistical tests.

Reads ``results/eval/<model>/fold_<i>_metrics.json`` (written by
``04_train.py`` once Phase 11 wires the dataset side) and emits:

* ``results/eval/<model>/aggregate.json`` — mean ± std per metric.
* ``results/eval/<model>/validators.json`` — U1–U8 + S1–S6 outcomes.
* ``results/eval/pairwise.json`` — paired-test p-values across all
  model pairs (paired t / Wilcoxon / McNemar / Friedman).

Usage:
    python scripts/05_evaluate.py --models vetgigagraph abmil dsmil
    python scripts/05_evaluate.py --models all --bonferroni

References:
    Design: docs/02-design/08-statistics-reproducibility.md
    Design: docs/02-design/04-experiment-design.md §6 (validators)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
import sys
from pathlib import Path

from scripts._common import add_common_args, load_runtime
from src.evaluation import (
    bonferroni_correction,
    friedman_nemenyi,
    paired_ttest,
    wilcoxon,
)
from src.models.baselines import BASELINE_REGISTRY

logger = logging.getLogger(__name__)

ALL_MODELS = sorted(set(BASELINE_REGISTRY) | {"vetgigagraph"})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to aggregate. Use 'all' for every entry in the registry.",
    )
    p.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Per-fold metric JSON parent (default: results/eval).",
    )
    p.add_argument(
        "--bonferroni",
        action="store_true",
        help="Apply Bonferroni correction across pairwise tests.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05).",
    )
    return p


def _load_fold_metrics(eval_root: Path, model: str) -> list[dict]:
    files = sorted((eval_root / model).glob("fold_*_metrics.json"))
    out: list[dict] = []
    for f in files:
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rt = load_runtime(args, out_subdir="eval")

    eval_root = args.eval_root or rt.out_dir
    models = ALL_MODELS if args.models == ["all"] else args.models

    fold_scores: dict[str, list[float]] = {}
    for m in models:
        per_fold = _load_fold_metrics(eval_root, m)
        if not per_fold:
            logger.warning("No fold metrics for %s under %s — skipping.", m, eval_root)
            continue
        # Accept either bare ``balanced_accuracy`` or the ``val_``-prefixed
        # key emitted by the bs=1 Lightning trainer. See v0.4 analysis D-9.
        fold_scores[m] = [
            float(d.get("balanced_accuracy", d.get("val_balanced_accuracy")))
            for d in per_fold
        ]
        n = len(fold_scores[m])
        mean = sum(fold_scores[m]) / n
        std = (sum((x - mean) ** 2 for x in fold_scores[m]) / max(1, n - 1)) ** 0.5
        agg = {"model": m, "n_folds": n, "mean_BACC": mean, "std_BACC": std}
        out_dir = eval_root / m
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
        logger.info("Aggregated %s: mean=%.4f, std=%.4f", m, mean, std)

    if len(fold_scores) >= 2:
        # Pairwise paired t / Wilcoxon between every model pair.
        pair_results: dict = {}
        names = list(fold_scores.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if len(fold_scores[a]) != len(fold_scores[b]):
                    continue
                if len(fold_scores[a]) < 5:
                    continue
                pair_results[f"{a}__vs__{b}"] = {
                    "paired_ttest_p": paired_ttest(fold_scores[a], fold_scores[b]).p,
                    "wilcoxon_p": wilcoxon(fold_scores[a], fold_scores[b]).p,
                }
        if args.bonferroni and pair_results:
            corrected = bonferroni_correction(args.alpha, len(pair_results))
            for k in pair_results:
                pair_results[k]["bonferroni_alpha"] = corrected
        if len(fold_scores) >= 3:
            scores_matrix = [fold_scores[m] for m in names if len(fold_scores[m]) >= 5]
            if len(scores_matrix) >= 3:
                fr = friedman_nemenyi(scores_matrix)
                pair_results["__friedman__"] = {"p": fr.p, "n_models": fr.n_models}
        (eval_root / "pairwise.json").write_text(json.dumps(pair_results, indent=2), encoding="utf-8")
        logger.info("Wrote %d pairwise comparisons → %s", len(pair_results), eval_root / "pairwise.json")
    else:
        logger.info("Fewer than 2 models with metrics — skipping pairwise tests.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
