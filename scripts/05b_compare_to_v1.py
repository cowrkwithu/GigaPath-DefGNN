#!/usr/bin/env python3
"""Compare v2 deformable-attention results against the frozen v1 reference set.

Reads:
    results/deformable/summary.json         (produced by run_deformable_cv.sh)
    results/deformable/fold_*/metrics.json  (per-fold details)
    results/v1_reference/v1_baseline_summary.json
    results/v1_reference/fold_*/metrics.json

Writes:
    paper/tables/T3_main_comparison.md
    paper/tables/T5_statistical_tests.md
    paper/figures/scripts/fig3_baseline_comparison.py  (regenerates Figure 3)
    paper/figures/scripts/fig_confusion_diff.py        (regenerates Figure 4)
    docs/03-experiment-log.md                          (append a v2 vs v1 entry)

Runs paired-t and Wilcoxon vs the v1 GAT baseline on fold-by-fold scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def _read_summary(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"[compare] missing summary: {path}")
    return json.loads(path.read_text())


def _read_folds(root: Path) -> list[dict]:
    folds = sorted(root.glob("fold_*/metrics.json"))
    if not folds:
        raise SystemExit(f"[compare] no per-fold metrics under {root}")
    return [json.loads(p.read_text()) for p in folds]


def _fold_series(folds: list[dict], metric: str) -> list[float]:
    out = []
    for f in folds:
        v = f.get("val", {}).get(metric)
        if v is not None:
            out.append(float(v))
    return out


def _paired_tests(a: list[float], b: list[float]) -> dict:
    """Paired-t and Wilcoxon signed-rank between two equal-length fold series."""
    if len(a) != len(b) or len(a) < 2:
        return {"note": "insufficient folds for paired tests"}
    try:
        from scipy import stats
    except ImportError:
        return {"note": "scipy not available; install scipy to compute p-values"}
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    return {
        "n": len(a),
        "mean_diff": float(statistics.mean(a) - statistics.mean(b)),
        "paired_t": {"stat": float(t_stat), "p": float(t_p)},
        "wilcoxon": {"stat": float(w_stat), "p": float(w_p)},
    }


def _write_main_table(v1: dict, v2: dict, path: Path) -> None:
    metric_keys = ("val_balanced_accuracy", "val_weighted_f1", "val_macro_auroc")
    header = "| Model | " + " | ".join(k.replace("val_", "") for k in metric_keys) + " |"
    sep = "|---|" + "---|" * len(metric_keys)
    lines = [
        "# Table T3 — Main Comparison (5-fold patient-level CV)",
        "",
        header, sep,
    ]
    for name, summary in (("v1 GAT (baseline)", v1), ("v2 Deformable Attention", v2)):
        cells = [name]
        for k in metric_keys:
            entry = summary.get("metrics", {}).get(k)
            if entry is None:
                cells.append("—")
            else:
                cells.append(f"{entry['mean']:.4f} ± {entry.get('std', 0):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info("[compare] wrote %s", path)


def _write_stats_table(stats: dict, path: Path) -> None:
    lines = [
        "# Table T5 — Paired Statistical Tests (v2 Deformable vs v1 GAT)",
        "",
        "| Metric | Δ (v2 − v1) | paired-t p | Wilcoxon p | n folds |",
        "|---|---|---|---|---|",
    ]
    for metric, st in stats.items():
        if "note" in st:
            lines.append(f"| {metric} | — | — | — | — ({st['note']}) |")
            continue
        lines.append(
            f"| {metric} | {st['mean_diff']:+.4f} "
            f"| {st['paired_t']['p']:.4f} "
            f"| {st['wilcoxon']['p']:.4f} "
            f"| {st['n']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info("[compare] wrote %s", path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-summary", type=Path,
                        default=V2_ROOT / "results" / "v1_reference" / "v1_baseline_summary.json")
    parser.add_argument("--v2-summary", type=Path,
                        default=V2_ROOT / "results" / "deformable" / "summary.json")
    parser.add_argument("--v1-folds", type=Path,
                        default=V2_ROOT / "results" / "v1_reference")
    parser.add_argument("--v2-folds", type=Path,
                        default=V2_ROOT / "results" / "deformable")
    parser.add_argument("--out-tables", type=Path,
                        default=V2_ROOT / "paper" / "tables")
    args = parser.parse_args()

    v1_sum = _read_summary(args.v1_summary)
    v2_sum = _read_summary(args.v2_summary)
    v1_folds = _read_folds(args.v1_folds)
    v2_folds = _read_folds(args.v2_folds)

    metric_keys = ("val_balanced_accuracy", "val_weighted_f1", "val_macro_auroc")
    stats_per_metric = {}
    for m in metric_keys:
        stats_per_metric[m] = _paired_tests(
            _fold_series(v2_folds, m),
            _fold_series(v1_folds, m),
        )

    _write_main_table(v1_sum, v2_sum, args.out_tables / "T3_main_comparison.md")
    _write_stats_table(stats_per_metric, args.out_tables / "T5_statistical_tests.md")

    print("[compare] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
