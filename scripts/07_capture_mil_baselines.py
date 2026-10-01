#!/usr/bin/env python3
"""Capture v1 MIL baseline metrics into v3-schema + build H2 comparison.

All 5 MIL aggregators (ABMIL, DSMIL, TransMIL, CLAM-SB, CLAM-MB) were
already trained in the v1 study with 5-fold patient-level CV on the same
locked seeds {42, 123, 456, 789, 1024} we use for the v2 deformable
model. Their per-fold metrics live at
/data/cia_outputs/checkpoints/exp1/<model>/fold_<i>_metrics.json.

This script:
  (1) Wraps each per-fold metrics file into the v3 schema
      (results/mil_baselines/<model>/fold_<i>/metrics.json) with model
      name, fold idx, locked seed, val block, and source path.
  (2) Aggregates per-model summary (results/mil_baselines/<model>/summary.json).
  (3) Aggregates one combined summary (results/mil_baselines/summary.json).
  (4) Reads the v2 deformable summary (results/deformable/summary.json)
      and builds the H2 paired comparison table:
        - Per-model paired-t and Wilcoxon vs v2 deformable
        - Combined ranking table sorted by val_balanced_accuracy
      Writes:
        paper/tables/T13_h2_mil_comparison.md
        paper/figures/T13_h2_raw.csv

Usage:
    python3 scripts/07_capture_mil_baselines.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(V2_ROOT))
from src.utils.env import output_dir, portable_path  # noqa: E402
V1_BASELINES_ROOT = output_dir() / "checkpoints/exp1"
MIL_MODELS = ("abmil", "dsmil", "transmil", "clam_sb", "clam_mb")
LOCKED_SEEDS = [42, 123, 456, 789, 1024]
V2_SUMMARY = V2_ROOT / "results" / "deformable" / "summary.json"

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- #
# Capture
# ----------------------------------------------------------------------- #


def capture_fold(model: str, fold: int, dst_root: Path) -> dict:
    src = V1_BASELINES_ROOT / model / f"fold_{fold}_metrics.json"
    raw = json.loads(src.read_text())
    wrapped = {
        "model": f"v1_{model}",
        "fold": fold,
        "seed": LOCKED_SEEDS[fold],
        "val": {
            "val_loss": raw.get("val_loss"),
            "val_balanced_accuracy": raw.get("val_balanced_accuracy"),
            "val_weighted_f1": raw.get("val_weighted_f1"),
        },
        "train_loss": raw.get("train_loss"),
        "source": portable_path(src),
    }
    dst = dst_root / model / f"fold_{fold}" / "metrics.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(wrapped, indent=2))
    return wrapped


def aggregate_model(fold_rows: list[dict]) -> dict:
    keys = ("val_loss", "val_balanced_accuracy", "val_weighted_f1")
    agg = {}
    for k in keys:
        vals = [r["val"][k] for r in fold_rows if r["val"][k] is not None]
        if not vals:
            continue
        agg[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "fold_values": vals,
        }
    return agg


# ----------------------------------------------------------------------- #
# H2 comparison
# ----------------------------------------------------------------------- #


def paired_tests(mil_vals: list[float], v2_vals: list[float]) -> dict:
    if len(mil_vals) != len(v2_vals) or len(mil_vals) < 2:
        return {"note": "insufficient folds"}
    try:
        from scipy import stats
    except ImportError:
        return {"note": "scipy not installed"}
    t_stat, t_p = stats.ttest_rel(v2_vals, mil_vals)
    try:
        w_stat, w_p = stats.wilcoxon(v2_vals, mil_vals)
        w_p = float(w_p)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    diff = [v2 - m for v2, m in zip(v2_vals, mil_vals)]
    return {
        "n": len(v2_vals),
        "mean_delta_v2_minus_mil": float(statistics.mean(diff)),
        "paired_t": {"stat": float(t_stat), "p": float(t_p)},
        "wilcoxon": {"stat": float(w_stat) if w_stat == w_stat else float("nan"),
                     "p": w_p},
    }


def build_h2_table(per_model: dict, v2_summary: dict, out_md: Path,
                   out_csv: Path) -> None:
    v2_metrics = v2_summary["metrics"]
    v2_bacc_folds = v2_metrics["val_balanced_accuracy"]["fold_values"]
    v2_f1_folds = v2_metrics["val_weighted_f1"]["fold_values"]
    v2_bacc_mean = v2_metrics["val_balanced_accuracy"]["mean"]
    v2_bacc_std  = v2_metrics["val_balanced_accuracy"]["std"]
    v2_f1_mean   = v2_metrics["val_weighted_f1"]["mean"]
    v2_f1_std    = v2_metrics["val_weighted_f1"]["std"]

    # ranking by bal_acc
    rows = []
    for model in MIL_MODELS:
        agg = per_model[model]
        bacc = agg["val_balanced_accuracy"]
        f1 = agg["val_weighted_f1"]
        stats_bacc = paired_tests(bacc["fold_values"], v2_bacc_folds)
        stats_f1   = paired_tests(f1["fold_values"], v2_f1_folds)
        rows.append({
            "model": f"v1_{model}",
            "bacc_mean": bacc["mean"], "bacc_std": bacc["std"],
            "f1_mean": f1["mean"], "f1_std": f1["std"],
            "delta_bacc_v2_minus_mil": v2_bacc_mean - bacc["mean"],
            "delta_f1_v2_minus_mil": v2_f1_mean - f1["mean"],
            "bacc_paired_t_p": stats_bacc.get("paired_t", {}).get("p"),
            "bacc_wilcoxon_p": stats_bacc.get("wilcoxon", {}).get("p"),
            "f1_paired_t_p": stats_f1.get("paired_t", {}).get("p"),
        })

    # Markdown table
    lines = [
        "# Table T13 — H2 Comparison: v2 Deformable vs v1 MIL Aggregators",
        "",
        "Per-model 5-fold mean ± pop-std on identical patient-level CV splits and locked seeds {42, 123, 456, 789, 1024}.",
        "Δ values are **v2 Deformable − v1 MIL** (positive = v2 better).",
        "Paired statistics computed across the matched 5 folds.",
        "",
        "| Model | val_bacc | val_f1 | Δ bacc vs v2 | Δ f1 vs v2 | paired-t p (bacc) | Wilcoxon p (bacc) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} "
            f"| {r['bacc_mean']:.4f} ± {r['bacc_std']:.4f} "
            f"| {r['f1_mean']:.4f} ± {r['f1_std']:.4f} "
            f"| {r['delta_bacc_v2_minus_mil']:+.4f} "
            f"| {r['delta_f1_v2_minus_mil']:+.4f} "
            f"| {(r['bacc_paired_t_p'] if r['bacc_paired_t_p'] is not None else float('nan')):.4f} "
            f"| {(r['bacc_wilcoxon_p'] if r['bacc_wilcoxon_p'] is not None else float('nan')):.4f} |"
        )
    # v2 row
    lines.append(
        f"| **v2_deformable (proposed)** "
        f"| **{v2_bacc_mean:.4f} ± {v2_bacc_std:.4f}** "
        f"| **{v2_f1_mean:.4f} ± {v2_f1_std:.4f}** "
        f"| — | — | — | — |"
    )

    # Ranking summary
    ranked = sorted(rows + [{
        "model": "v2_deformable",
        "bacc_mean": v2_bacc_mean,
    }], key=lambda r: r["bacc_mean"], reverse=True)
    lines += [
        "",
        "## Ranking by val_balanced_accuracy (descending)",
        "",
        "| Rank | Model | val_bacc |",
        "|---|---|---|",
    ]
    for i, r in enumerate(ranked, 1):
        bold = "**" if r["model"] == "v2_deformable" else ""
        lines.append(f"| {i} | {bold}{r['model']}{bold} | {bold}{r['bacc_mean']:.4f}{bold} |")

    # Summary block
    n_v2_wins = sum(1 for r in rows if r["delta_bacc_v2_minus_mil"] > 0)
    n_v2_ties = sum(1 for r in rows if r["delta_bacc_v2_minus_mil"] == 0)
    n_v2_loses = sum(1 for r in rows if r["delta_bacc_v2_minus_mil"] < 0)
    lines += [
        "",
        "## H2 verdict",
        "",
        f"- v2 Deformable mean val_bacc: **{v2_bacc_mean:.4f} ± {v2_bacc_std:.4f}** (5 folds)",
        f"- v2 wins / ties / loses vs 5 MIL aggregators (mean comparison): **{n_v2_wins} / {n_v2_ties} / {n_v2_loses}**",
        f"- All v2 vs MIL paired-t p-values (val_bacc) are reported in column 6 above.",
    ]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    logger.info("[h2] wrote %s", out_md)

    # CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "val_balanced_accuracy", "val_weighted_f1",
                    "val_loss"])
        for model in MIL_MODELS:
            for fold in range(5):
                row = per_model[model]
                bacc = row["val_balanced_accuracy"]["fold_values"][fold]
                f1   = row["val_weighted_f1"]["fold_values"][fold]
                vl   = row["val_loss"]["fold_values"][fold] if "val_loss" in row else None
                w.writerow([f"v1_{model}", fold, bacc, f1, vl])
        # v2 rows
        for fold in range(5):
            bacc = v2_bacc_folds[fold]
            f1 = v2_f1_folds[fold]
            vl = v2_metrics["val_loss"]["fold_values"][fold]
            w.writerow(["v2_deformable", fold, bacc, f1, vl])
    logger.info("[h2] wrote %s", out_csv)


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dst-root", type=Path,
                        default=V2_ROOT / "results" / "mil_baselines")
    parser.add_argument("--out-md", type=Path,
                        default=V2_ROOT / "paper" / "tables" / "T13_h2_mil_comparison.md")
    parser.add_argument("--out-csv", type=Path,
                        default=V2_ROOT / "paper" / "figures" / "T13_h2_raw.csv")
    args = parser.parse_args()

    if not V2_SUMMARY.exists():
        raise SystemExit(f"v2 summary missing: {V2_SUMMARY}")

    per_model: dict[str, dict] = {}
    for model in MIL_MODELS:
        fold_rows: list[dict] = []
        for fold in range(5):
            src = V1_BASELINES_ROOT / model / f"fold_{fold}_metrics.json"
            if not src.exists():
                logger.warning("[capture] %s fold %d missing: %s", model, fold, src)
                continue
            row = capture_fold(model, fold, args.dst_root)
            fold_rows.append(row)
            logger.info("[capture] %s fold %d → bacc=%.4f f1=%.4f",
                        model, fold,
                        row["val"]["val_balanced_accuracy"],
                        row["val"]["val_weighted_f1"])
        agg = aggregate_model(fold_rows)
        summary = {
            "model": f"v1_{model}",
            "n_folds": len(fold_rows),
            "metrics": agg,
        }
        summary_path = args.dst_root / model / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("[summary] %s mean_bacc=%.4f ± %.4f",
                    model,
                    agg["val_balanced_accuracy"]["mean"],
                    agg["val_balanced_accuracy"]["std"])
        per_model[model] = agg

    combined_summary = {
        "n_models": len(per_model),
        "per_model": {f"v1_{m}": s for m, s in per_model.items()},
    }
    combined_path = args.dst_root / "summary.json"
    combined_path.write_text(json.dumps(combined_summary, indent=2))
    logger.info("[summary] combined wrote %s", combined_path)

    v2_summary = json.loads(V2_SUMMARY.read_text())
    build_h2_table(per_model, v2_summary, args.out_md, args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
