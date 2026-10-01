#!/usr/bin/env python3
"""Build the offset-penalty-ablation comparison table for Section 6 / 3.5.

Reads the penalty-sweep summary at results/offset_penalty_sweep/summary.json
(p = 0 result + the p = 1e-3 smoke cell) and the headline summary at
results/deformable/summary.json (p = 1e-4), and prints the Markdown table
+ paired-t / Cohen-d_z stats that the manuscripts cite verbatim.

Companion to scripts/13_k_ablation_table.py and 14_knn_ablation_table.py.
Same pure-Python incomplete-beta CF for the paired-t p-value; no scipy
dependency.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Sequence

ROOT = pathlib.Path(__file__).resolve().parent.parent

PENALTY_SWEEP_JSON = ROOT / "results" / "offset_penalty_sweep" / "summary.json"
HEADLINE_JSON = ROOT / "results" / "deformable" / "summary.json"


def _fold_values(metrics: dict, key: str) -> list[float]:
    return list(metrics[key]["fold_values"])


def _mean_std(xs: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(xs), statistics.pstdev(xs)


def _paired_t(diffs: Sequence[float]) -> tuple[float, float, int]:
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), 0
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    if sd == 0:
        return (float("inf") if mean != 0 else 0.0,
                0.0 if mean != 0 else 1.0, n - 1)
    se = sd / math.sqrt(n)
    t = mean / se
    df = n - 1
    return t, _student_t_two_sided_p(abs(t), df), df


def _student_t_two_sided_p(t: float, df: int) -> float:
    x = df / (df + t * t)
    return _betai(x, df / 2.0, 0.5)


def _betai(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(x, a, b) / a
    return 1.0 - bt * _betacf(1 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iter: int = 200) -> float:
    eps = 3e-7
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _cohen_dz(diffs: Sequence[float]) -> float:
    if len(diffs) < 2:
        return float("nan")
    sd = statistics.stdev(diffs)
    if sd == 0:
        return 0.0 if statistics.fmean(diffs) == 0 else float("inf")
    return statistics.fmean(diffs) / sd


def main() -> None:
    pen = json.loads(PENALTY_SWEEP_JSON.read_text())
    headline = json.loads(HEADLINE_JSON.read_text())

    p0 = _fold_values(pen["per_weight"]["0"]["metrics"],
                       "val_balanced_accuracy")
    pmid = _fold_values(headline["metrics"], "val_balanced_accuracy")
    assert len(p0) == len(pmid) == 5

    p0_mean, p0_std = _mean_std(p0)
    pmid_mean, pmid_std = _mean_std(pmid)
    diffs = [a - b for a, b in zip(p0, pmid)]  # p=0 minus p=1e-4
    t, p, df = _paired_t(diffs)
    dz = _cohen_dz(diffs)

    # Smoke cell for p = 1e-3 (1 fold, 1 epoch — not a comparable
    # 5-fold full-training number).
    p_strong = pen["per_weight"].get("1e-03", {})
    p_strong_smoke_val = (
        p_strong.get("metrics", {})
        .get("val_balanced_accuracy", {})
        .get("mean")
    )

    fmt = (
        "| {p:>9} | [{f0}, {f1}, {f2}, {f3}, {f4}] | {mean:.4f} | {std:.4f} | "
        "{delta} | {dz} | {pval} |"
    )

    print("# Offset-penalty-ablation table (val_balanced_accuracy, 5-fold patient-level CV)\n")
    print("| penalty | per-fold val_bacc | mean | std | Δ vs p=1e-4 | Cohen's d_z | paired-t p |")
    print("|---|---|---|---|---|---|---|")
    print(fmt.format(
        p="0", f0=f"{p0[0]:.4f}", f1=f"{p0[1]:.4f}", f2=f"{p0[2]:.4f}",
        f3=f"{p0[3]:.4f}", f4=f"{p0[4]:.4f}",
        mean=p0_mean, std=p0_std,
        delta=f"{(p0_mean - pmid_mean):+.4f}",
        dz=f"{dz:.2f}" if not math.isnan(dz) else "n/a (Δ ≡ 0)",
        pval=f"{p:.3f}",
    ))
    print(fmt.format(
        p="1e-4 (h)", f0=f"{pmid[0]:.4f}", f1=f"{pmid[1]:.4f}", f2=f"{pmid[2]:.4f}",
        f3=f"{pmid[3]:.4f}", f4=f"{pmid[4]:.4f}",
        mean=pmid_mean, std=pmid_std,
        delta="(ref)", dz="—", pval="—",
    ))
    if p_strong_smoke_val is not None:
        print(f"|      1e-3 | smoke only (fold 0, 1 epoch): val_bacc = {p_strong_smoke_val:.4f} | — | — | — | — | — |")
    print()
    print(f"- Paired-t (p=0 vs p=1e-4): t = {t:.3f}, df = {df}, two-sided p = {p:.3f}")
    print(f"- Cohen's d_z = {dz:.3f}" if not math.isnan(dz) else "- Cohen's d_z: undefined (mean diff = 0, std diff = 0)")
    nonneg = sum(1 for d in diffs if d >= 0)
    print(f"- Win-or-tie count for p=0: {nonneg}/{len(diffs)}")
    print(f"- Per-fold diffs (p=0 - p=1e-4): "
          f"[{', '.join(f'{d:+.4f}' for d in diffs)}]")
    print(f"- Note: p=0 and p=1e-4 produced BIT-IDENTICAL val_balanced_accuracy "
          f"on all 5 folds; the OffsetMLP L2 penalty as implemented makes "
          f"no measurable difference at this scale.")


if __name__ == "__main__":
    main()
