#!/usr/bin/env python3
"""Build the k_NN-ablation comparison table for Section 6 / 3.5.

Reads the k_NN-sweep summary at results/knn_sweep/summary.json (k_NN = 4
result; k_NN = 16 is OOM-flagged in the same file) and the headline
summary at results/deformable/summary.json (k_NN = 8), and prints the
Markdown table + paired-t / Cohen-d_z / Wilcoxon stats that the
manuscripts cite verbatim.

Companion to scripts/13_k_ablation_table.py — same pure-Python
incomplete-beta CF for the paired-t p-value; no scipy dependency.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Sequence

ROOT = pathlib.Path(__file__).resolve().parent.parent

KNN_SWEEP_JSON = ROOT / "results" / "knn_sweep" / "summary.json"
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
        return float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0, n - 1
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
    return statistics.fmean(diffs) / sd if sd > 0 else float("inf")


def _wilcoxon(diffs: Sequence[float]) -> dict:
    nz = [d for d in diffs if d != 0]
    return {"n_effective": len(nz), "n_positive": sum(1 for d in nz if d > 0)}


def main() -> None:
    knn = json.loads(KNN_SWEEP_JSON.read_text())
    headline = json.loads(HEADLINE_JSON.read_text())

    knn4 = _fold_values(knn["per_knn"]["4"]["metrics"], "val_balanced_accuracy")
    knn8 = _fold_values(headline["metrics"], "val_balanced_accuracy")
    assert len(knn4) == len(knn8) == 5

    knn4_mean, knn4_std = _mean_std(knn4)
    knn8_mean, knn8_std = _mean_std(knn8)
    diffs = [a - b for a, b in zip(knn4, knn8)]  # k_NN=4 minus k_NN=8
    t, p, df = _paired_t(diffs)
    dz = _cohen_dz(diffs)
    wx = _wilcoxon(diffs)

    # k_NN = 16 OOM flag
    knn16_status = knn["per_knn"].get("16", {}).get("note", "no result")

    fmt = (
        "| {k:>2} | [{f0}, {f1}, {f2}, {f3}, {f4}] | {mean:.4f} | {std:.4f} | "
        "{delta} | {dz} | {p} |"
    )

    print("# k_NN-ablation table (val_balanced_accuracy, 5-fold patient-level CV)\n")
    print("| k_NN | per-fold val_bacc | mean | std | Δ vs k_NN=8 | Cohen's d_z | paired-t p |")
    print("|---|---|---|---|---|---|---|")
    print(fmt.format(
        k=4, f0=f"{knn4[0]:.4f}", f1=f"{knn4[1]:.4f}", f2=f"{knn4[2]:.4f}",
        f3=f"{knn4[3]:.4f}", f4=f"{knn4[4]:.4f}",
        mean=knn4_mean, std=knn4_std,
        delta=f"+{(knn4_mean - knn8_mean):.4f}",
        dz=f"{dz:.2f}",
        p=f"{p:.3f}",
    ))
    print(fmt.format(
        k=8, f0=f"{knn8[0]:.4f}", f1=f"{knn8[1]:.4f}", f2=f"{knn8[2]:.4f}",
        f3=f"{knn8[3]:.4f}", f4=f"{knn8[4]:.4f}",
        mean=knn8_mean, std=knn8_std,
        delta="(ref)", dz="—", p="—",
    ))
    print(f"|  16 | OOM at backward (1000 MiB short of 24 GiB at 22.29 GiB used) | — | — | — | — | — |")
    print()
    print(f"- Paired-t (k_NN=4 vs k_NN=8): t = {t:.3f}, df = {df}, two-sided p = {p:.3f}")
    print(f"- Cohen's d_z = {dz:.3f}")
    print(f"- Wilcoxon: n_effective = {wx['n_effective']}, n_positive = {wx['n_positive']}")
    nonneg = sum(1 for d in diffs if d >= 0)
    print(f"- Win-or-tie count for k_NN=4: {nonneg}/{len(diffs)}")
    print(f"- Per-fold diffs (k_NN=4 - k_NN=8): "
          f"[{', '.join(f'{d:+.4f}' for d in diffs)}]")
    print(f"- k_NN = 16 status: {knn16_status}")


if __name__ == "__main__":
    main()
