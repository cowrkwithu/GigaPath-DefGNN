#!/usr/bin/env python3
"""Build the K-ablation comparison table for Section 6 (IMRaD) / 3.5 (MDPI).

Reads the K-sweep summary at results/k_sweep/summary.json (currently K = 1)
and the headline summary at results/deformable/summary.json (K = 2), and
prints a Markdown table + paired-t / Cohen-d_z / Wilcoxon stats that the
manuscripts cite verbatim.

Re-run after any new K-sweep dispatch to refresh the table. Outputs are
deterministic given fixed input JSON.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Sequence

ROOT = pathlib.Path(__file__).resolve().parent.parent

K_SWEEP_JSON = ROOT / "results" / "k_sweep" / "summary.json"
HEADLINE_JSON = ROOT / "results" / "deformable" / "summary.json"


def _fold_values(metrics: dict, key: str) -> list[float]:
    return list(metrics[key]["fold_values"])


def _mean_std(xs: Sequence[float]) -> tuple[float, float]:
    m = statistics.fmean(xs)
    s = statistics.pstdev(xs)  # population std (n-divisor) — matches v1 convention
    return m, s


def _paired_t(diffs: Sequence[float]) -> tuple[float, float, int]:
    """Return (t, two-tailed p approximation, df).

    p is the Student-t survival from a series-expansion; for the n = 5
    case here, this matches scipy.stats.ttest_rel to ~3 decimal places.
    """
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), 0
    mean = statistics.fmean(diffs)
    sample_std = statistics.stdev(diffs)  # n-1 divisor
    if sample_std == 0:
        return float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0, n - 1
    se = sample_std / math.sqrt(n)
    t = mean / se
    df = n - 1
    p = _student_t_two_sided_p(abs(t), df)
    return t, p, df


def _student_t_two_sided_p(t: float, df: int) -> float:
    """Two-tailed p via regularized incomplete beta (closed-form for df=4)."""
    x = df / (df + t * t)
    # incomplete beta I_x(df/2, 1/2) using continued-fraction
    a, b = df / 2.0, 0.5
    return _betai(x, a, b)


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
    else:
        return 1.0 - bt * _betacf(1 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iter: int = 200) -> float:
    eps = 3e-7
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
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


def _wilcoxon_signed_rank(diffs: Sequence[float]) -> dict:
    """Two-sided Wilcoxon signed-rank (zeros excluded, ties averaged)."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return {"W": 0.0, "n_effective": 0, "p": 1.0}
    abs_diffs = sorted(((abs(d), d) for d in nz), key=lambda p: p[0])
    ranks = list(range(1, n + 1))
    # Average tie ranks
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        if j > i:
            avg = (ranks[i] + ranks[j]) / 2
            for k in range(i, j + 1):
                ranks[k] = avg
        i = j + 1
    w_pos = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d > 0)
    w_neg = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d < 0)
    W = min(w_pos, w_neg)
    return {"W": W, "n_effective": n, "p": "n/a (n_effective<6)" if n < 6 else "see scipy"}


def main() -> None:
    k_sweep = json.loads(K_SWEEP_JSON.read_text())
    headline = json.loads(HEADLINE_JSON.read_text())

    k1 = _fold_values(k_sweep["per_K"]["1"]["metrics"], "val_balanced_accuracy")
    k2 = _fold_values(headline["metrics"], "val_balanced_accuracy")
    assert len(k1) == len(k2) == 5, "expected 5 folds for both K=1 and K=2"

    k1_mean, k1_std = _mean_std(k1)
    k2_mean, k2_std = _mean_std(k2)
    diffs = [b - a for a, b in zip(k1, k2)]  # K=2 minus K=1
    t, p, df = _paired_t(diffs)
    dz = _cohen_dz(diffs)
    wx = _wilcoxon_signed_rank(diffs)

    fmt_row = (
        "| {k:>1} | [{f0}, {f1}, {f2}, {f3}, {f4}] | {mean:.4f} | {std:.4f} | "
        "{delta} | {dz} | {p} |"
    )

    print("# K-ablation table (val_balanced_accuracy, 5-fold patient-level CV)\n")
    print("| K | per-fold val_bacc | mean | std | Δ vs K=1 | Cohen d_z | paired-t p |")
    print("|---|---|---|---|---|---|---|")
    print(fmt_row.format(
        k=1, f0=f"{k1[0]:.4f}", f1=f"{k1[1]:.4f}", f2=f"{k1[2]:.4f}",
        f3=f"{k1[3]:.4f}", f4=f"{k1[4]:.4f}",
        mean=k1_mean, std=k1_std,
        delta="(ref)", dz="—", p="—",
    ))
    print(fmt_row.format(
        k=2, f0=f"{k2[0]:.4f}", f1=f"{k2[1]:.4f}", f2=f"{k2[2]:.4f}",
        f3=f"{k2[3]:.4f}", f4=f"{k2[4]:.4f}",
        mean=k2_mean, std=k2_std,
        delta=f"+{(k2_mean - k1_mean):.4f}",
        dz=f"{dz:.2f}",
        p=f"{p:.3f}",
    ))
    print()
    print(f"- Paired-t: t = {t:.3f}, df = {df}, two-sided p = {p:.3f}")
    print(f"- Cohen d_z = {dz:.3f}")
    print(f"- Wilcoxon signed-rank: W = {wx['W']}, n_effective = {wx['n_effective']}, p = {wx['p']}")
    nonneg = sum(1 for d in diffs if d >= 0)
    print(f"- Win-or-tie count for K=2: {nonneg}/{len(diffs)}")
    print(f"- Per-fold diffs (K=2 − K=1): "
          f"[{', '.join(f'{d:+.4f}' for d in diffs)}]")
    print(f"- K=4 / K=8: OOM on 24 GiB RTX 3090 (largest CATCH WSI ≈ 94 K tiles); "
          "not part of this table.")


if __name__ == "__main__":
    main()
