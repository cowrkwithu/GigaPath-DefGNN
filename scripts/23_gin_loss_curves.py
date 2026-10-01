#!/usr/bin/env python3
"""Figure S1 (R2-5): per-epoch training loss of GIN variants.

Published GIN runs were stopped at their first training batch (before any
parameter update) by the old NaNGuard (first fp16 GradScaler overflow); the retrained variants use the fixed guard.
Small multiples, one panel per variant: thin lines = folds, thick = median
over folds (to the shortest fold); shared log y-axis.

Run:  python3 scripts/23_gin_loss_curves.py
Out:  results/figures_rev1/Figure_S1_gin_training.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(ROOT))
from src.utils.env import output_dir  # noqa: E402
RUNS = output_dir() / "checkpoints/rev1"
VARIANTS = [  # (run dir, label, colour) — categorical slots 1-4, validated palette
    ("gin_fix", "GIN, sum, fp16 (retrained)", "#2a78d6"),
    ("gin_fp32", "GIN, sum, fp32", "#eb6834"),
    ("gin_layernorm", "GIN, sum + LayerNorm, fp16", "#1baf7a"),
    ("gin_mean", "GIN, mean aggregation, fp16", "#eda100"),
]

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), dpi=300, sharex=True, sharey=True)
pub = [json.loads((output_dir() / "checkpoints/exp3/gin/vetgigagraph" / f"fold_{f}_metrics.json").read_text())["train_loss_epoch"] for f in range(5)]
for ax, (run, label, colour) in zip(axes.ravel(), VARIANTS):
    curves = []
    for f in range(5):
        h = json.loads((RUNS / run / f"fold_{f}" / "fold_metrics.history.json").read_text())
        loss = [r["train_loss_epoch"] for r in h if "train_loss_epoch" in r]
        curves.append(loss)
        ax.plot(range(1, len(loss) + 1), loss, color=colour, lw=0.7, alpha=0.4)
    n = min(len(c) for c in curves)
    ax.plot(range(1, n + 1), np.median([c[:n] for c in curves], axis=0), color=colour, lw=2)
    if run == "gin_fix":  # same configuration as the published runs
        ax.scatter([1] * 5, pub, marker="x", s=30, color="#3d3d3a", zorder=5)
        ax.annotate("published runs: loss on the\nfirst batch, where they stopped", (1, max(pub)), xytext=(16, 0),
                    textcoords="offset points", fontsize=6.5, color="#3d3d3a", va="top")
    ax.set_title(label, fontsize=8, color="#3d3d3a", loc="left")
    ax.set_yscale("log")
    ax.grid(axis="y", color="#e4e3dc", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7)
for ax in axes[1]:
    ax.set_xlabel("Epoch", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("Training loss (log scale)", fontsize=8)
fig.tight_layout()
out = ROOT / "results/figures_rev1/Figure_S1_gin_training.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out, "published epoch-1 losses:", [round(x, 2) for x in pub])
