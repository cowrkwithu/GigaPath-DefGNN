#!/usr/bin/env python3
"""Revised Figures 3 and 4 (held-out test, 1st revision).

Figure 3: held-out balanced accuracy per aggregator with patient-clustered
          bootstrap 95% CI, best-validation (filled circle), latest-top-3
          (open circle) and final weights (diamond). The published
          GigaPath-DefGNN run never saved final weights, so its final marker
          is the re-run of the same configuration (defgnn_dual).
Figure 4: pooled held-out confusion matrices, GAT vs GigaPath-DefGNN.

Run:  python3 scripts/24_revision_figures.py
Out:  results/figures_rev1/Figure_3_test_forest.png, Figure_4_confusion_test.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures_rev1"
ST = json.loads((ROOT / "results/test_eval/test_stats.json").read_text())["rules"]
PRED = json.loads((ROOT / "results/test_eval/predictions.json").read_text())
FAMILY = ["gcn_fix", "gin_fix", "graphsage", "gat", "abmil", "dsmil", "transmil", "clam_sb", "clam_mb",
          "clam_sb_inst", "clam_mb_inst", "acmil", "wikg", "defgnn"]
NAMES = {"gcn_fix": "GCN", "gin_fix": "GIN", "graphsage": "GraphSAGE", "gat": "GAT", "abmil": "ABMIL",
         "dsmil": "DSMIL", "transmil": "TransMIL", "clam_sb": "CLAM-SB", "clam_mb": "CLAM-MB",
         "clam_sb_inst": "CLAM-SB (inst.)", "clam_mb_inst": "CLAM-MB (inst.)", "acmil": "ACMIL",
         "wikg": "WiKG", "defgnn": "GigaPath-DefGNN"}
CLASSES = ["MEL", "MCT", "SCC", "PNST", "PLC", "TRB", "HIS"]
BLUE, ORANGE, GREEN, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1c", "#6b6a63", "#e4e3dc"

# ------------------------------------------------------------------ Figure 3
best, top3, final = ST["best"]["models"], ST["top3"]["models"], ST["final"]["models"]
FINAL_OF = {"defgnn": "defgnn_dual"}  # see the docstring
# ascending, drawn bottom-up; ties keep the order of Table 3 (FAMILY order, top to bottom)
models = sorted([m for m in FAMILY if m in best], key=lambda m: (best[m]["bacc"], -FAMILY.index(m)))
fig, ax = plt.subplots(figsize=(6.4, 0.46 * len(models) + 1.0), dpi=300)
y = np.arange(len(models))
for i, m in enumerate(models):
    b = best[m]
    ax.plot(b["bacc_ci95"], [i + 0.22] * 2, color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(b["bacc"], i + 0.22, "o", ms=6, color=BLUE, mec="white", mew=1.5, zorder=3)
    if m in top3:
        t = top3[m]
        ax.plot(t["bacc_ci95"], [i] * 2, color=ORANGE, lw=2, solid_capstyle="round", alpha=0.9)
        ax.plot(t["bacc"], i, "o", ms=6, mfc="white", mec=ORANGE, mew=1.8, zorder=3)
    fm = FINAL_OF.get(m, m)
    if fm in final:
        f = final[fm]
        ax.plot(f["bacc_ci95"], [i - 0.22] * 2, color=GREEN, lw=2, solid_capstyle="round", alpha=0.9)
        ax.plot(f["bacc"], i - 0.22, "D", ms=5, color=GREEN, mec="white", mew=1.2, zorder=3)
ax.set_yticks(y, [NAMES[m] for m in models], fontsize=7.5, color=INK)
for lab, m in zip(ax.get_yticklabels(), models):
    if m == "defgnn":
        lab.set_fontweight("bold")
ax.set_xlabel("Held-out balanced accuracy (350 WSIs; 95% CI)", fontsize=8, color=INK)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.tick_params(axis="x", labelsize=7, colors=MUTED)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.plot([], [], "o-", color=BLUE, ms=5, lw=2, label="best-validation checkpoint (primary)")
ax.plot([], [], "o-", color=ORANGE, mfc="white", ms=5, lw=2, label="latest-top-3 checkpoint")
ax.plot([], [], "D-", color=GREEN, ms=4.5, lw=2, label="final weights*")
ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.text(0.01, 0.0, "*GigaPath-DefGNN: re-run of the same configuration (the original run did not save final weights).",
         fontsize=6.5, color=MUTED, ha="left", va="top")
fig.tight_layout()
fig.savefig(OUT / "Figure_3_test_forest.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Figure 4
def confusion(m):
    cm = np.zeros((7, 7), int)
    for f in range(5):
        t = PRED[f"{m}/fold{f}/best"]["test"]
        for a, b in zip(t["y_true"], t["y_pred"]):
            cm[a, b] += 1
    return cm


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), dpi=300)
for ax, m in zip(axes, ("gat", "defgnn")):
    cm = confusion(m)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=50)
    for i in range(7):
        for j in range(7):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 30 else INK)
    ax.set_xticks(range(7), CLASSES, fontsize=6.5, rotation=45)
    ax.set_yticks(range(7), CLASSES, fontsize=6.5)
    ax.set_xlabel("Predicted class", fontsize=7.5)
    ax.set_title(f"{NAMES[m]} ({np.trace(cm)} / {cm.sum()} correct)", fontsize=8, color=INK, loc="left")
axes[0].set_ylabel("True class", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "Figure_4_confusion_test.png", bbox_inches="tight")
print("wrote Figure 3 and Figure 4 to", OUT)
