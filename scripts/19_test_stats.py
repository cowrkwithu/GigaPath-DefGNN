#!/usr/bin/env python3
"""Statistics on the pooled out-of-fold test predictions from 18_test_eval.py.

The five per-fold test splits partition CATCH, so every WSI has exactly one
test prediction per (model, rule). This moves the unit of analysis from the
five folds (df = 4) to 350 WSIs from 282 patients:

* pooled balanced accuracy, macro / weighted F1, Cohen kappa, macro AUROC
* per-fold test balanced accuracy (mean ± population std), as in the paper
* for each checkpoint rule of 18_test_eval.py (best, top3, final)
* 95 % CI by patient-clustered bootstrap (patients resampled with
  replacement, all of a patient's WSIs kept together)
* GigaPath-DefGNN vs each baseline: paired patient-clustered bootstrap CI for
  Δ balanced accuracy and exact McNemar test on per-WSI correctness,
  Holm-adjusted across the baselines
* per-class F1 with bootstrap CIs, and per-class McNemar restricted to the
  WSIs of that class (Reviewer 2, point 4)

Run:  python3 scripts/19_test_stats.py
Out:  results/test_eval/test_stats.json, results/test_eval/test_tables.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(ROOT))
from src.utils.env import output_dir  # noqa: E402
PRED = ROOT / "results/test_eval/predictions.json"
SPLITS = output_dir() / "splits/cv5fold.csv"
CLASSES = ["MEL", "MCT", "SCC", "PNST", "PLC", "TRB", "HIS"]
#: Comparison family of the revision (Holm correction is applied over its
#: baselines): GCN and GIN are the retrained runs (the published ones were
#: stopped early by the NaNGuard bug), plus the four baselines added in revision.
FAMILY = ["gcn_fix", "gin_fix", "graphsage", "gat", "abmil", "dsmil", "transmil",
          "clam_sb", "clam_mb", "clam_sb_inst", "clam_mb_inst", "acmil", "wikg", "defgnn"]
#: Reported for reference only (not in the Holm family): the published,
#: early-stopped GCN / GIN runs, the GIN stabilization variants, and the
#: graph-construction / 20x variants.
EXTRA = ["gcn", "gin", "gin_fp32", "gin_layernorm", "gin_mean", "gat_spatial", "gat_feature", "gat_featknn", "defgnn_spatial", "defgnn_feature", "defgnn_featknn",
         "defgnn_dual"]
ORDER = FAMILY + EXTRA
#: 20x models vs their 40x counterparts (Reviewer 2, point 3). A slide can be
#: excluded at 20x (too few tissue tiles), so these are compared separately on
#: the WSIs both magnifications share; {ref} is REFERENCE[rule].
MAG_PAIRS = [("gat_20x", "gat"), ("defgnn_20x", "{ref}")]
PROPOSED = "defgnn"
#: Reference model of the comparisons per checkpoint rule. The published
#: GigaPath-DefGNN run never saved its final weights, so the final-weights
#: rule compares against defgnn_dual, a rerun of the same configuration.
REFERENCE = {"best": PROPOSED, "top3": PROPOSED, "final": "defgnn_dual"}
N_BOOT = 10_000
SEED = 20260926


def pooled(results: dict, model: str, rule: str) -> pd.DataFrame | None:
    rows = []
    for fold in range(5):
        r = results.get(f"{model}/fold{fold}/{rule}")
        if r is None:
            return None
        t = r["test"]
        for sid, y, p, prob in zip(t["slide_id"], t["y_true"], t["y_pred"], t["y_prob"]):
            rows.append({"slide_id": sid, "fold": fold, "y": y, "pred": p, "prob": prob})
    return pd.DataFrame(rows).set_index("slide_id").sort_index()


def metrics(y, pred, prob) -> dict:
    out = {
        "bacc": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=range(7), zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", labels=range(7), zero_division=0),
        "kappa": cohen_kappa_score(y, pred),
    }
    try:
        out["macro_auroc"] = roc_auc_score(y, np.asarray(prob), multi_class="ovr",
                                           average="macro", labels=list(range(7)))
    except ValueError:
        out["macro_auroc"] = float("nan")
    return out


def cluster_indices(patients: np.ndarray, rng: np.random.Generator, n_boot: int):
    """Yield row-index arrays, resampling patients with replacement."""
    uniq, inv = np.unique(patients, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[i] for i in pick])


def ci(values) -> list[float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    b = int(np.sum(a_correct & ~b_correct))   # proposed right, baseline wrong
    c = int(np.sum(~a_correct & b_correct))   # proposed wrong, baseline right
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"proposed_only_correct": b, "baseline_only_correct": c, "p": float(p)}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def magnification(results: dict, patient: pd.Series, out: dict, md: list[str]) -> None:
    """20x vs 40x of the same model, paired on the WSIs present at both."""
    rows = []
    for rule in ("best", "top3", "final"):
        pvals, comp = {}, {}
        for m20, m40 in MAG_PAIRS:
            m40 = m40.format(ref=REFERENCE[rule])
            a, b = pooled(results, m20, rule), pooled(results, m40, rule)
            if a is None or b is None:
                continue
            ids = a.index.intersection(b.index)
            excluded = sorted(set(b.index) - set(ids))
            a, b = a.loc[ids], b.loc[ids]
            assert (a["y"].values == b["y"].values).all(), m20
            y, pa, pb = a["y"].values, a["pred"].values, b["pred"].values
            rng = np.random.default_rng(SEED)
            d_boot = [balanced_accuracy_score(y[i], pa[i]) - balanced_accuracy_score(y[i], pb[i])
                      for i in cluster_indices(patient.loc[ids].values, rng, N_BOOT)]
            comp[m20] = {
                "vs": m40, "n_wsi": len(ids),
                "excluded_at_20x": excluded,
                "bacc_20x": balanced_accuracy_score(y, pa), "bacc_40x": balanced_accuracy_score(y, pb),
                "macro_f1_20x": f1_score(y, pa, average="macro", labels=range(7), zero_division=0),
                "macro_f1_40x": f1_score(y, pb, average="macro", labels=range(7), zero_division=0),
                "delta_bacc": balanced_accuracy_score(y, pa) - balanced_accuracy_score(y, pb),
                "delta_bacc_ci95": ci(d_boot),
                "mcnemar": mcnemar_exact(pa == y, pb == y),
            }
            pvals[m20] = comp[m20]["mcnemar"]["p"]
        for m, p_adj in holm(pvals).items():
            comp[m]["mcnemar"]["p_holm"] = p_adj
        if comp:
            out.setdefault("magnification", {})[rule] = comp
            for m20, c in comp.items():
                rows.append(f"| {rule} | {m20} vs {c['vs']} | {c['n_wsi']} | {c['bacc_20x']:.4f} "
                            f"| {c['bacc_40x']:.4f} | {c['delta_bacc']:+.4f} [{c['delta_bacc_ci95'][0]:+.4f}, "
                            f"{c['delta_bacc_ci95'][1]:+.4f}] | {c['mcnemar']['proposed_only_correct']}/"
                            f"{c['mcnemar']['baseline_only_correct']} | {c['mcnemar']['p']:.3g} "
                            f"({c['mcnemar']['p_holm']:.3g}) |")
    if rows:
        md += ["## Magnification: 20x vs 40x (paired on shared WSIs)", "",
               "| Rule | Pair | n WSI | bacc 20x | bacc 40x | Δ (20x − 40x) [95% CI] "
               "| McNemar 20x/40x only-correct | p (Holm over pairs) |",
               "|---|---|---|---|---|---|---|---|", *rows, ""]


def main() -> None:
    results = json.loads(PRED.read_text())
    patient = pd.read_csv(SPLITS).drop_duplicates("slide_id").set_index("slide_id")["patient_id"]
    out: dict = {"n_boot": N_BOOT, "bootstrap": "patient-clustered", "rules": {}}
    md: list[str] = []

    for rule in ("best", "top3", "final"):
        frames = {m: pooled(results, m, rule) for m in ORDER}
        frames = {m: f for m, f in frames.items() if f is not None}
        if not frames:
            continue
        proposed = REFERENCE[rule]
        ref = frames.get(proposed, next(iter(frames.values())))
        ids = ref.index
        assert len(ids) == 350 and ids.is_unique, "test folds must partition 350 WSIs"
        for m, f in frames.items():
            assert f.index.equals(ids) and (f["y"].values == ref["y"].values).all(), m
        pats = patient.loc[ids].values
        y = ref["y"].values

        # Bootstrap resamples are shared by all models so paired Δs are coherent.
        rng = np.random.default_rng(SEED)
        boots = list(cluster_indices(pats, rng, N_BOOT))

        per_model = {}
        for m, f in frames.items():
            pred, prob = f["pred"].values, np.stack(f["prob"].values)
            point = metrics(y, pred, prob)
            fold_bacc = [balanced_accuracy_score(f[f.fold == k]["y"], f[f.fold == k]["pred"])
                         for k in range(5)]
            b_bacc = [balanced_accuracy_score(y[i], pred[i]) for i in boots]
            f1_pc = f1_score(y, pred, average=None, labels=range(7), zero_division=0)
            b_f1 = np.array([f1_score(y[i], pred[i], average=None, labels=range(7),
                                      zero_division=0) for i in boots])
            per_model[m] = {
                **point,
                "bacc_ci95": ci(b_bacc),
                "fold_bacc": fold_bacc,
                "fold_bacc_mean": float(np.mean(fold_bacc)),
                "fold_bacc_pstd": float(np.std(fold_bacc)),
                "per_class_f1": dict(zip(CLASSES, map(float, f1_pc))),
                "per_class_f1_ci95": {c: ci(b_f1[:, j]) for j, c in enumerate(CLASSES)},
                "n_correct": int((pred == y).sum()),
            }

        has_prop = proposed in frames
        prop = frames[proposed]["pred"].values if has_prop else None
        prop_ok = prop == y if has_prop else None
        comparisons, pvals = {}, {}
        for m, f in frames.items():
            if m == proposed or not has_prop:
                continue
            base = f["pred"].values
            base_ok = base == y
            d_boot = [balanced_accuracy_score(y[i], prop[i]) - balanced_accuracy_score(y[i], base[i])
                      for i in boots]
            per_class = {}
            for j, c in enumerate(CLASSES):
                mask = y == j
                per_class[c] = mcnemar_exact(prop_ok[mask], base_ok[mask])
            comparisons[m] = {
                "delta_bacc": per_model[proposed]["bacc"] - per_model[m]["bacc"],
                "delta_bacc_ci95": ci(d_boot),
                "p_boot_delta_le_0": float(np.mean(np.asarray(d_boot) <= 0)),
                "mcnemar": mcnemar_exact(prop_ok, base_ok),
                "per_class_mcnemar": per_class,
            }
            if m in FAMILY:
                pvals[m] = comparisons[m]["mcnemar"]["p"]
        for m, p_adj in holm(pvals).items():
            comparisons[m]["mcnemar"]["p_holm"] = p_adj
        for m in comparisons:
            comparisons[m]["mcnemar"].setdefault("p_holm", None)
        holm_family = [m for m in FAMILY if m in pvals]

        out["rules"][rule] = {"models": per_model, "reference": proposed,
                              f"{proposed}_vs": comparisons, "holm_family": holm_family}

        md += [f"## Checkpoint rule: {rule}", "", f"Reference model (Δ, McNemar): {proposed}", "",
               "| Model | pooled bacc [95% CI] | fold mean ± pop-std | macro F1 | κ | macro AUROC "
               "| Δ bacc vs proposed [95% CI] | McNemar b/c | p (Holm) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for m in ORDER:
            if m not in per_model:
                continue
            s = per_model[m]
            row = (f"| {m} | {s['bacc']:.4f} [{s['bacc_ci95'][0]:.4f}, {s['bacc_ci95'][1]:.4f}] "
                   f"| {s['fold_bacc_mean']:.4f} ± {s['fold_bacc_pstd']:.4f} | {s['macro_f1']:.4f} "
                   f"| {s['kappa']:.4f} | {s['macro_auroc']:.4f} ")
            if m == proposed or m not in comparisons:
                row += "| — | — | — |"
            else:
                c = comparisons[m]
                row += (f"| {c['delta_bacc']:+.4f} [{c['delta_bacc_ci95'][0]:+.4f}, "
                        f"{c['delta_bacc_ci95'][1]:+.4f}] | {c['mcnemar']['proposed_only_correct']}/"
                        f"{c['mcnemar']['baseline_only_correct']} | {c['mcnemar']['p']:.3g} "
                        + (f"({c['mcnemar']['p_holm']:.3g}) |" if c['mcnemar']['p_holm'] is not None else "(not in family) |"))
            md.append(row)
        md += ["", "Per-class F1 (pooled 350 WSIs):", "",
               "| Model | " + " | ".join(CLASSES) + " |", "|---" * (len(CLASSES) + 1) + "|"]
        for m in ORDER:
            if m in per_model:
                md.append(f"| {m} | " + " | ".join(f"{per_model[m]['per_class_f1'][c]:.3f}"
                                                   for c in CLASSES) + " |")
        md.append("")

    magnification(results, patient, out, md)

    (ROOT / "results/test_eval/test_stats.json").write_text(json.dumps(out, indent=1))
    (ROOT / "results/test_eval/test_tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
