#!/usr/bin/env python3
"""How different are frozen Prov-GigaPath tile features at 20x vs 40x? (R2-3a)

Answers Reviewer 2 point 3(a) (1st revision, applsci-4523976). Both grids
start at the level-0 origin: a 20x tile at (X, Y) covers the 512 x 512
level-0 pixels that the 40x tiles at (X + dx, Y + dy), dx, dy in {0, 256},
cover. Each 20x embedding is therefore paired with the mean embedding of
the 40x tiles of the same region ("matched region"). Per slide:

* cosine(20x, matched 40x mean), against two references:
  - a random other region of the same slide (unmatched baseline);
  - two 40x tiles of the same region (natural 40x variability; regions
    with >= 2 kept 40x tiles);
  - one 40x tile vs the mean of the region's other 40x tiles (the like-for-
    like reference for "single embedding vs region mean");
* linear CKA between the matched 20x and 40x-mean embedding sets;
* region retrieval: fraction of 20x tiles whose most cosine-similar
  40x region mean in the slide is their own region (top-1);
* L2 norm ratio ||e20|| / ||e40 mean||.

Cohort level (slides with both magnifications): slide mean embeddings are
classified with logistic regression over the five outer CV folds (train =
train + val split of the fold, test = its test split), for all four
train/test magnification combinations, to test whether the two feature
spaces are interchangeable for slide-level subtyping.

Run:  python3 scripts/22_magnification_features.py
Out:  results/magnification/feature_shift.json (+ .md summary)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(ROOT))
from src.utils.env import output_dir  # noqa: E402
F40 = output_dir() / "features"
F20 = output_dir() / "features_20x"
SPLITS = output_dir() / "splits/cv5fold.csv"
CLASSES = ["MEL", "MCT", "SCC", "PNST", "PLC", "TRB", "HIS"]
MAX_TILES_CKA = 2000
SEED = 0


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path) as h:
        return h["coordinates"][:].astype(np.int64), h["embeddings"][:].astype(np.float32)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    hsic = np.linalg.norm(a.T @ b, "fro") ** 2
    return float(hsic / (np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")))


def slide_stats(sid: str, rng: np.random.Generator) -> dict | None:
    c40, e40 = load(F40 / f"{sid}.h5")
    c20, e20 = load(F20 / f"{sid}.h5")
    # region id of every 40x tile = its enclosing 512-px cell
    reg40 = {}
    for i, (x, y) in enumerate(c40):
        reg40.setdefault((x // 512, y // 512), []).append(i)
    keep20, mean40, pair_cos40, loo_cos40 = [], [], [], []
    for j, (x, y) in enumerate(c20):
        members = reg40.get((x // 512, y // 512))
        if not members:
            continue
        keep20.append(j)
        mean40.append(e40[members].mean(0))
        if len(members) >= 2:
            a, b = rng.choice(members, 2, replace=False)
            pair_cos40.append(float(unit(e40[a]) @ unit(e40[b])))
            others = [m for m in members if m != a]
            loo_cos40.append(float(unit(e40[a]) @ unit(e40[others].mean(0))))
    if len(keep20) < 10:
        return None
    E20 = e20[keep20]
    M40 = np.stack(mean40)
    u20, u40 = unit(E20), unit(M40)
    matched = np.sum(u20 * u40, axis=1)
    perm = rng.permutation(len(u40))
    perm = np.where(perm == np.arange(len(u40)), np.roll(perm, 1), perm)
    unmatched = np.sum(u20 * u40[perm], axis=1)
    # top-1 region retrieval, in row chunks (a full N x N matrix is ~2.5 GB
    # for the largest slides)
    best = np.concatenate([(u20[i:i + 4096] @ u40.T).argmax(1) for i in range(0, len(u20), 4096)])
    top1 = float(np.mean(best == np.arange(len(u20))))
    sub = rng.choice(len(E20), min(MAX_TILES_CKA, len(E20)), replace=False)
    return {
        "slide_id": sid,
        "n_20x": int(len(c20)), "n_40x": int(len(c40)),
        "n_matched": int(len(keep20)),
        "coverage_20x": float(len(keep20) / len(c20)),
        "cos_matched_median": float(np.median(matched)),
        "cos_unmatched_median": float(np.median(unmatched)),
        "cos_40x_same_region_median": float(np.median(pair_cos40)) if pair_cos40 else None,
        "cos_40x_vs_region_mean_loo_median": float(np.median(loo_cos40)) if loo_cos40 else None,
        "cka_linear": linear_cka(E20[sub], M40[sub]),
        "retrieval_top1": top1,
        "norm_ratio_median": float(np.median(np.linalg.norm(E20, axis=1) / np.linalg.norm(M40, axis=1))),
        "mean40": e40.mean(0).tolist(),
        "mean20": e20.mean(0).tolist(),
    }


def probe(rows: list[dict], splits: pd.DataFrame) -> dict:
    """Slide-mean logistic regression over the outer CV folds."""
    by = {r["slide_id"]: r for r in rows}
    y_of = splits.drop_duplicates("slide_id").set_index("slide_id")["tumor_class"].map(CLASSES.index)
    out = {}
    for tr_mag in ("40", "20"):
        for te_mag in ("40", "20"):
            ys, ps = [], []
            for fold in range(5):
                f = splits[splits.fold == fold]
                tr = [s for s in f[f.split != "test"].slide_id if s in by]
                te = [s for s in f[f.split == "test"].slide_id if s in by]
                if not tr or not te or len(set(y_of[tr])) < 2:
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
                clf.fit(np.array([by[s][f"mean{tr_mag}"] for s in tr]), y_of[tr].values)
                ps += clf.predict(np.array([by[s][f"mean{te_mag}"] for s in te])).tolist()
                ys += y_of[te].tolist()
            out[f"train{tr_mag}x_test{te_mag}x"] = {
                "balanced_accuracy": float(balanced_accuracy_score(ys, ps)) if ys else None,
                "n_test": len(ys)}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "results/magnification/feature_shift.json")
    args = p.parse_args()
    rng = np.random.default_rng(SEED)
    splits = pd.read_csv(SPLITS)
    sids = sorted(s.stem for s in F20.glob("*.h5") if (F40 / s.name).is_file())
    rows = [r for s in sids if (r := slide_stats(s, rng)) is not None]
    keys = ["coverage_20x", "cos_matched_median", "cos_unmatched_median",
            "cos_40x_same_region_median", "cos_40x_vs_region_mean_loo_median", "cka_linear", "retrieval_top1", "norm_ratio_median"]
    summ = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
        summ[k] = {"median": float(np.median(v)), "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}
    per_class = {c: {k: float(np.median([r[k] for r in rows if r["slide_id"].startswith(c + "_") and r[k] is not None]))
                     for k in ("cos_matched_median", "cka_linear", "retrieval_top1")}
                 for c in CLASSES if any(r["slide_id"].startswith(c + "_") for r in rows)}
    result = {"n_slides": len(rows), "summary": summ, "per_class": per_class,
              "probe": probe(rows, splits),
              "per_slide": [{k: v for k, v in r.items() if not k.startswith("mean")} for r in rows]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    lines = [f"# 20x vs 40x Prov-GigaPath features ({len(rows)} slides)", "",
             "| Metric (per-slide median, then median over slides) | Median [IQR] |", "|---|---|"]
    lines += [f"| {k} | {s['median']:.3f} [{s['iqr'][0]:.3f}, {s['iqr'][1]:.3f}] |" for k, s in summ.items()]
    lines += ["", "| Slide-mean probe (outer CV) | Balanced accuracy | n |", "|---|---|---|"]
    lines += [f"| {k} | {v['balanced_accuracy']:.3f} | {v['n_test']} |" if v["balanced_accuracy"] is not None
              else f"| {k} | — | 0 |" for k, v in result["probe"].items()]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
