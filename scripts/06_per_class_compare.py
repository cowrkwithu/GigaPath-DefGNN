#!/usr/bin/env python3
"""Re-run inference on per-fold val splits and emit per-class metrics + figures.

For each of the 5 patient-level CV folds:
  * Find the best v1 GAT checkpoint at
    /data/cia_outputs/checkpoints/exp2/dual_edge/vetgigagraph/fold_<i>/
    (or the model_name'd subdir — script discovers it).
  * Find the best v2 deformable checkpoint at
    results/deformable/fold_<i>/checkpoints/.
  * Build the matching model (v1 VetGigaGraph.from_config / v2
    build_deformable_model), load state_dict, run val_dataloader, collect
    (y_true, y_pred, y_prob[7]).

Aggregate across folds and write:
  paper/tables/T4_per_class_metrics.md           — markdown table
  paper/figures/T4_per_class_metrics.csv          — raw per-fold numbers
  paper/figures/fig4_confusion_matrices.png       — side-by-side panel
  paper/figures/fig4_per_class_data.json          — values consumed by the figure
  paper/figures/scripts/fig4_confusion_matrices.py — regenerator (writes png from json)

Usage:
    python3 scripts/06_per_class_compare.py
    python3 scripts/06_per_class_compare.py --folds 0 1 2          # subset
    python3 scripts/06_per_class_compare.py --v1-only               # skip v2 re-inference
    python3 scripts/06_per_class_compare.py --max-tiles-per-wsi 30000  # mem cap
"""

from __future__ import annotations

import os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
# Phase B unification (2026-05-31): v1 src/ is now under this repo's src/;
# see scripts/04b_train_deformable.py for the same single-root pattern.
_sys.path.insert(0, str(_ROOT))
from src.utils.env import output_dir, portable_path  # noqa: E402
_sys.path.insert(0, str(_ROOT / "src"))

import argparse                                             # noqa: E402
import json                                                 # noqa: E402
import logging                                              # noqa: E402
import re                                                   # noqa: E402
import statistics                                           # noqa: E402
from collections import defaultdict                         # noqa: E402

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from omegaconf import OmegaConf                             # noqa: E402

# v1
from src.utils.config import load_config                    # noqa: E402
from src.utils.seed import set_global_seed                  # noqa: E402
from src.training import GraphSlideDataModule, vetgigagraph_forward_fn  # noqa: E402
from src.models import VetGigaGraph                         # noqa: E402
from src.models.gigapath_slide import production_slide_backbone_loader  # noqa: E402

# v2
from deformable_attention import build_deformable_model     # noqa: E402

logger = logging.getLogger(__name__)

CLASSES = ["MEL", "MCT", "SCC", "PNST", "PLC", "TRB", "HIS"]   # locked CATCH order
NUM_CLASSES = 7
LOCKED_SEEDS = [42, 123, 456, 789, 1024]

V1_DEFAULT_CONFIG = _ROOT / "configs" / "default.yaml"
V2_OVERRIDE_CONFIG = _ROOT / "configs" / "experiment_deformable.yaml"
V1_CKPT_ROOT = output_dir() / "checkpoints/exp2/dual_edge/vetgigagraph"
V2_CKPT_ROOT = _ROOT / "results" / "deformable"

# Best-ckpt filename pattern: "epoch{NNN}-val_bacc{X.XXXX}[-vN].ckpt"
_CKPT_PAT = re.compile(
    r"epoch(?P<epoch>\d+)-val_bacc(?P<bacc>\d+\.\d+)(?P<rev>-v\d+)?\.ckpt$"
)


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #


def find_best_ckpt(directory: _Path) -> _Path | None:
    """Pick highest val_bacc (then latest epoch, then highest -vN rev)."""
    if not directory.is_dir():
        return None
    best: tuple[float, int, int, _Path] | None = None  # (bacc, epoch, rev, path)
    for p in directory.glob("*.ckpt"):
        m = _CKPT_PAT.search(p.name)
        if not m:
            continue
        bacc = float(m.group("bacc"))
        epoch = int(m.group("epoch"))
        rev = int(m.group("rev").lstrip("-v")) if m.group("rev") else 0
        key = (bacc, epoch, rev)
        if best is None or key > best[:3]:
            best = (*key, p)
    return best[3] if best else None


def v1_fold_ckpt(fold: int) -> _Path | None:
    """Best v1 production-GAT checkpoint for the fold.

    Layout under /data/cia_outputs/checkpoints/exp2/dual_edge/vetgigagraph/:
      - Preferred: ``fold_<i>.ckpt`` (canonical per-fold ckpt the v1
        paper numbers were computed from; matches ``fold_<i>_metrics.json``).
      - Fallback: best ``epoch{NNN}-val_bacc{X.XXXX}[-vN].ckpt`` (these
        are top-3-by-epoch but mixed across folds in this directory, so
        used only as a last resort).
    """
    canonical = V1_CKPT_ROOT / f"fold_{fold}.ckpt"
    if canonical.is_file():
        return canonical
    cand = V1_CKPT_ROOT / f"fold_{fold}"
    if cand.is_dir():
        ckpt = find_best_ckpt(cand)
        if ckpt:
            return ckpt
    return find_best_ckpt(V1_CKPT_ROOT)


def v2_fold_ckpt(fold: int) -> _Path | None:
    return find_best_ckpt(V2_CKPT_ROOT / f"fold_{fold}" / "checkpoints")


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


def load_v1_config():
    return load_config(V1_DEFAULT_CONFIG)


def load_v2_config():
    base = load_config(V1_DEFAULT_CONFIG)
    override = OmegaConf.load(V2_OVERRIDE_CONFIG)
    return OmegaConf.merge(base, override)


# --------------------------------------------------------------------------- #
# State-dict load (strip Lightning "model." prefix)
# --------------------------------------------------------------------------- #


def load_state_dict_strip_prefix(model: torch.nn.Module, ckpt_path: _Path) -> None:
    """Load a Lightning-saved state_dict into a bare model."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = blob.get("state_dict", blob)
    model_sd: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("model."):
            model_sd[k[len("model."):]] = v
        else:
            model_sd[k] = v
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        logger.warning("[load] %s missing keys (first 5): %s …",
                       ckpt_path.name, missing[:5])
    if unexpected:
        logger.warning("[load] %s unexpected keys (first 5): %s …",
                       ckpt_path.name, unexpected[:5])


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


def run_inference(model: torch.nn.Module, dataloader, device: str = "cuda"
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true [N], y_pred [N], y_prob [N, NUM_CLASSES])."""
    model.eval()
    model.to(device)
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            # `batch` is a PyG `Data` (batch_size=1, _single_graph_collate).
            batch = batch.to(device)
            args, target = vetgigagraph_forward_fn(batch)
            target = target.to(device)
            out = model(*args)
            logits = out[0] if isinstance(out, tuple) else out
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            prob = torch.softmax(logits, dim=-1)
            pred = prob.argmax(dim=-1)
            ys.append(int(target.item() if target.ndim == 0 else target[0].item()))
            preds.append(int(pred[0].item()))
            probs.append(prob[0].float().cpu().numpy())
    return np.array(ys), np.array(preds), np.stack(probs)


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: np.ndarray) -> dict:
    """Per-class F1 + AUROC + counts. AUROC is one-vs-rest; gracefully NaN
    when a class has fewer than 2 positives in this fold's validation set
    (CATCH per-fold val sets have only ~5-6 WSIs per class)."""
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
    f1 = f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                  average=None, zero_division=0).tolist()
    auroc = []
    for c in range(NUM_CLASSES):
        positives = (y_true == c)
        if positives.sum() < 1 or positives.sum() == len(y_true):
            auroc.append(float("nan"))
        else:
            try:
                auroc.append(float(roc_auc_score(positives.astype(int), y_prob[:, c])))
            except ValueError:
                auroc.append(float("nan"))
    counts = np.bincount(y_true, minlength=NUM_CLASSES).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()
    return {
        "f1_per_class": f1,
        "auroc_per_class": auroc,
        "val_n_per_class": counts,
        "confusion_matrix": cm,
    }


# --------------------------------------------------------------------------- #
# Per-fold pipeline
# --------------------------------------------------------------------------- #


def evaluate_fold(model_kind: str, fold: int, ckpt_path: _Path,
                  device: str = "cuda") -> dict:
    """Build + load + infer + per-class metrics for one (model, fold)."""
    seed = LOCKED_SEEDS[fold]
    set_global_seed(seed)

    if model_kind == "v1_gat":
        cfg = load_v1_config()
        fusion_strategy = str(cfg["model"]["fusion"]["strategy"])
        slide_loader = production_slide_backbone_loader if fusion_strategy != "gnn_only" else None
        model = VetGigaGraph.from_config(cfg, slide_backbone_loader=slide_loader)
    elif model_kind == "v2_deformable":
        cfg = load_v2_config()
        fusion_strategy = str(cfg["model"]["fusion"]["strategy"])
        slide_loader = production_slide_backbone_loader if fusion_strategy != "gnn_only" else None
        model = build_deformable_model(cfg, slide_backbone_loader=slide_loader)
    else:
        raise ValueError(f"unknown model_kind={model_kind!r}")

    load_state_dict_strip_prefix(model, ckpt_path)

    splits_csv = _Path(cfg["paths"]["splits"]) / "cv5fold.csv"
    graphs_root = _Path(cfg["paths"]["graphs"]) / str(cfg["graph"]["type"])
    dm = GraphSlideDataModule(
        splits_csv=splits_csv,
        graphs_root=graphs_root,
        fold=fold,
        num_workers=0,            # eval doesn't need workers
        attach_tile_labels=False,
    )
    dm.setup(stage="fit")
    val_loader = dm.val_dataloader()

    y_true, y_pred, y_prob = run_inference(model, val_loader, device=device)
    metrics = per_class_metrics(y_true, y_pred, y_prob)
    metrics.update({
        "model_kind": model_kind,
        "fold": fold,
        "seed": seed,
        "ckpt": portable_path(ckpt_path),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "n_val": int(len(y_true)),
    })
    # Free GPU memory between fold/model rotations
    del model
    torch.cuda.empty_cache()
    return metrics


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate(folds_results: list[dict]) -> dict:
    """Aggregate per-fold per-class metrics into mean ± std across folds."""
    n_folds = len(folds_results)
    out_f1 = {c: [] for c in range(NUM_CLASSES)}
    out_auroc = {c: [] for c in range(NUM_CLASSES)}
    cm_sum = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for r in folds_results:
        for c in range(NUM_CLASSES):
            out_f1[c].append(r["f1_per_class"][c])
            v = r["auroc_per_class"][c]
            if v == v:  # not NaN
                out_auroc[c].append(v)
        cm_sum += np.array(r["confusion_matrix"], dtype=int)
    def _meanstd(xs):
        if not xs:
            return (float("nan"), float("nan"), 0)
        return (statistics.mean(xs),
                statistics.pstdev(xs) if len(xs) > 1 else 0.0,
                len(xs))
    per_class = []
    for c in range(NUM_CLASSES):
        f1_mean, f1_std, n_f1 = _meanstd(out_f1[c])
        au_mean, au_std, n_au = _meanstd(out_auroc[c])
        per_class.append({
            "class": CLASSES[c],
            "f1_mean": f1_mean, "f1_std": f1_std, "f1_n_folds": n_f1,
            "auroc_mean": au_mean, "auroc_std": au_std, "auroc_n_folds": n_au,
        })
    return {
        "n_folds": n_folds,
        "per_class": per_class,
        "confusion_matrix_sum": cm_sum.tolist(),
        "classes": CLASSES,
    }


# --------------------------------------------------------------------------- #
# Table T4 and Figure 4 writers
# --------------------------------------------------------------------------- #


def write_t4_markdown(v1_agg: dict, v2_agg: dict, out: _Path) -> None:
    lines = [
        "# Table T4 — Per-class F1 and AUROC (mean ± std across 5 folds)",
        "",
        "Per-class F1 scores and one-vs-rest AUROC for v1 GAT (baseline) and",
        "v2 deformable attention on the per-fold patient-level validation set.",
        "Fold n ≈ 38 WSIs (≈ 5–6 per class); AUROC NaN when a class is absent",
        "from a fold's val set.",
        "",
        "| Class | v1 GAT F1 | v2 Deformable F1 | Δ F1 | v1 GAT AUROC | v2 Deformable AUROC | Δ AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for v1_c, v2_c in zip(v1_agg["per_class"], v2_agg["per_class"]):
        df1 = v2_c["f1_mean"] - v1_c["f1_mean"]
        dau = v2_c["auroc_mean"] - v1_c["auroc_mean"]
        lines.append(
            f"| {v1_c['class']} "
            f"| {v1_c['f1_mean']:.3f} ± {v1_c['f1_std']:.3f} "
            f"| {v2_c['f1_mean']:.3f} ± {v2_c['f1_std']:.3f} "
            f"| {df1:+.3f} "
            f"| {v1_c['auroc_mean']:.3f} ± {v1_c['auroc_std']:.3f} "
            f"| {v2_c['auroc_mean']:.3f} ± {v2_c['auroc_std']:.3f} "
            f"| {dau:+.3f} |"
        )
    # Aggregate row (mean over classes)
    def _meancol(rows, key):
        vs = [r[key] for r in rows if r[key] == r[key]]
        return statistics.mean(vs) if vs else float("nan")
    lines += [
        "",
        f"**Macro-mean across 7 classes**: v1 F1 = {_meancol(v1_agg['per_class'], 'f1_mean'):.3f}, "
        f"v2 F1 = {_meancol(v2_agg['per_class'], 'f1_mean'):.3f}; "
        f"v1 AUROC = {_meancol(v1_agg['per_class'], 'auroc_mean'):.3f}, "
        f"v2 AUROC = {_meancol(v2_agg['per_class'], 'auroc_mean'):.3f}.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    logger.info("[t4] wrote %s", out)


def write_t4_csv(v1_agg: dict, v2_agg: dict, folds_v1: list[dict],
                 folds_v2: list[dict], out: _Path) -> None:
    import csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "class_idx", "class", "f1", "auroc", "n_val"])
        for fr in folds_v1:
            for c in range(NUM_CLASSES):
                w.writerow(["v1_gat", fr["fold"], c, CLASSES[c],
                            fr["f1_per_class"][c], fr["auroc_per_class"][c],
                            fr["val_n_per_class"][c]])
        for fr in folds_v2:
            for c in range(NUM_CLASSES):
                w.writerow(["v2_deformable", fr["fold"], c, CLASSES[c],
                            fr["f1_per_class"][c], fr["auroc_per_class"][c],
                            fr["val_n_per_class"][c]])
    logger.info("[t4] wrote %s", out)


def write_fig4(v1_agg: dict, v2_agg: dict, out_png: _Path, out_json: _Path,
               out_script: _Path) -> None:
    """Side-by-side normalized confusion matrices."""
    data = {
        "v1": {
            "cm": v1_agg["confusion_matrix_sum"],
            "classes": v1_agg["classes"],
            "label": "v1 GAT (baseline)",
        },
        "v2": {
            "cm": v2_agg["confusion_matrix_sum"],
            "classes": v2_agg["classes"],
            "label": "v2 Deformable Attention",
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2))
    _render_fig4(data, out_png)
    out_script.parent.mkdir(parents=True, exist_ok=True)
    out_script.write_text(_FIG4_SCRIPT_TEMPLATE)
    logger.info("[fig4] wrote %s + %s + %s", out_png, out_json, out_script)


def _render_fig4(data: dict, out_png: _Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[fig4] matplotlib not installed; skip PNG render. "
                       "Data saved to JSON; run paper/figures/scripts/"
                       "fig4_confusion_matrices.py on a machine with matplotlib.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, key in zip(axes, ("v1", "v2")):
        cm = np.array(data[key]["cm"], dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(data[key]["label"])
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(data[key]["classes"], rotation=30)
        ax.set_yticklabels(data[key]["classes"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                t = int(cm[i, j])
                if t > 0:
                    ax.text(j, i, str(t), ha="center", va="center",
                            color="white" if cm_norm[i, j] > 0.5 else "black",
                            fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Figure 4 — Per-class confusion matrices, summed over 5 folds",
                 y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


_FIG4_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Regenerate Figure 4 — side-by-side per-class confusion matrices.

Reads fig4_per_class_data.json and writes fig4_confusion_matrices.png.
Run on any machine with matplotlib; no GPU / data access required.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent.parent
DATA = HERE / "fig4_per_class_data.json"
OUT  = HERE / "fig4_confusion_matrices.png"

data = json.loads(DATA.read_text())
classes = data["v1"]["classes"]
N = len(classes)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, key in zip(axes, ("v1", "v2")):
    cm = np.array(data[key]["cm"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(data[key]["label"])
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(classes, rotation=30)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    for i in range(N):
        for j in range(N):
            t = int(cm[i, j])
            if t > 0:
                ax.text(j, i, str(t), ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black",
                        fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.045)
fig.suptitle("Figure 4 — Per-class confusion matrices, summed over 5 folds",
             y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"wrote {OUT}")
'''


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   choices=[0, 1, 2, 3, 4])
    p.add_argument("--device", default="cuda")
    p.add_argument("--v1-only", action="store_true")
    p.add_argument("--v2-only", action="store_true")
    p.add_argument("--out-table", type=_Path,
                   default=_ROOT / "paper" / "tables" / "T4_per_class_metrics.md")
    p.add_argument("--out-csv", type=_Path,
                   default=_ROOT / "paper" / "figures" / "T4_per_class_metrics.csv")
    p.add_argument("--out-fig", type=_Path,
                   default=_ROOT / "paper" / "figures" / "fig4_confusion_matrices.png")
    p.add_argument("--out-fig-json", type=_Path,
                   default=_ROOT / "paper" / "figures" / "fig4_per_class_data.json")
    p.add_argument("--out-fig-script", type=_Path,
                   default=_ROOT / "paper" / "figures" / "scripts"
                            / "fig4_confusion_matrices.py")
    p.add_argument("--per-fold-cache", type=_Path,
                   default=_ROOT / "results" / "per_class_per_fold.json",
                   help="Cache per-fold raw results to skip GPU re-inference on reruns.")
    p.add_argument("--no-cache", action="store_true",
                   help="Force re-inference even if cache exists.")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()

    # Cache hit? — keeps reruns cheap (e.g. when refining table/figure layout)
    cached: dict | None = None
    if args.per_fold_cache.exists() and not args.no_cache:
        try:
            cached = json.loads(args.per_fold_cache.read_text())
            logger.info("[cache] loaded %s — pass --no-cache to force re-inference",
                        args.per_fold_cache)
        except Exception as e:
            logger.warning("[cache] failed to load %s: %s", args.per_fold_cache, e)
            cached = None

    folds_v1: list[dict] = []
    folds_v2: list[dict] = []

    if not args.v2_only:
        for fold in args.folds:
            if cached and any(r["fold"] == fold for r in cached.get("v1", [])):
                hit = next(r for r in cached["v1"] if r["fold"] == fold)
                folds_v1.append(hit)
                logger.info("[v1] fold %d cached → bal_acc=%.4f",
                            fold, _bacc(hit))
                continue
            ckpt = v1_fold_ckpt(fold)
            if not ckpt:
                logger.warning("[v1] fold %d: no checkpoint found under %s",
                               fold, V1_CKPT_ROOT)
                continue
            logger.info("[v1] fold %d: %s", fold, ckpt.name)
            folds_v1.append(evaluate_fold("v1_gat", fold, ckpt, device=args.device))
            logger.info("[v1] fold %d done → bal_acc=%.4f",
                        fold, _bacc(folds_v1[-1]))

    if not args.v1_only:
        for fold in args.folds:
            if cached and any(r["fold"] == fold for r in cached.get("v2", [])):
                hit = next(r for r in cached["v2"] if r["fold"] == fold)
                folds_v2.append(hit)
                logger.info("[v2] fold %d cached → bal_acc=%.4f",
                            fold, _bacc(hit))
                continue
            ckpt = v2_fold_ckpt(fold)
            if not ckpt:
                logger.warning("[v2] fold %d: no checkpoint under %s",
                               fold, V2_CKPT_ROOT / f"fold_{fold}" / "checkpoints")
                continue
            logger.info("[v2] fold %d: %s", fold, ckpt.name)
            folds_v2.append(evaluate_fold("v2_deformable", fold, ckpt,
                                          device=args.device))
            logger.info("[v2] fold %d done → bal_acc=%.4f",
                        fold, _bacc(folds_v2[-1]))

    # Persist raw cache for future cheap reruns
    args.per_fold_cache.parent.mkdir(parents=True, exist_ok=True)
    args.per_fold_cache.write_text(json.dumps({
        "v1": folds_v1, "v2": folds_v2, "classes": CLASSES,
    }, indent=2))
    logger.info("[cache] wrote %s", args.per_fold_cache)

    if not folds_v1 or not folds_v2:
        logger.error("Need at least one fold for both v1 and v2; aborting outputs.")
        return 1

    v1_agg = aggregate(folds_v1)
    v2_agg = aggregate(folds_v2)
    write_t4_markdown(v1_agg, v2_agg, args.out_table)
    write_t4_csv(v1_agg, v2_agg, folds_v1, folds_v2, args.out_csv)
    write_fig4(v1_agg, v2_agg, args.out_fig, args.out_fig_json,
               args.out_fig_script)
    return 0


def _bacc(fold_record: dict) -> float:
    """Macro-averaged balanced accuracy from per-class confusion matrix.

    Recomputed from y_true/y_pred so the value is independent of how the
    record was produced (cache or live inference)."""
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(fold_record["y_true"], fold_record["y_pred"]))


if __name__ == "__main__":
    raise SystemExit(main())
