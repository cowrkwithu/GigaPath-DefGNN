#!/usr/bin/env python3
"""Extract per-tile attention from v1 GAT and v2 deformable best ckpts,
then compute attention-vs-CATCH-polygon-IoU + Pearson + AUC-PR for each WSI.

For each fold (0-4):
  * Load best v1 ckpt (canonical `fold_<i>.ckpt`) and v2 ckpt
    (best `epoch*-val_bacc*.ckpt`).
  * Iterate the val-split dataset directly (we need slide_id, which the
    DataLoader strips), forward each WSI through the model in eval mode,
    pull `gnn.readout.node_attention` (the global-attention pooling weights;
    shared between v1 and v2 since both wrap VetGigaGraph's readout).
  * Load the per-WSI CATCH tile labels (.npy at
    `/data/cia_outputs/annotations/tile_labels/<slide>.npy`).
  * Compute IoU / Pearson / AUC-PR via the v1 evaluator
    `src.evaluation.attention_iou.compute_attention_iou`.

Aggregate across all val WSIs (overall + per-class):
  * IoU, Pearson, AUC-PR with paired statistics (v2 - v1).
  * Wilcoxon signed-rank of Pearson against zero (the v1 protocol).

Writes:
  paper/tables/T6_attention_iou.md           — markdown table
  paper/figures/T6_attention_iou.csv          — raw per-WSI per-model rows
  paper/figures/fig5_attention_overlay/<slide_id>.png  — 4-panel WSI panels
                                                          (GT, v1 attn, v2 attn, v2 offsets)
  paper/figures/fig5_per_wsi_data.json        — figure source data
  paper/figures/scripts/fig5_attention_overlay.py  — zero-dep regenerator

Usage:
    python3 scripts/06_extract_attention_maps.py
    python3 scripts/06_extract_attention_maps.py --folds 0 2    # subset
    python3 scripts/06_extract_attention_maps.py --no-figures   # skip Fig 5
    python3 scripts/06_extract_attention_maps.py --max-wsis-per-class 2  # default
"""

from __future__ import annotations

import os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
# Phase B unification (2026-05-31): v1 src/ is now under this repo's src/;
# see scripts/04b_train_deformable.py for the same single-root pattern.
_sys.path.insert(0, str(_ROOT))
from src.utils.env import output_dir  # noqa: E402
_sys.path.insert(0, str(_ROOT / "src"))

import argparse                                                           # noqa: E402
import json                                                               # noqa: E402
import logging                                                            # noqa: E402
import re                                                                 # noqa: E402
import statistics                                                         # noqa: E402
from collections import defaultdict                                       # noqa: E402

import numpy as np                                                        # noqa: E402
import torch                                                              # noqa: E402
from omegaconf import OmegaConf                                           # noqa: E402

# v1
from src.utils.config import load_config                                  # noqa: E402
from src.utils.seed import set_global_seed                                # noqa: E402
from src.training.dataset import GraphSlideDataset                        # noqa: E402
from src.models import VetGigaGraph                                       # noqa: E402
from src.models.gigapath_slide import production_slide_backbone_loader   # noqa: E402
from src.evaluation.attention_iou import compute_attention_iou            # noqa: E402

# v2
from deformable_attention import build_deformable_model                   # noqa: E402

logger = logging.getLogger(__name__)

CLASSES = ["MEL", "MCT", "SCC", "PNST", "PLC", "TRB", "HIS"]
NUM_CLASSES = 7
LOCKED_SEEDS = [42, 123, 456, 789, 1024]

V1_DEFAULT_CONFIG = _ROOT / "configs" / "default.yaml"
V2_OVERRIDE_CONFIG = _ROOT / "configs" / "experiment_deformable.yaml"
V1_CKPT_ROOT = output_dir() / "checkpoints/exp2/dual_edge/vetgigagraph"
V2_CKPT_ROOT = _ROOT / "results" / "deformable"
TILE_LABELS_ROOT = output_dir() / "annotations/tile_labels"

_CKPT_PAT = re.compile(
    r"epoch(?P<epoch>\d+)-val_bacc(?P<bacc>\d+\.\d+)(?P<rev>-v\d+)?\.ckpt$"
)


# --------------------------------------------------------------------------- #
# Checkpoint discovery (same as 06_per_class_compare.py)
# --------------------------------------------------------------------------- #


def find_best_ckpt(directory: _Path) -> _Path | None:
    if not directory.is_dir():
        return None
    best: tuple[float, int, int, _Path] | None = None
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
    canonical = V1_CKPT_ROOT / f"fold_{fold}.ckpt"
    if canonical.is_file():
        return canonical
    sub = V1_CKPT_ROOT / f"fold_{fold}"
    if sub.is_dir():
        c = find_best_ckpt(sub)
        if c:
            return c
    return find_best_ckpt(V1_CKPT_ROOT)


def v2_fold_ckpt(fold: int) -> _Path | None:
    return find_best_ckpt(V2_CKPT_ROOT / f"fold_{fold}" / "checkpoints")


# --------------------------------------------------------------------------- #
# Config + model loading
# --------------------------------------------------------------------------- #


def load_v1_config():
    return load_config(V1_DEFAULT_CONFIG)


def load_v2_config():
    base = load_config(V1_DEFAULT_CONFIG)
    override = OmegaConf.load(V2_OVERRIDE_CONFIG)
    return OmegaConf.merge(base, override)


def load_state_dict_strip_prefix(model: torch.nn.Module, ckpt_path: _Path) -> None:
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
        logger.warning("[load] %s missing (first 3): %s …", ckpt_path.name, missing[:3])
    if unexpected:
        logger.warning("[load] %s unexpected (first 3): %s …", ckpt_path.name, unexpected[:3])


def build_v1(cfg) -> torch.nn.Module:
    fusion_strategy = str(cfg["model"]["fusion"]["strategy"])
    slide_loader = production_slide_backbone_loader if fusion_strategy != "gnn_only" else None
    return VetGigaGraph.from_config(cfg, slide_backbone_loader=slide_loader)


def build_v2(cfg) -> torch.nn.Module:
    fusion_strategy = str(cfg["model"]["fusion"]["strategy"])
    slide_loader = production_slide_backbone_loader if fusion_strategy != "gnn_only" else None
    return build_deformable_model(cfg, slide_backbone_loader=slide_loader)


# --------------------------------------------------------------------------- #
# Per-WSI attention + IoU
# --------------------------------------------------------------------------- #


def load_tile_labels(slide_id: str) -> np.ndarray | None:
    p = TILE_LABELS_ROOT / f"{slide_id}.npy"
    if not p.exists():
        return None
    return np.load(p)


def rule_ckpts(fold: int, rule: str) -> tuple[_Path, _Path]:
    """(GAT, GigaPath-DefGNN) checkpoints under the 18_test_eval.py rules."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_eval", _ROOT / "scripts" / "18_test_eval.py")
    te = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(te)
    out = []
    for name in ("gat", "defgnn"):
        kind, ckpt_dir, _ = te.MODELS[name]
        last = te.last_ckpt(kind, ckpt_dir, fold)
        out.append(last if rule == "last" else te.callback_state(last)[0])
    return out[0], out[1]


def extract_for_fold(model_kind: str, fold: int, ckpt_path: _Path,
                     splits_csv: _Path, graphs_root: _Path,
                     device: str = "cuda", split: str = "val") -> list[dict]:
    """Return a list of per-WSI dicts: slide_id, class, attention, iou metrics."""
    seed = LOCKED_SEEDS[fold]
    set_global_seed(seed)

    cfg = load_v1_config() if model_kind == "v1_gat" else load_v2_config()
    builder = build_v1 if model_kind == "v1_gat" else build_v2
    model = builder(cfg)
    load_state_dict_strip_prefix(model, ckpt_path)
    model.eval().to(device)

    ds = GraphSlideDataset(splits_csv, graphs_root, fold=fold, split=split,
                           attach_tile_labels=False)
    rows: list[dict] = []
    with torch.no_grad():
        for idx in range(len(ds)):
            slide_id = ds._slide_ids[idx]
            tumor_class = slide_id.split("_")[0]  # e.g. "HIS_01_1" → "HIS"
            data = ds[idx].to(device)
            out = model(data, data.x, data.pos)
            logits, attentions = (out if isinstance(out, tuple) else (out, {}))
            node_attn = attentions.get("gnn.readout.node_attention")
            if node_attn is None:
                logger.warning("[%s/%d/%s] no readout node_attention; skip",
                               model_kind, fold, slide_id)
                continue
            attn_np = node_attn.detach().float().cpu().numpy()
            labels = load_tile_labels(slide_id)
            if labels is None:
                logger.warning("[%s/%d/%s] missing tile_labels; skip",
                               model_kind, fold, slide_id)
                continue
            if labels.shape[0] != attn_np.shape[0]:
                logger.warning("[%s/%d/%s] tile count mismatch labels=%d attn=%d; skip",
                               model_kind, fold, slide_id,
                               labels.shape[0], attn_np.shape[0])
                continue
            metrics = compute_attention_iou(attn_np, labels)
            # Pull v2-specific offsets too, for Figure 5 (the unique v2 signal)
            offsets_np = None
            for k, v in attentions.items():
                if k.endswith("deformable.offsets") and v is not None:
                    offsets_np = v.detach().float().cpu().numpy()
                    break
            # Coords for Figure 5 layout
            coords_np = data.pos.detach().float().cpu().numpy()
            row = {
                "model": model_kind,
                "fold": fold,
                "seed": seed,
                "slide_id": slide_id,
                "tumor_class": tumor_class,
                "attention": attn_np.tolist(),
                "coords": coords_np.tolist(),
                "tile_labels": labels.tolist(),
                **metrics,
            }
            if offsets_np is not None:
                row["offsets"] = offsets_np.tolist()
            rows.append(row)
    del model
    torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def aggregate(rows: list[dict]) -> dict:
    """Group by model_kind, optionally subset to n_tumor_gt > 0 WSIs."""
    valid = [r for r in rows if r.get("iou") is not None]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_model[r["model"]].append(r)
    out: dict[str, dict] = {}
    for model_kind, mrows in by_model.items():
        ious = [r["iou"] for r in mrows]
        pears = [r["pearson"] for r in mrows if r.get("pearson") is not None]
        aucs = [r["auc_pr"] for r in mrows if r.get("auc_pr") is not None]
        out[model_kind] = {
            "n_wsis": len(mrows),
            "iou_mean": float(np.mean(ious)),
            "iou_std":  float(np.std(ious, ddof=0)),
            "iou_median": float(np.median(ious)),
            "pearson_n": len(pears),
            "pearson_median": float(np.median(pears)) if pears else float("nan"),
            "pearson_mean": float(np.mean(pears)) if pears else float("nan"),
            "auc_pr_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "auc_pr_median": float(np.median(aucs)) if aucs else float("nan"),
        }
        # Per-class stats
        by_class: dict[str, list[dict]] = defaultdict(list)
        for r in mrows:
            by_class[r["tumor_class"]].append(r)
        out[model_kind]["per_class"] = {
            cls: {
                "n": len(crows),
                "iou_mean": float(np.mean([r["iou"] for r in crows])),
                "iou_std":  float(np.std([r["iou"] for r in crows], ddof=0)),
                "pearson_median": float(np.median(
                    [r["pearson"] for r in crows if r.get("pearson") is not None]
                )) if any(r.get("pearson") is not None for r in crows) else float("nan"),
            }
            for cls, crows in sorted(by_class.items())
        }
    return out


def paired_tests(rows_v1: list[dict], rows_v2: list[dict]) -> dict:
    """Wilcoxon (vs zero) on each model's Pearson, and paired Wilcoxon between
    v1 and v2 IoU on the WSI subset that both models have valid IoU for."""
    from scipy import stats
    # Sort by slide_id so v1[i] and v2[i] are the same WSI
    v1_by = {r["slide_id"]: r for r in rows_v1 if r.get("iou") is not None}
    v2_by = {r["slide_id"]: r for r in rows_v2 if r.get("iou") is not None}
    common = sorted(set(v1_by) & set(v2_by))
    v1_iou = np.array([v1_by[s]["iou"] for s in common])
    v2_iou = np.array([v2_by[s]["iou"] for s in common])
    v1_pe  = np.array([v1_by[s]["pearson"] for s in common if v1_by[s].get("pearson") is not None])
    v2_pe  = np.array([v2_by[s]["pearson"] for s in common if v2_by[s].get("pearson") is not None])
    out = {"n_common": len(common)}
    if len(common) >= 2:
        diff = v2_iou - v1_iou
        if (diff != 0).any():
            ws = stats.wilcoxon(v1_iou, v2_iou)
            out["paired_iou_wilcoxon"] = {"stat": float(ws.statistic),
                                          "p": float(ws.pvalue),
                                          "mean_delta": float(diff.mean())}
        else:
            out["paired_iou_wilcoxon"] = {"note": "all WSIs tied — no test"}
    if len(v1_pe) >= 2:
        ws1 = stats.wilcoxon(v1_pe)
        out["v1_pearson_vs_zero"] = {"stat": float(ws1.statistic),
                                     "p": float(ws1.pvalue),
                                     "median": float(np.median(v1_pe))}
    if len(v2_pe) >= 2:
        ws2 = stats.wilcoxon(v2_pe)
        out["v2_pearson_vs_zero"] = {"stat": float(ws2.statistic),
                                     "p": float(ws2.pvalue),
                                     "median": float(np.median(v2_pe))}
    return out


# --------------------------------------------------------------------------- #
# Table T6
# --------------------------------------------------------------------------- #


def write_t6_markdown(agg: dict, paired: dict, out: _Path) -> None:
    v1 = agg.get("v1_gat", {})
    v2 = agg.get("v2_deformable", {})
    lines = [
        "# Table T6 — Attention–Annotation Alignment (n WSIs with > 0 GT tumor tiles)",
        "",
        "Per-WSI metrics computed via the v1 evaluator `src.evaluation.attention_iou.compute_attention_iou`:",
        "  * **IoU**: between top-K binarized attention map and CATCH polygon tumor mask, K = #GT tumor tiles.",
        "  * **Pearson**: continuous attention vs binary tumor mask.",
        "  * **AUC-PR**: attention as a tumor-vs-non-tumor classifier.",
        "",
        f"Subset (n_tumor_gt > 0): v1 n = {v1.get('n_wsis', '—')}, v2 n = {v2.get('n_wsis', '—')}, paired n = {paired.get('n_common', '—')}.",
        "",
        "| Metric | v1 GAT | v2 Deformable | Δ |",
        "|---|---|---|---|",
    ]
    def fmt(d, key, fmt_str="{:.3f}"):
        v = d.get(key)
        if v is None or (isinstance(v, float) and (v != v)):
            return "—"
        return fmt_str.format(v)
    iou_d = (v2.get("iou_mean", 0) - v1.get("iou_mean", 0)) if v1 and v2 else float("nan")
    pear_d = (v2.get("pearson_median", 0) - v1.get("pearson_median", 0)) if v1 and v2 else float("nan")
    auc_d = (v2.get("auc_pr_mean", 0) - v1.get("auc_pr_mean", 0)) if v1 and v2 else float("nan")
    lines += [
        f"| IoU (mean ± std)          | {fmt(v1, 'iou_mean')} ± {fmt(v1, 'iou_std')} | {fmt(v2, 'iou_mean')} ± {fmt(v2, 'iou_std')} | {iou_d:+.3f} |",
        f"| Pearson (median)           | {fmt(v1, 'pearson_median')} | {fmt(v2, 'pearson_median')} | {pear_d:+.3f} |",
        f"| AUC-PR (mean)              | {fmt(v1, 'auc_pr_mean')} | {fmt(v2, 'auc_pr_mean')} | {auc_d:+.3f} |",
    ]
    # Statistical block
    p1 = paired.get("v1_pearson_vs_zero", {})
    p2 = paired.get("v2_pearson_vs_zero", {})
    pi = paired.get("paired_iou_wilcoxon", {})
    lines += [
        "",
        "**Significance**:",
        f"  * Wilcoxon (v1 Pearson vs 0): p = {fmt(p1, 'p', '{:.4f}')} (median {fmt(p1, 'median')})",
        f"  * Wilcoxon (v2 Pearson vs 0): p = {fmt(p2, 'p', '{:.4f}')} (median {fmt(p2, 'median')})",
        f"  * Paired Wilcoxon (v2 IoU − v1 IoU on common WSIs): p = {fmt(pi, 'p', '{:.4f}')}, mean Δ IoU = {fmt(pi, 'mean_delta', '{:+.4f}')}",
        "",
        "## Per-class IoU",
        "",
        "| Class | n | v1 IoU (mean ± std) | v2 IoU (mean ± std) | Δ |",
        "|---|---|---|---|---|",
    ]
    v1_pc = v1.get("per_class", {})
    v2_pc = v2.get("per_class", {})
    classes = sorted(set(v1_pc) | set(v2_pc))
    for cls in classes:
        v1c = v1_pc.get(cls, {})
        v2c = v2_pc.get(cls, {})
        delta = (v2c.get("iou_mean", 0) - v1c.get("iou_mean", 0)) if v1c and v2c else float("nan")
        n = v1c.get("n") or v2c.get("n") or 0
        lines.append(
            f"| {cls} | {n} | "
            f"{fmt(v1c, 'iou_mean')} ± {fmt(v1c, 'iou_std')} | "
            f"{fmt(v2c, 'iou_mean')} ± {fmt(v2c, 'iou_std')} | "
            f"{delta:+.3f} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    logger.info("[t6] wrote %s", out)


def write_t6_csv(rows: list[dict], out: _Path) -> None:
    import csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "fold", "slide_id", "tumor_class",
                    "n_tiles", "n_tumor_gt", "n_attn_top",
                    "iou", "pearson", "auc_pr"])
        for r in rows:
            w.writerow([r["model"], r["fold"], r["slide_id"], r["tumor_class"],
                        r["n_tiles"], r["n_tumor_gt"], r["n_attn_top"],
                        r["iou"], r["pearson"], r["auc_pr"]])
    logger.info("[t6] wrote %s", out)


# --------------------------------------------------------------------------- #
# Figure 5 — qualitative attention overlay
# --------------------------------------------------------------------------- #


def pick_demo_wsis(rows: list[dict], max_per_class: int = 2) -> list[str]:
    """Pick representative WSIs: for each tumor class, the ones where v2
    improves IoU the most (or matches v1 best if no improvement)."""
    v1_by = {r["slide_id"]: r for r in rows if r["model"] == "v1_gat" and r.get("iou") is not None}
    v2_by = {r["slide_id"]: r for r in rows if r["model"] == "v2_deformable" and r.get("iou") is not None}
    common = sorted(set(v1_by) & set(v2_by))
    by_class: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for s in common:
        d = v2_by[s]["iou"] - v1_by[s]["iou"]
        by_class[v2_by[s]["tumor_class"]].append((d, s))
    out: list[str] = []
    for cls in CLASSES:
        ranked = sorted(by_class.get(cls, []), reverse=True)  # highest delta first
        out.extend(s for _, s in ranked[:max_per_class])
    return out


def pick_random_wsis(rows: list[dict], max_per_class: int = 2,
                     min_tumor_tiles: int = 20, seed: int = 0) -> list[str]:
    """Pick WSIs per class uniformly at random (fixed seed) among slides that
    both models scored and that have >= ``min_tumor_tiles`` annotated tumor
    tiles. Unlike :func:`pick_demo_wsis`, the choice does not depend on either
    model's IoU, so the panels are not selected in favor of one model."""
    v1 = {r["slide_id"] for r in rows if r["model"] == "v1_gat" and r.get("iou") is not None}
    v2 = {r["slide_id"]: r for r in rows if r["model"] == "v2_deformable" and r.get("iou") is not None}
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for cls in CLASSES:
        pool = sorted(sid for sid, r in v2.items() if sid in v1 and r["tumor_class"] == cls
                      and int(r.get("n_tumor_gt") or 0) >= min_tumor_tiles)
        k = min(max_per_class, len(pool))
        out.extend(sorted(rng.choice(pool, size=k, replace=False).tolist()) if k else [])
    return out


def render_fig5_composite(slide_ids: list[str], rows: list[dict], out_png: _Path) -> None:
    """One figure: a row per class, each row = 2 WSIs x (GT, GAT, DefGNN)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.io_utils import CATCH_TUMOR_RANGE
    lo, hi = CATCH_TUMOR_RANGE
    by_model = {(r["model"], r["slide_id"]): r for r in rows}
    per_class: dict[str, list[str]] = defaultdict(list)
    for sid in slide_ids:
        per_class[by_model[("v2_deformable", sid)]["tumor_class"]].append(sid)
    classes = [c for c in CLASSES if per_class.get(c)]
    ncol = 3 * max(len(v) for v in per_class.values())
    fig, axes = plt.subplots(len(classes), ncol, figsize=(2.3 * ncol, 2.4 * len(classes)))
    axes = np.atleast_2d(axes)
    for i, cls in enumerate(classes):
        for j in range(ncol // 3):
            trio = axes[i, 3 * j: 3 * j + 3]
            if j >= len(per_class[cls]):
                for ax in trio:
                    ax.axis("off")
                continue
            sid = per_class[cls][j]
            v1, v2 = by_model[("v1_gat", sid)], by_model[("v2_deformable", sid)]
            xy = np.array(v2["coords"])
            labels = np.array(v2["tile_labels"])
            gt = ((labels >= lo) & (labels <= hi)).astype(float)
            panels = [(gt, "Reds", f"{sid}\nannotated tumor"),
                      (np.array(v1["attention"]), "viridis", f"GAT\nIoU {v1['iou']:.2f}, r {v1['pearson']:.2f}".replace("-", "−")),
                      (np.array(v2["attention"]), "viridis", f"GigaPath-DefGNN\nIoU {v2['iou']:.2f}, r {v2['pearson']:.2f}".replace("-", "−"))]
            size = 2.0 if len(xy) > 5000 else 5.0
            for ax, (vals, cmap, title) in zip(trio, panels):
                rank = vals.argsort().argsort() / max(len(vals) - 1, 1) if cmap == "viridis" else vals
                ax.scatter(xy[:, 0], -xy[:, 1], c=rank, cmap=cmap, s=size, marker="s", linewidth=0)
                ax.set_title(title, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        axes[i, 0].set_ylabel(cls, fontsize=10, rotation=0, labelpad=18)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_fig5_panel(slide_id: str, rows: list[dict], out_png: _Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[fig5] matplotlib missing; skip render for %s", slide_id)
        return
    v1 = next((r for r in rows if r["model"] == "v1_gat" and r["slide_id"] == slide_id), None)
    v2 = next((r for r in rows if r["model"] == "v2_deformable" and r["slide_id"] == slide_id), None)
    if v1 is None or v2 is None:
        return
    coords = np.array(v2["coords"])
    labels = np.array(v2["tile_labels"])
    from src.utils.io_utils import CATCH_TUMOR_RANGE
    lo, hi = CATCH_TUMOR_RANGE
    gt = ((labels >= lo) & (labels <= hi)).astype(int)
    a1 = np.array(v1["attention"]); a2 = np.array(v2["attention"])
    # Normalize attentions for display
    def norm(a):
        if a.max() == a.min(): return a * 0
        return (a - a.min()) / (a.max() - a.min())
    a1n, a2n = norm(a1), norm(a2)

    has_offsets = "offsets" in v2 and v2["offsets"]
    n_panels = 4 if has_offsets else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.2))
    titles = [f"{slide_id} — GT tumor", "v1 GAT attention", "v2 Deformable attention"]
    if has_offsets:
        titles.append("v2 deformable offsets")
    values = [gt, a1n, a2n]

    # Invert y so the WSI orientation looks natural
    s = 8.0 if len(coords) > 5000 else 16.0
    for ax, title, vals in zip(axes[:3], titles[:3], values):
        sc = ax.scatter(coords[:, 0], -coords[:, 1], c=vals,
                        cmap=("Reds" if "GT" in title else "viridis"),
                        s=s, marker="s", alpha=0.85, linewidth=0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, fraction=0.045)

    if has_offsets:
        ax = axes[3]
        ax.scatter(coords[:, 0], -coords[:, 1], c=a2n, cmap="viridis",
                   s=s, marker="s", alpha=0.5, linewidth=0)
        offs = np.array(v2["offsets"])  # [N, K, 2] in normalized coords
        # Convert normalized offsets back to slide-pixel space for display
        cmin = coords.min(axis=0); cmax = coords.max(axis=0)
        span = (cmax - cmin).clip(min=1)
        # Mean offset across K samples per node
        mean_off = offs.mean(axis=1) * span
        # Subsample tiles for readability
        step = max(1, len(coords) // 800)
        idx = np.arange(0, len(coords), step)
        ax.quiver(coords[idx, 0], -coords[idx, 1],
                  mean_off[idx, 0], -mean_off[idx, 1],
                  angles="xy", scale_units="xy", scale=1,
                  width=0.0025, alpha=0.6, color="black")
        ax.set_title(titles[3], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    fig.suptitle(f"{slide_id} ({v2['tumor_class']}) — "
                 f"v1 IoU = {v1['iou']:.3f}, v2 IoU = {v2['iou']:.3f}",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


_FIG5_README = """# Figure 5 — Attention–annotation overlay panels

Per-WSI 3- or 4-panel images saved as `<slide_id>.png`. Generated from
`fig5_per_wsi_data.json` (one entry per selected WSI). To regenerate the
PNGs on a machine without GPU/CATCH data access, run
`paper/figures/scripts/fig5_attention_overlay.py`.

Panels (left → right):
  1. CATCH polygon tumor mask (binary, red).
  2. v1 GAT readout attention (normalized 0-1, viridis).
  3. v2 deformable readout attention (normalized 0-1, viridis).
  4. (v2-only) Mean deformable offsets per tile (quiver overlay on v2 attn).

Slide subtitle reports per-WSI IoU for both models.
"""

_FIG5_REGEN_SCRIPT = '''#!/usr/bin/env python3
"""Regenerate Figure 5 attention-overlay panels from JSON.
No GPU/data access required; only matplotlib + numpy.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent.parent
DATA = HERE / "fig5_per_wsi_data.json"
OUTDIR = HERE / "fig5_attention_overlay"
OUTDIR.mkdir(parents=True, exist_ok=True)

CATCH_TUMOR_LO, CATCH_TUMOR_HI = 7, 13

def norm(a):
    a = np.asarray(a, dtype=float)
    if a.max() == a.min():
        return a * 0
    return (a - a.min()) / (a.max() - a.min())

data = json.loads(DATA.read_text())
for entry in data:
    sid = entry["slide_id"]
    coords = np.array(entry["coords"])
    labels = np.array(entry["tile_labels"])
    gt = ((labels >= CATCH_TUMOR_LO) & (labels <= CATCH_TUMOR_HI)).astype(int)
    a1 = norm(entry["v1_attention"])
    a2 = norm(entry["v2_attention"])
    has_offsets = "v2_offsets" in entry
    n_panels = 4 if has_offsets else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.2))
    s = 8.0 if len(coords) > 5000 else 16.0
    panels = [("GT tumor", gt, "Reds"),
              ("v1 GAT attention", a1, "viridis"),
              ("v2 Deformable attention", a2, "viridis")]
    for ax, (title, vals, cmap) in zip(axes[:3], panels):
        sc = ax.scatter(coords[:, 0], -coords[:, 1], c=vals, cmap=cmap,
                        s=s, marker="s", alpha=0.85, linewidth=0)
        ax.set_title(title, fontsize=10); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.045)
    if has_offsets:
        ax = axes[3]
        ax.scatter(coords[:, 0], -coords[:, 1], c=a2, cmap="viridis",
                   s=s, marker="s", alpha=0.5, linewidth=0)
        offs = np.array(entry["v2_offsets"])
        cmin = coords.min(axis=0); cmax = coords.max(axis=0)
        span = (cmax - cmin).clip(min=1)
        mean_off = offs.mean(axis=1) * span
        step = max(1, len(coords) // 800)
        idx = np.arange(0, len(coords), step)
        ax.quiver(coords[idx, 0], -coords[idx, 1],
                  mean_off[idx, 0], -mean_off[idx, 1],
                  angles="xy", scale_units="xy", scale=1,
                  width=0.0025, alpha=0.6, color="black")
        ax.set_title("v2 deformable offsets", fontsize=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{sid} ({entry['tumor_class']}) — "
                 f"v1 IoU = {entry['v1_iou']:.3f}, v2 IoU = {entry['v2_iou']:.3f}",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{sid}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUTDIR / f"{sid}.png")
'''


def write_fig5(rows: list[dict], demo_slide_ids: list[str], out_dir: _Path,
               out_json: _Path, out_script: _Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON: compact per-WSI bundles for the regen script
    bundles = []
    for sid in demo_slide_ids:
        v1 = next((r for r in rows if r["model"] == "v1_gat" and r["slide_id"] == sid), None)
        v2 = next((r for r in rows if r["model"] == "v2_deformable" and r["slide_id"] == sid), None)
        if v1 is None or v2 is None:
            continue
        bundle = {
            "slide_id": sid,
            "tumor_class": v2["tumor_class"],
            "v1_iou": v1["iou"], "v2_iou": v2["iou"],
            "coords": v2["coords"],
            "tile_labels": v2["tile_labels"],
            "v1_attention": v1["attention"],
            "v2_attention": v2["attention"],
        }
        if "offsets" in v2:
            bundle["v2_offsets"] = v2["offsets"]
        bundles.append(bundle)
        render_fig5_panel(sid, rows, out_dir / f"{sid}.png")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(bundles))
    out_script.parent.mkdir(parents=True, exist_ok=True)
    out_script.write_text(_FIG5_REGEN_SCRIPT)
    (out_dir / "README.md").write_text(_FIG5_README)
    logger.info("[fig5] wrote %d WSI panels + %s + %s",
                len(bundles), out_json, out_script)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   choices=[0, 1, 2, 3, 4])
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", choices=("val", "test"), default="val",
                   help="Per-fold split to score (test pools to all 350 WSIs).")
    p.add_argument("--rule", choices=("legacy", "best", "last"), default="legacy",
                   help="Checkpoint rule. legacy = published analysis (GAT final "
                        "weights, DefGNN highest-val latest-epoch file); best/last "
                        "= the rules of scripts/18_test_eval.py.")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip Figure 5 rendering (table-only run).")
    p.add_argument("--max-wsis-per-class", type=int, default=2)
    p.add_argument("--demo-selection", choices=("max_delta", "random"), default="max_delta",
                   help="max_delta = published Fig. 5 (largest DefGNN IoU gain; favors the "
                        "proposed model); random = fixed-seed draw per class.")
    p.add_argument("--min-tumor-tiles", type=int, default=20)
    p.add_argument("--demo-seed", type=int, default=0)
    p.add_argument("--out-composite", type=_Path, default=None,
                   help="Also write a single multi-row composite figure here.")
    p.add_argument("--out-table", type=_Path,
                   default=_ROOT / "paper" / "tables" / "T6_attention_iou.md")
    p.add_argument("--out-csv", type=_Path,
                   default=_ROOT / "paper" / "figures" / "T6_attention_iou.csv")
    p.add_argument("--out-fig-dir", type=_Path,
                   default=_ROOT / "paper" / "figures" / "fig5_attention_overlay")
    p.add_argument("--out-fig-json", type=_Path,
                   default=_ROOT / "paper" / "figures" / "fig5_per_wsi_data.json")
    p.add_argument("--out-fig-script", type=_Path,
                   default=_ROOT / "paper" / "figures" / "scripts"
                            / "fig5_attention_overlay.py")
    p.add_argument("--cache", type=_Path,
                   default=_ROOT / "results" / "attention_extraction.json")
    p.add_argument("--no-cache", action="store_true")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()

    cached_rows: list[dict] | None = None
    if args.cache.exists() and not args.no_cache:
        try:
            cached_rows = json.loads(args.cache.read_text())
            logger.info("[cache] loaded %d rows from %s",
                        len(cached_rows), args.cache)
        except Exception as e:
            logger.warning("[cache] failed to load %s: %s", args.cache, e)

    rows: list[dict] = []
    cfg_v1 = load_v1_config()
    splits_csv = _Path(cfg_v1["paths"]["splits"]) / "cv5fold.csv"
    graphs_root = _Path(cfg_v1["paths"]["graphs"]) / str(cfg_v1["graph"]["type"])

    for fold in args.folds:
        if args.rule == "legacy":
            v1_ckpt, v2_ckpt = v1_fold_ckpt(fold), v2_fold_ckpt(fold)
        else:
            v1_ckpt, v2_ckpt = rule_ckpts(fold, args.rule)
        for kind, ckpt in (("v1_gat", v1_ckpt), ("v2_deformable", v2_ckpt)):
            if cached_rows is not None and any(
                r["model"] == kind and r["fold"] == fold for r in cached_rows
            ):
                fold_rows = [r for r in cached_rows
                             if r["model"] == kind and r["fold"] == fold]
                rows.extend(fold_rows)
                logger.info("[cache] %s fold %d → %d WSIs", kind, fold, len(fold_rows))
                continue
            if ckpt is None:
                logger.warning("[%s] fold %d: no ckpt found", kind, fold)
                continue
            logger.info("[%s] fold %d: %s", kind, fold, ckpt.name)
            fold_rows = extract_for_fold(kind, fold, ckpt, splits_csv,
                                         graphs_root, device=args.device,
                                         split=args.split)
            valid = [r for r in fold_rows if r.get("iou") is not None]
            logger.info("[%s] fold %d done → %d WSIs, %d with IoU; "
                        "mean IoU %.3f, median Pearson %.3f",
                        kind, fold, len(fold_rows), len(valid),
                        float(np.mean([r["iou"] for r in valid])) if valid else float("nan"),
                        float(np.median([r["pearson"] for r in valid
                                         if r.get("pearson") is not None]))
                            if any(r.get("pearson") is not None for r in valid)
                            else float("nan"))
            rows.extend(fold_rows)

    # Persist cache (large; contains attention arrays for Figure 5)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(rows))
    logger.info("[cache] wrote %s (%d rows)", args.cache, len(rows))

    rows_v1 = [r for r in rows if r["model"] == "v1_gat"]
    rows_v2 = [r for r in rows if r["model"] == "v2_deformable"]
    agg = aggregate(rows)
    paired = paired_tests(rows_v1, rows_v2)
    write_t6_markdown(agg, paired, args.out_table)
    write_t6_csv(rows, args.out_csv)

    if not args.no_figures:
        if args.demo_selection == "random":
            demos = pick_random_wsis(rows, max_per_class=args.max_wsis_per_class,
                                     min_tumor_tiles=args.min_tumor_tiles, seed=args.demo_seed)
        else:
            demos = pick_demo_wsis(rows, max_per_class=args.max_wsis_per_class)
        logger.info("[fig5] selected %d demo WSIs: %s", len(demos), demos)
        write_fig5(rows, demos, args.out_fig_dir, args.out_fig_json,
                   args.out_fig_script)
        if args.out_composite is not None:
            render_fig5_composite(demos, rows, args.out_composite)
            logger.info("[fig5] wrote composite %s", args.out_composite)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
