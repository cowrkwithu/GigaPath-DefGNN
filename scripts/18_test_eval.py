#!/usr/bin/env python3
"""Held-out test-set evaluation for GigaPath-DefGNN and all nine baselines.

Answers Reviewer 2 point 2 (1st revision, applsci-4523976): the per-fold test
splits in ``cv5fold.csv`` partition the full cohort (66+75+68+73+68 = 350), so
pooling the five test folds yields one out-of-fold prediction per WSI.

Checkpoint-selection rules, each resolved from the ModelCheckpoint state
Lightning stored in the checkpoints:

* ``best``  — ``best_model_path``: first epoch reaching the maximum
  val_balanced_accuracy. Available for every model (primary rule).
* ``top3``  — ``last_model_path``: the most recent of the top-3 checkpoints.
  Lightning 2.x refreshes ``last.ckpt`` only when a top-k checkpoint is
  saved, so this is the latest epoch that entered the top 3, not the final
  epoch. Available for every model.
* ``final`` — weights after early stopping (``fold_<i>.ckpt``, saved by
  ``04_train.py`` after ``fit``). Not available for GigaPath-DefGNN:
  ``04b_train_deformable.py`` never saved its final weights.

The published validation numbers mixed rules (baselines: final-epoch val
metric; GigaPath-DefGNN: best checkpoint).

Before scoring test, each (model, fold, rule) re-scores its validation split
and records it next to the value stored at training time, as a checkpoint
identity check.

Run:  python3 scripts/18_test_eval.py [--models gat defgnn] [--folds 0 1]
Out:  results/test_eval/predictions.json  (per-WSI y_true / y_pred / y_prob)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
from src.utils.env import defgnn_run_dir, output_dir, portable_path  # noqa: E402
_sys.path.insert(0, str(_ROOT / "src"))

import argparse                                             # noqa: E402
import copy                                                 # noqa: E402
import gc                                                   # noqa: E402
import json                                                 # noqa: E402
import logging                                              # noqa: E402
import time                                                 # noqa: E402

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from omegaconf import OmegaConf                             # noqa: E402
from sklearn.metrics import balanced_accuracy_score         # noqa: E402

from src.utils.config import load_config                    # noqa: E402
from src.utils.seed import set_global_seed                  # noqa: E402
from src.training import GraphSlideDataModule              # noqa: E402
from src.training.dataset import baseline_forward_fn, vetgigagraph_forward_fn  # noqa: E402
from src.models import VetGigaGraph                         # noqa: E402
from src.models.baselines import BASELINE_REGISTRY          # noqa: E402
from deformable_attention import build_deformable_model     # noqa: E402

logger = logging.getLogger("test_eval")

LOCKED_SEEDS = [42, 123, 456, 789, 1024]
CKPT = output_dir() / "checkpoints"
DEFGNN_CKPT = defgnn_run_dir()
GRAPHS = output_dir() / "graphs/dual_edge"
SPLITS = output_dir() / "splits/cv5fold.csv"
DEFAULT_CFG = _ROOT / "configs" / "default.yaml"
# knn_chunk_size 16,384 instead of 1,024: same neighbours (fp32 max diff 9.5e-7),
# ~12x faster inference. Evaluations run before this change used the 1,024 config.
DEFORM_CFG = _ROOT / "configs" / "rev1" / "experiment_deformable_fastknn.yaml"

# name -> (kind, checkpoint dir, gnn backbone for v1 graph models)
MODELS = {
    "gcn":       ("gnn", CKPT / "exp3/gcn/vetgigagraph", "gcn"),
    "gin":       ("gnn", CKPT / "exp3/gin/vetgigagraph", "gin"),
    "graphsage": ("gnn", CKPT / "exp3/graphsage/vetgigagraph", "graphsage"),
    "gat":       ("gnn", CKPT / "exp2/dual_edge/vetgigagraph", "gat"),
    "abmil":     ("mil", CKPT / "exp1/abmil", None),
    "dsmil":     ("mil", CKPT / "exp1/dsmil", None),
    "transmil":  ("mil", CKPT / "exp1/transmil", None),
    "clam_sb":   ("mil", CKPT / "exp1/clam_sb", None),
    "clam_mb":   ("mil", CKPT / "exp1/clam_mb", None),
    "defgnn":    ("deform", DEFGNN_CKPT, None),
    # Added in the 1st revision (scripts/rev1_queue.sh); per-fold run dirs.
    "acmil":        ("mil_rev1", CKPT / "rev1/acmil", None),
    "wikg":         ("mil_rev1", CKPT / "rev1/wikg", None),
    "clam_sb_inst": ("mil_rev1", CKPT / "rev1/clam_sb_inst", None),
    "clam_mb_inst": ("mil_rev1", CKPT / "rev1/clam_mb_inst", None),
    # GNNs retrained in the 1st revision; third field = full config file.
    "gcn_fix":       ("gnn_rev1", CKPT / "rev1/gcn_fix", _ROOT / "configs/rev1/gcn_fix.yaml"),
    "gin_fix":       ("gnn_rev1", CKPT / "rev1/gin_fix", _ROOT / "configs/rev1/gin_fix.yaml"),
    "gin_fp32":      ("gnn_rev1", CKPT / "rev1/gin_fp32", _ROOT / "configs/rev1/gin_fp32.yaml"),
    "gin_layernorm": ("gnn_rev1", CKPT / "rev1/gin_layernorm", _ROOT / "configs/rev1/gin_layernorm.yaml"),
    "gin_mean":      ("gnn_rev1", CKPT / "rev1/gin_mean", _ROOT / "configs/rev1/gin_mean.yaml"),
    # Graph-construction ablation (R1-4) and 20x re-run (R2-3b).
    "gat_spatial":    ("gnn", CKPT / "exp2/spatial_knn/vetgigagraph", "gat"),
    "gat_feature":    ("gnn", CKPT / "exp2/feature_sim/vetgigagraph", "gat"),
    "defgnn_spatial": ("deform_rev1", CKPT / "rev1/defgnn_spatial", None),
    "defgnn_feature": ("deform_rev1", CKPT / "rev1/defgnn_feature", None),
    "gat_featknn":    ("gnn_rev1", CKPT / "rev1/gat_featknn", _ROOT / "configs/default.yaml"),
    "defgnn_featknn": ("deform_rev1", CKPT / "rev1/defgnn_featknn", None),
    "defgnn_dual":    ("deform_rev1", CKPT / "rev1/defgnn_dual", None),
    "gat_20x":        ("gnn_rev1", CKPT / "rev1/gat_20x", _ROOT / "configs/default.yaml"),
    "defgnn_20x":     ("deform_rev1", CKPT / "rev1/defgnn_20x", None),
}

#: Graph root per model (default: the 40x dual-edge graphs).
GRAPHS_OF = {
    "gat_spatial": output_dir() / "graphs/spatial_knn",
    "gat_feature": output_dir() / "graphs/feature_sim",
    "defgnn_spatial": output_dir() / "graphs/spatial_knn",
    "defgnn_feature": output_dir() / "graphs/feature_sim",
    "gat_featknn": output_dir() / "graphs/feature_knn",
    "defgnn_featknn": output_dir() / "graphs/feature_knn",
    "gat_20x": output_dir() / "graphs_20x/dual_edge",
    "defgnn_20x": output_dir() / "graphs_20x/dual_edge",
}


def last_ckpt(kind: str, ckpt_dir: _Path, fold: int) -> _Path:
    """The run's latest saved checkpoint (carries the ModelCheckpoint state)."""
    if kind in ("deform", "deform_rev1"):
        return ckpt_dir / f"fold_{fold}" / "checkpoints" / "last.ckpt"
    if kind in ("mil_rev1", "gnn_rev1"):
        return ckpt_dir / f"fold_{fold}" / f"fold_{fold}.ckpt"
    return ckpt_dir / f"fold_{fold}.ckpt"


def top3_latest(ckpt_path: _Path) -> _Path:
    """``last_model_path`` from the ModelCheckpoint state of ``ckpt_path``."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key, state in blob.get("callbacks", {}).items():
        if "ModelCheckpoint" in key and state.get("last_model_path"):
            local = ckpt_path.parent / _Path(state["last_model_path"]).name
            if local.is_file():
                return local
    logger.warning("%s: no last_model_path recorded, using the file itself", ckpt_path)
    return ckpt_path


def callback_state(ckpt_path: _Path) -> tuple[_Path, float, int]:
    """Return (best_model_path, best_model_score, epoch) stored in a ckpt."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key, state in blob.get("callbacks", {}).items():
        if "ModelCheckpoint" in key:
            if state.get("best_model_score") is None:
                # No finite val score was ever logged (GIN diverged), so there
                # is no best checkpoint; fall back to the final weights.
                logger.warning("%s: no best checkpoint recorded, using last", ckpt_path)
                return ckpt_path, float("nan"), int(blob["epoch"])
            best = _Path(state["best_model_path"])
            # Resolve against the ckpt's own directory: run dirs were moved
            # after training, so the absolute path may be stale.
            local = ckpt_path.parent / best.name
            return (local if local.is_file() else best,
                    float(state["best_model_score"]), int(blob["epoch"]))
    raise KeyError(f"no ModelCheckpoint state in {ckpt_path}")


def build(name: str):
    kind, _, backbone = MODELS[name]
    if kind in ("mil", "mil_rev1"):
        return BASELINE_REGISTRY[name](embed_dim=1536, hidden_dim=256, num_classes=7), baseline_forward_fn
    if kind == "gnn_rev1":
        cfg = load_config(backbone)  # full merged config saved in configs/rev1/
        return VetGigaGraph.from_config(cfg, slide_backbone_loader=None), vetgigagraph_forward_fn
    if kind == "gnn":
        cfg = load_config(DEFAULT_CFG)
        cfg = OmegaConf.merge(cfg, {"model": {"gnn": {"backbone": backbone}}})
        return VetGigaGraph.from_config(cfg, slide_backbone_loader=None), vetgigagraph_forward_fn
    cfg = OmegaConf.merge(load_config(DEFAULT_CFG), OmegaConf.load(DEFORM_CFG))
    return build_deformable_model(cfg, slide_backbone_loader=None), vetgigagraph_forward_fn


def load_weights(model: torch.nn.Module, ckpt_path: _Path) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in raw.items()}
    model.load_state_dict(sd, strict=True)


@torch.no_grad()
def infer(model, forward_fn, loader, device="cuda") -> dict:
    model.eval().to(device)
    ids, ys, preds, probs = [], [], [], []
    for cached in loader:
        # Data.to() moves tensors in place; copy first so cached CPU graphs
        # are not left resident on the GPU (that exhausted memory on 09-28).
        batch = copy.copy(cached).to(device)
        args, target = forward_fn(batch)
        # Training and its val metrics ran under 16-mixed autocast (GPU only;
        # the --device cpu fallback runs in fp32).
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = model(*args)
        logits = (out[0] if isinstance(out, tuple) else out).float()
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        prob = torch.softmax(logits, dim=-1)[0]
        ids.append(str(batch.slide_id))
        ys.append(int(target.reshape(-1)[0]))
        preds.append(int(prob.argmax()))
        probs.append([round(float(p), 6) for p in prob.cpu()])
    return {"slide_id": ids, "y_true": ys, "y_pred": preds, "y_prob": probs,
            "bacc": float(balanced_accuracy_score(ys, preds))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="cpu: fp32 fallback for models too large to share the GPU (GIN).")
    p.add_argument("--skip-missing", action="store_true",
                   help="Skip (model, fold) pairs whose run has not finished yet.")
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--rules", nargs="+", default=["best", "top3", "final"],
                   choices=["best", "top3", "final"])
    p.add_argument("--out", type=_Path, default=_ROOT / "results/test_eval/predictions.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.out.read_text()) if args.out.is_file() else {}

    for fold in args.folds:
        datamodules: dict = {}

        def datamodule(root: _Path):
            if root not in datamodules:
                datamodules.clear()  # keep one graph root in RAM (~17 GB per fold)
                gc.collect()
                dm = GraphSlideDataModule(splits_csv=SPLITS, graphs_root=root, fold=fold,
                                          num_workers=4, attach_tile_labels=False)
                dm.setup(stage="fit")
                dm.setup(stage="test")
                # Materialize once per (fold, graph root): every model and rule
                # of this fold reuses the same graphs instead of re-reading
                # ~17 GB from disk per evaluation.
                # Batches from worker processes live in /dev/shm; clone them into
                # private memory so the shared segments are released at once
                # (caching the shm-backed batches reached 60 GB and triggered
                # the OOM killer on 09-28).
                datamodules[root] = ([b.clone() for b in dm.val_dataloader()],
                                     [b.clone() for b in dm.test_dataloader()])
            return datamodules[root]

        for name in args.models:
            kind, ckpt_dir, _ = MODELS[name]
            last = last_ckpt(kind, ckpt_dir, fold)
            if args.skip_missing and not last.is_file():
                logger.info("skip %s fold %d: %s not found", name, fold, last)
                continue
            best, best_score, last_epoch = callback_state(last)
            metrics_file = {"deform": ckpt_dir / f"fold_{fold}" / "metrics.json",
                            "deform_rev1": ckpt_dir / f"fold_{fold}" / "metrics.json",
                            "mil_rev1": ckpt_dir / f"fold_{fold}" / "fold_metrics.json",
                            "gnn_rev1": ckpt_dir / f"fold_{fold}" / "fold_metrics.json",
                            }.get(kind, ckpt_dir / f"fold_{fold}_metrics.json")
            stored_last_val = None
            if kind not in ("deform", "deform_rev1") and metrics_file.is_file():
                stored_last_val = json.loads(metrics_file.read_text()).get("val_balanced_accuracy")
            candidates = {"best": (best, best_score), "top3": (top3_latest(last), None)}
            if kind == "deform_rev1":
                final = last.parent / f"fold_{fold}.ckpt"  # saved by 04b since the revision
                if final.is_file():
                    candidates["final"] = (final, None)
            elif kind != "deform":
                candidates["final"] = (last, stored_last_val)
            for rule in args.rules:
                if rule not in candidates:
                    continue
                path, stored = candidates[rule]
                key = f"{name}/fold{fold}/{rule}"
                if key in results:
                    continue
                t0 = time.time()
                set_global_seed(LOCKED_SEEDS[fold])
                model, fwd = build(name)
                load_weights(model, path)
                val_batches, test_batches = datamodule(GRAPHS_OF.get(name, GRAPHS))
                val = infer(model, fwd, val_batches, device=args.device)
                test = infer(model, fwd, test_batches, device=args.device)
                results[key] = {
                    "model": name, "fold": fold, "rule": rule, "ckpt": portable_path(path),
                    "last_epoch": last_epoch,
                    "val_bacc_stored": stored, "val_bacc_rescored": val["bacc"],
                    "val": val, "test": test, "device": args.device,
                }
                logger.info("%-24s val %.4f (stored %s)  test %.4f  [%.0fs]", key,
                            val["bacc"], f"{stored:.4f}" if stored is not None else "-",
                            test["bacc"], time.time() - t0)
                args.out.write_text(json.dumps(results))
                del model
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    _sys.exit(main())
