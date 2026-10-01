#!/usr/bin/env python3
"""Train the v2 deformable-attention model on one fold.

Mirrors v1's ``pw-vetGigagraph/scripts/04_train.py`` `main()` flow, replacing
only the model construction step with our `build_deformable_model`. Loads v1
default config + v2 override via OmegaConf.merge so a pure-overrides v2 YAML
suffices.

Usage:
    python scripts/04b_train_deformable.py --fold 0 --fast-dev-run
    python scripts/04b_train_deformable.py --fold 0 --config configs/experiment_deformable.yaml
"""

from __future__ import annotations

import os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
# After Phase B unification (2026-05-31), the formerly-separate v1 src/ tree
# lives under this repo's src/; what used to require V1_ROOT + V2_ROOT side
# by side now just needs the GigaPath-DefGNN repo root on sys.path. The
# extra src/-on-sys.path entry keeps the v2-style `from deformable_attention
# import X` (no `src.` prefix) statements working unchanged.
_sys.path.insert(0, str(_ROOT))
_sys.path.insert(0, str(_ROOT / "src"))

import argparse                                              # noqa: E402
import json                                                  # noqa: E402
import logging                                               # noqa: E402
from pathlib import Path                                     # noqa: E402

import torch                                                 # noqa: E402
from omegaconf import OmegaConf                              # noqa: E402

# v1
from src.utils.config import load_config, _validate_config   # noqa: E402
from src.utils.logger import setup_logging                   # noqa: E402
from src.utils.seed import set_global_seed                   # noqa: E402
from src.training import (                                   # noqa: E402
    GraphSlideDataModule,
    NaNGuard,
    VetGigaGraphLitModule,
    build_class_weighted_ce_loss,
    build_early_stopping,
    build_model_checkpoint,
    build_trainer,
    build_wandb_logger,
    vetgigagraph_forward_fn,
)
from src.models.gigapath_slide import production_slide_backbone_loader  # noqa: E402

# v2
from deformable_attention import build_deformable_model                 # noqa: E402

logger = logging.getLogger(__name__)

_V1_DEFAULT_CONFIG = _ROOT / "configs" / "default.yaml"
_GNN_ONLY_STRATEGIES = frozenset({"gnn_only"})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=_ROOT / "configs" / "experiment_deformable.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpus", type=str, default="auto")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    p.add_argument("--fold", type=int, choices=(0, 1, 2, 3, 4), required=True)
    p.add_argument("--splits-csv", type=Path, default=None)
    p.add_argument("--graphs-root", type=Path, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--mixed-precision", default=None,
                   help="'true' | 'false' | bf16-mixed | 16-mixed")
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--wandb", action="store_true", default=False,
                   help="Enable W&B logging (best-effort).")
    p.add_argument("--wandb-project", type=str, default="vetgigagraph_v2")
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--metrics-out", type=Path, default=None)
    p.add_argument("--num-offsets", type=int, default=None,
                   help="Override model.gnn.deformable.num_offsets (K). "
                        "Used by scripts/08_k_sweep.sh for the K ablation. "
                        "Default: value from config (currently 2).")
    p.add_argument("--knn-k", type=int, default=None,
                   help="Override model.gnn.deformable.knn_k. "
                        "For the k_NN ablation. Default: value from config (8).")
    p.add_argument("--offset-penalty-weight", type=float, default=None,
                   help="Override model.gnn.deformable.offset_penalty_weight. "
                        "For the offset-L2-penalty ablation. Default: config (1e-4).")
    p.add_argument("--offset-penalty-epochs", type=int, default=None,
                   help="Override model.gnn.deformable.offset_penalty_epochs. "
                        "How many epochs the offset L2 penalty applies for. "
                        "Default: config (20).")
    p.add_argument("--offset-mlp-depth", type=int, default=None,
                   help="Override model.gnn.deformable.offset_mlp_depth. "
                        "OffsetMLP depth ablation. Default: config (1).")
    p.add_argument("--offset-mlp-hidden", type=int, default=None,
                   help="Override model.gnn.deformable.offset_mlp_hidden. "
                        "OffsetMLP hidden-dim ablation. Default: config (64).")
    p.add_argument("--no-default-callbacks", action="store_true")
    return p


def load_merged_cfg(config_path: Path):
    """Load v1 default, merge v2 override, re-validate."""
    base = load_config(_V1_DEFAULT_CONFIG)
    if config_path and Path(config_path) != _V1_DEFAULT_CONFIG:
        override = OmegaConf.load(config_path)
        merged = OmegaConf.merge(base, override)
        _validate_config(merged)
        logger.info("Merged config: base=%s + override=%s", _V1_DEFAULT_CONFIG, config_path)
        return merged
    return base


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(level=args.log_level)
    cfg = load_merged_cfg(args.config)
    # Ablation overrides — CLI takes precedence over the config so the
    # K-sweep / k_NN-sweep scripts don't need to write temp YAMLs.
    if args.num_offsets is not None:
        cfg["model"]["gnn"]["deformable"]["num_offsets"] = int(args.num_offsets)
        logger.info("[ablation] CLI override num_offsets (K) = %d",
                    args.num_offsets)
    if args.knn_k is not None:
        cfg["model"]["gnn"]["deformable"]["knn_k"] = int(args.knn_k)
        logger.info("[ablation] CLI override knn_k = %d", args.knn_k)
    if args.offset_penalty_weight is not None:
        cfg["model"]["gnn"]["deformable"]["offset_penalty_weight"] = float(args.offset_penalty_weight)
        logger.info("[ablation] CLI override offset_penalty_weight = %g",
                    args.offset_penalty_weight)
    if args.offset_penalty_epochs is not None:
        cfg["model"]["gnn"]["deformable"]["offset_penalty_epochs"] = int(args.offset_penalty_epochs)
        logger.info("[ablation] CLI override offset_penalty_epochs = %d",
                    args.offset_penalty_epochs)
    if args.offset_mlp_depth is not None:
        cfg["model"]["gnn"]["deformable"]["offset_mlp_depth"] = int(args.offset_mlp_depth)
        logger.info("[ablation] CLI override offset_mlp_depth = %d",
                    args.offset_mlp_depth)
    if args.offset_mlp_hidden is not None:
        cfg["model"]["gnn"]["deformable"]["offset_mlp_hidden"] = int(args.offset_mlp_hidden)
        logger.info("[ablation] CLI override offset_mlp_hidden = %d",
                    args.offset_mlp_hidden)
    # Per-fold seed (locked schedule)
    seeds = list(cfg["cross_validation"]["seeds"])
    fold_seed = seeds[args.fold]
    set_global_seed(fold_seed)
    logger.info("[v2] fold=%d seed=%d project=%s", args.fold, fold_seed,
                cfg["logging"].get("project", args.wandb_project))

    splits_csv = args.splits_csv or Path(cfg["paths"]["splits"]) / "cv5fold.csv"
    if not splits_csv.exists():
        logger.error("splits CSV missing: %s", splits_csv)
        return 1
    graph_type = str(cfg["graph"]["type"])
    graphs_root = args.graphs_root or Path(cfg["paths"]["graphs"]) / graph_type
    if not graphs_root.exists() or not any(graphs_root.glob("*.pt")):
        logger.error("no .pt graphs under %s — run v1 scripts/03_build_graphs.py first.", graphs_root)
        return 1

    num_classes = int(cfg["project"]["num_classes"])

    # --- Build deformable model (replaces v1's VetGigaGraph.from_config) ---
    fusion_strategy = str(cfg["model"]["fusion"]["strategy"])
    slide_loader = production_slide_backbone_loader if fusion_strategy not in _GNN_ONLY_STRATEGIES else None
    model = build_deformable_model(cfg, slide_backbone_loader=slide_loader)
    logger.info("[v2] built model with deformable GNN stack: %s",
                type(model.gnn).__name__)

    # --- Lightning module (v1's same signature) ---
    train_cfg = cfg["training"]
    max_epochs = int(args.max_epochs if args.max_epochs is not None else train_cfg["max_epochs"])
    warmup_epochs = min(int(train_cfg["warmup_epochs"]), max(1, max_epochs - 1))
    lit = VetGigaGraphLitModule(
        model=model,
        loss_fn=build_class_weighted_ce_loss(labels=None, num_classes=num_classes),
        forward_fn=vetgigagraph_forward_fn,
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        num_classes=num_classes,
    )

    # --- DataModule ---
    dm = GraphSlideDataModule(
        splits_csv=splits_csv,
        graphs_root=graphs_root,
        fold=args.fold,
        num_workers=args.num_workers,
        attach_tile_labels=False,
    )

    # --- Output dirs ---
    out_root = _ROOT / "results" / "deformable" / f"fold_{args.fold}"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.checkpoint_dir or out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mixed_precision: bool | str
    if args.mixed_precision is not None:
        if args.mixed_precision.lower() in ("true", "false"):
            mixed_precision = args.mixed_precision.lower() == "true"
        else:
            mixed_precision = args.mixed_precision
    elif args.fast_dev_run:
        mixed_precision = False
    else:
        mixed_precision = bool(train_cfg.get("mixed_precision", True))

    # --- Run-config snapshot ---
    config_snapshot = {
        "model": "vetgigagraph_v2_deformable",
        "fold": args.fold,
        "seed": fold_seed,
        "gpus": args.gpus,
        "max_epochs": max_epochs,
        "warmup_epochs": warmup_epochs,
        "mixed_precision": str(mixed_precision),
        "graph_type": graph_type,
        "num_classes": num_classes,
        "splits_csv": str(splits_csv),
        "graphs_root": str(graphs_root),
        "ckpt_dir": str(ckpt_dir),
        "wandb_enabled": bool(args.wandb),
        "config_path": str(args.config),
    }
    (out_root / "config.json").write_text(json.dumps(config_snapshot, indent=2, sort_keys=True))
    logger.info("[v2] config snapshot → %s", out_root / "config.json")

    # --- W&B (optional) ---
    wandb_logger = None
    if args.wandb:
        wandb_logger = build_wandb_logger(
            project=args.wandb_project,
            name=f"deformable_fold{args.fold}_seed{fold_seed}",
            config=config_snapshot,
            api_key_required=False,
        )

    # --- Callbacks ---
    callbacks = []
    if not args.no_default_callbacks:
        log_dir = Path(cfg["paths"].get("logs", out_root / "logs"))
        callbacks = [
            NaNGuard(marker_dir=log_dir),
            build_early_stopping(),
            build_model_checkpoint(dirpath=ckpt_dir),
        ]

    # --- Trainer ---
    trainer = build_trainer(
        max_epochs=max_epochs,
        accumulate_grad_batches=int(train_cfg.get("accumulation_steps", 1)),
        gradient_clip_val=float(train_cfg.get("gradient_clip_val", 1.0)),
        mixed_precision=mixed_precision,
        callbacks=callbacks,
        logger_obj=wandb_logger,
        checkpoint_dir=ckpt_dir,
        deterministic=False,
        fast_dev_run=args.fast_dev_run,
    )

    # --- Train ---
    trainer.fit(lit, datamodule=dm)

    # Final weights after early stopping, saved before validate(ckpt_path="best")
    # swaps the best checkpoint in. Matches 04_train.py's fold_<i>.ckpt so the
    # "final" checkpoint rule can be applied to this model too.
    if not args.fast_dev_run:
        final_path = ckpt_dir / f"fold_{args.fold}.ckpt"
        trainer.save_checkpoint(str(final_path))
        logger.info("[v2] saved final weights → %s", final_path)

    # --- Validate at best ckpt (skip if fast-dev-run produced no ckpt) ---
    val_results = []
    try:
        val_results = trainer.validate(lit, datamodule=dm, ckpt_path="best")
    except Exception as e:
        logger.warning("[v2] validate(ckpt='best') failed: %s — falling back to current weights", e)
        try:
            val_results = trainer.validate(lit, datamodule=dm)
        except Exception as e2:
            logger.error("[v2] validate failed: %s", e2)

    summary = {
        "model": "vetgigagraph_v2_deformable",
        "fold": args.fold,
        "seed": fold_seed,
        "val": val_results[0] if val_results else {},
    }
    metrics_path = args.metrics_out or (out_root / "metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2))
    logger.info("[v2] wrote %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
