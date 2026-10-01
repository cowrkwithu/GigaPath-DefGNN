#!/usr/bin/env python3
"""Phase 10.5 — Train one model on one fold.

Wraps :func:`src.training.build_trainer` + :class:`VetGigaGraphLitModule`
with the :class:`GraphSlideDataModule` Phase-11 dataloader. Loads the
chosen model (baseline or VetGigaGraph) via the appropriate forward-fn
adapter, fits one outer-CV fold end-to-end, and writes a metrics JSON
the Step-5 verifier consumes.

Usage:
    python scripts/04_train.py --model vetgigagraph --fold 0
    python scripts/04_train.py --model abmil --fold 2 --max-epochs 50
    python scripts/04_train.py --model abmil --fold 0 --fast-dev-run

References:
    Design: docs/02-design/03-architecture.md §6.1 (Training settings)
    Design: docs/02-design/features/vetgigagraph.do.md Phase 7 + 10.5 + 11
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch.nn as nn

from scripts._common import add_common_args, load_runtime
from src.models import VetGigaGraph
from src.models.baselines import BASELINE_REGISTRY
from src.models.gigapath_slide import production_slide_backbone_loader
from src.training import (
    GraphSlideDataModule,
    NaNGuard,
    VetGigaGraphLitModule,
    baseline_forward_fn,
    build_class_weighted_ce_loss,
    EpochHistory,
    build_early_stopping,
    build_model_checkpoint,
    build_trainer,
    build_wandb_logger,
    vetgigagraph_forward_fn,
)
from src.utils.errors import ConfigError

logger = logging.getLogger(__name__)

#: Model names accepted by ``--model``: 5 baselines + the proposed model.
SUPPORTED_MODELS = tuple(BASELINE_REGISTRY.keys()) + ("vetgigagraph",)

#: Fusion strategies that do NOT require the GigaPath slide encoder.
#: Used by ``_build_model`` for the Phase-12-fallback pre-check.
#: Cross-reference: ``src/models/fusion.py`` ``GnnOnlyFusion.uses_slide = False``.
_GNN_ONLY_STRATEGIES: frozenset[str] = frozenset({"gnn_only"})


def _build_slide_backbone_loader(
    cfg: Mapping[str, Any],
) -> Optional[Callable[[], nn.Module]]:
    """Pick the slide-backbone loader to thread into :func:`_build_model`.

    Returns ``None`` when ``cfg.model.fusion.strategy == "gnn_only"`` (the
    documented Phase-12 fallback; no slide encoder needed). For any other
    fusion strategy, returns
    :func:`src.models.gigapath_slide.production_slide_backbone_loader` —
    the zero-arg callable that downloads + initialises the LongNet slide
    encoder lazily inside :meth:`VetGigaGraph.from_config`. Lazy invocation
    avoids the ~345 MB HF download when the operator runs ``gnn_only``.
    """
    fusion_strategy = str(cfg["model"]["fusion"]["strategy"])
    if fusion_strategy in _GNN_ONLY_STRATEGIES:
        return None
    return production_slide_backbone_loader


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        default="vetgigagraph",
        help="Model to train (default: vetgigagraph). Baselines: " + ", ".join(BASELINE_REGISTRY),
    )
    p.add_argument(
        "--fold",
        type=int,
        choices=(0, 1, 2, 3, 4),
        required=True,
        help="Outer-CV fold index (0–4).",
    )
    p.add_argument(
        "--splits-csv",
        type=Path,
        default=None,
        help="Path to data/splits/cv5fold.csv (default: paths.splits/cv5fold.csv).",
    )
    p.add_argument(
        "--graphs-root",
        type=Path,
        default=None,
        help="Per-slide .pt root (default: paths.graphs/<graph.type>/).",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override training.max_epochs from config.",
    )
    p.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        help="Override training.mixed_precision (true/false/16-mixed/bf16-mixed).",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override paths.checkpoints.",
    )
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Per-fold metrics JSON output path (default: <out>/<model>/fold_<i>_metrics.json).",
    )
    p.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Lightning's single-batch smoke run.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0 for deterministic + smoke-friendly).",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging (requires WANDB_API_KEY in .env). "
             "Without this flag the run still satisfies U1 by writing a "
             "config.json sidecar next to the metrics JSON.",
    )
    p.add_argument(
        "--no-default-callbacks",
        action="store_true",
        help="Skip the locked-default callback set (NaNGuard + EarlyStopping + ModelCheckpoint). "
             "Smoke uses this; production runs should not.",
    )
    p.add_argument(
        "--aux-lambda",
        type=float,
        default=0.0,
        help="µPDCA #8 Phase C: auxiliary tile-classification loss weight. "
             "0 = single-task (default). >0 = multi-task: aux head reads "
             "per-tile GNN features and predicts 13-way CATCH tile labels "
             "with masked CE (unmapped tiles ignore_index=-1). Joint loss: "
             "L = L_slide + aux_lambda * L_tile_masked. Requires "
             "/data/cia_outputs/annotations/tile_labels/*.npy.",
    )
    return p


def _build_model(
    name: str,
    *,
    embed_dim: int,
    num_classes: int,
    cfg: Optional[Mapping[str, Any]] = None,
    slide_backbone_loader: Optional[Callable[[], nn.Module]] = None,
) -> nn.Module:
    """Construct the chosen model with embed_dim matching the data.

    For ``vetgigagraph`` the model is built from the merged top-level
    config via :meth:`VetGigaGraph.from_config` — every GNN / fusion /
    classifier hyperparameter is read from ``cfg.model.*``. The
    ``slide_backbone_loader`` plumb keeps the documented Phase-12 fallback
    (``fusion.strategy=gnn_only``, no slide-encoder construction) working
    without a loader; non-``gnn_only`` strategies raise ``ConfigError``
    until the sibling ``slide-encoder-injection`` micro-PDCA lands a
    production GigaPath loader.

    Baselines (5 MIL models) ignore ``cfg`` and ``slide_backbone_loader``;
    they take ``embed_dim`` directly.
    """
    if name == "vetgigagraph":
        if cfg is None:
            raise ConfigError(
                "_build_model('vetgigagraph') requires the merged config "
                "(pass cfg=rt.config from main())."
            )
        fusion_strategy = str(cfg["model"]["fusion"]["strategy"])
        if fusion_strategy not in _GNN_ONLY_STRATEGIES and slide_backbone_loader is None:
            raise ConfigError(
                f"fusion.strategy='{fusion_strategy}' requires a "
                f"slide_backbone_loader, but none was injected. Phase-12 "
                f"fallback is fusion.strategy='gnn_only'. See queued "
                f"micro-PDCA `slide-encoder-injection` for production wiring."
            )
        # D-13 invariant (slide-encoder-tests µPDCA): slide-using fusion
        # strategies require ``slide_encoder.proj_dim`` to match
        # ``gnn.output_dim`` because LearnableWeightedFusion / Concat / etc.
        # combine ``h_gnn[output_dim]`` with ``h_slide[proj_dim]``. Surface a
        # friendly ConfigError instead of a shape mismatch deep inside
        # ``fusion._require_pair``.
        if fusion_strategy not in _GNN_ONLY_STRATEGIES:
            proj_dim = int(cfg["model"]["slide_encoder"]["proj_dim"])
            output_dim = int(cfg["model"]["gnn"]["output_dim"])
            if proj_dim != output_dim:
                raise ConfigError(
                    f"fusion.strategy='{fusion_strategy}' requires "
                    f"model.slide_encoder.proj_dim == model.gnn.output_dim, "
                    f"got proj_dim={proj_dim} ≠ output_dim={output_dim}. "
                    f"Align both YAML keys (see slide-encoder-tests µPDCA D-13)."
                )
        return VetGigaGraph.from_config(
            cfg,
            slide_backbone_loader=slide_backbone_loader,
        )

    cls = BASELINE_REGISTRY[name]
    # Baselines accept embed_dim; pick a small hidden dim so they fit
    # comfortably whether the data is real (1536-d) or smoke (32-d).
    return cls(embed_dim=embed_dim, hidden_dim=min(256, embed_dim), num_classes=num_classes)


def _detect_embed_dim(graphs_root: Path) -> int:
    """Probe one .pt file to read ``data.x.shape[1]``."""
    import torch

    sample = next(iter(graphs_root.glob("*.pt")), None)
    if sample is None:
        raise FileNotFoundError(f"no .pt graphs under {graphs_root}")
    data = torch.load(sample, map_location="cpu", weights_only=False)
    return int(data.x.shape[1])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rt = load_runtime(args, out_subdir="checkpoints")
    cfg = rt.config

    logger.info(
        "train: model=%s fold=%d seed=%d gpus=%s out=%s",
        args.model, args.fold, args.seed, args.gpus, rt.out_dir,
    )

    splits_csv = args.splits_csv or Path(cfg["paths"]["splits"]) / "cv5fold.csv"
    if not splits_csv.exists():
        logger.error(
            "splits CSV %s not found — generate it via "
            "src.evaluation.cross_validation.make_5fold_splits first.",
            splits_csv,
        )
        return 1

    graph_type = str(cfg["graph"]["type"])
    graphs_root = args.graphs_root or Path(cfg["paths"]["graphs"]) / graph_type
    if not graphs_root.exists() or not any(graphs_root.glob("*.pt")):
        logger.error("no .pt graphs under %s — run scripts/03_build_graphs.py first.", graphs_root)
        return 1

    num_classes = int(cfg["project"]["num_classes"])
    embed_dim = _detect_embed_dim(graphs_root)
    logger.info("Detected embed_dim=%d from %s", embed_dim, graphs_root)

    model = _build_model(
        args.model,
        embed_dim=embed_dim,
        num_classes=num_classes,
        cfg=cfg,
        slide_backbone_loader=_build_slide_backbone_loader(cfg),
    )
    forward_fn = vetgigagraph_forward_fn if args.model == "vetgigagraph" else baseline_forward_fn

    train_cfg = cfg["training"]
    max_epochs = int(args.max_epochs if args.max_epochs is not None else train_cfg["max_epochs"])
    warmup_epochs = min(int(train_cfg["warmup_epochs"]), max(1, max_epochs - 1))

    # µPDCA #8 Phase C: multi-task switch. aux_lambda > 0 swaps the LitModule
    # to one that backprop's an auxiliary tile-classification loss alongside
    # the slide-classification loss. Only meaningful for ``vetgigagraph``
    # (baselines don't expose per-tile features).
    if args.aux_lambda > 0 and args.model == "vetgigagraph":
        from src.training import MultiTaskVetGigaGraphLitModule
        lit = MultiTaskVetGigaGraphLitModule(
            model=model,
            loss_fn=build_class_weighted_ce_loss(labels=None, num_classes=num_classes),
            forward_fn=forward_fn,
            learning_rate=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            num_classes=num_classes,
            aux_lambda=float(args.aux_lambda),
            n_tile_classes=13,
            tile_feature_dim=int(cfg["model"]["gnn"]["output_dim"]),
        )
        attach_tile_labels = True
        logger.info("Multi-task mode active: aux_lambda=%.3f, tile_feature_dim=%d",
                    args.aux_lambda, int(cfg["model"]["gnn"]["output_dim"]))
    else:
        lit = VetGigaGraphLitModule(
            model=model,
            loss_fn=build_class_weighted_ce_loss(labels=None, num_classes=num_classes),
            forward_fn=forward_fn,
            learning_rate=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            num_classes=num_classes,
        )
        attach_tile_labels = False
    dm = GraphSlideDataModule(
        splits_csv=splits_csv,
        graphs_root=graphs_root,
        fold=args.fold,
        num_workers=args.num_workers,
        attach_tile_labels=attach_tile_labels,
    )

    ckpt_dir = args.checkpoint_dir or rt.out_dir / args.model
    mixed_precision: bool | str = (
        args.mixed_precision if args.mixed_precision is not None
        else (bool(train_cfg.get("mixed_precision", True)) if not args.fast_dev_run else False)
    )

    # --- Run-config snapshot (U1: "config logged" validator) ---
    # Always write a JSON sidecar so U1 is satisfied even when WandB is
    # disabled or the API key is missing. The sidecar is the single
    # source of truth for what was run; WandB (when on) is duplicate.
    metrics_out = args.metrics_out or rt.out_dir / args.model / f"fold_{args.fold}_metrics.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    config_sidecar = metrics_out.with_name(metrics_out.stem.replace("_metrics", "_config") + ".json")
    config_snapshot = {
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "gpus": args.gpus,
        "max_epochs": max_epochs,
        "warmup_epochs": warmup_epochs,
        "mixed_precision": mixed_precision,
        "graph_type": graph_type,
        "embed_dim": embed_dim,
        "num_classes": num_classes,
        "splits_csv": str(splits_csv),
        "graphs_root": str(graphs_root),
        "ckpt_dir": str(ckpt_dir),
        "wandb_enabled": bool(args.wandb),
        "config_path": str(args.config),
    }
    config_sidecar.write_text(json.dumps(config_snapshot, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote run-config sidecar → %s", config_sidecar)

    # --- WandB logger (optional; best-effort) ---
    wandb_logger = None
    if args.wandb:
        wandb_logger = build_wandb_logger(
            project=args.wandb_project,
            name=f"{args.model}_fold{args.fold}_seed{args.seed}",
            config=config_snapshot,
            api_key_required=False,  # never block training on missing key in a run
        )

    # --- Default production callback set (NaNGuard + EarlyStopping + Top-3 ModelCheckpoint) ---
    callbacks: list = []
    if not args.no_default_callbacks:
        log_dir = Path(cfg["paths"].get("logs", rt.out_dir.parent / "logs"))
        callbacks = [
            NaNGuard(marker_dir=log_dir),
            build_early_stopping(),
            build_model_checkpoint(dirpath=ckpt_dir),
        ]
    callbacks.append(EpochHistory(Path(metrics_out).with_suffix(".history.json")))

    # ``deterministic=False`` is a deliberate trade-off: Lightning + the
    # PyG GAT path requires non-deterministic scatter ops. The L1
    # reproducibility level (single forward pass) is preserved by
    # set_global_seed(); L2 fold-level reproducibility is documented
    # to be exact within fp16 tolerance, not bit-identical, per
    # docs/02-design/08-statistics-reproducibility.md §5.2.
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
        accelerator="cpu" if str(args.gpus).lower() in ("0", "cpu") else "auto",
        devices="auto",
    )
    trainer.fit(lit, datamodule=dm)

    # Persist a metrics JSON for Step-5 verifier + 05_evaluate.
    metrics: dict[str, Any] = {
        k: float(v) for k, v in trainer.callback_metrics.items() if v is not None
    }
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote metrics → %s", metrics_out)

    ckpt_path = ckpt_dir / f"fold_{args.fold}.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    logger.info("Saved checkpoint → %s", ckpt_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
