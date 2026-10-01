#!/usr/bin/env python3
"""Capture frozen v1 reference metrics for fair v2 comparison.

The v1 production GAT baseline has already been trained and logged. To compare
v2 deformable attention to it on identical 5-fold patient-level splits, we
need stable per-fold metric JSONs in `results/v1_reference/`. This script
either:

  (a) regenerates them by running v1's `scripts/05_evaluate.py` on the saved
      best checkpoints under `/data/cia_outputs/checkpoints/`, OR
  (b) extracts them from saved W&B run files (`/data/cia_outputs/wandb/`).

Mode (a) is more authoritative but slower (requires reloading checkpoints).
Mode (b) is fast but depends on the W&B local files being intact.

Usage:
    # (a) Re-evaluate v1 production GAT from saved checkpoints
    python scripts/01_capture_v1_reference.py --mode reevaluate \
        --v1-checkpoints /data/cia_outputs/checkpoints/production

    # (b) Pull metrics from W&B local runs
    python scripts/01_capture_v1_reference.py --mode wandb \
        --wandb-root /data/cia_outputs/wandb \
        --run-tag production_gat
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))
from src.utils.env import output_dir, v1_root  # noqa: E402
V1_ROOT = v1_root()

logger = logging.getLogger(__name__)


def _ensure_v1_root() -> None:
    if not V1_ROOT.exists():
        raise SystemExit(
            f"V1 repo not found at {V1_ROOT}; set V1_ROOT env var."
        )


def mode_reevaluate(args: argparse.Namespace) -> None:
    """Run v1's 05_evaluate.py per fold and copy outputs into v1_reference/."""
    _ensure_v1_root()
    out_root = args.out_root or (V2_ROOT / "results" / "v1_reference")
    out_root.mkdir(parents=True, exist_ok=True)

    for fold in range(5):
        ckpt_dir = args.v1_checkpoints / f"fold_{fold}"
        if not ckpt_dir.exists():
            logger.warning("[capture] fold %d ckpt dir missing: %s", fold, ckpt_dir)
            continue
        fold_out = out_root / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(V1_ROOT / "scripts" / "05_evaluate.py"),
            "--fold", str(fold),
            "--ckpt-dir", str(ckpt_dir),
            "--out", str(fold_out),
            "--config", str(V1_ROOT / "configs" / "default.yaml"),
        ]
        logger.info("[capture] fold %d: %s", fold, " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=V1_ROOT)


def mode_wandb(args: argparse.Namespace) -> None:
    """Extract per-fold metrics from W&B offline run dirs."""
    try:
        from wandb.sdk.lib.import_hooks import wandb        # type: ignore
        # Prefer the dedicated parser:
        import wandb                                         # type: ignore
    except Exception:
        wandb = None    # type: ignore

    wandb_root = args.wandb_root
    if not wandb_root.exists():
        raise SystemExit(f"W&B root not found: {wandb_root}")
    out_root = args.out_root or (V2_ROOT / "results" / "v1_reference")
    out_root.mkdir(parents=True, exist_ok=True)

    tag = args.run_tag
    metric_re = re.compile(r'"val_(?P<k>[a-z_]+)":\s*(?P<v>[0-9.]+)')
    fold_metrics: dict[int, dict[str, float]] = {}
    for run_dir in sorted(wandb_root.glob("*run-*")):
        meta_path = run_dir / "files" / "wandb-metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        tags = meta.get("tags", []) or []
        if tag and tag not in tags and tag not in meta.get("name", ""):
            continue
        # Locate the val metrics from the summary file
        summary_path = run_dir / "files" / "wandb-summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        fold = meta.get("config", {}).get("cross_validation.fold")
        if fold is None:
            m = re.search(r"fold[_=](\d+)", run_dir.name)
            fold = int(m.group(1)) if m else None
        if fold is None:
            continue
        fold = int(fold)
        per = {k: v for k, v in summary.items() if k.startswith("val_") and isinstance(v, (int, float))}
        fold_metrics[fold] = per

    for fold, m in sorted(fold_metrics.items()):
        out = out_root / f"fold_{fold}" / "metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": "v1_production_gat", "fold": fold, "val": m}, indent=2))
        logger.info("[capture] wrote %s", out)

    if not fold_metrics:
        raise SystemExit("[capture] no matching W&B runs found.")


def write_summary(out_root: Path) -> None:
    """Aggregate the per-fold JSONs into a summary that scripts/05b reads."""
    folds = sorted(out_root.glob("fold_*/metrics.json"))
    rows = [json.loads(p.read_text()) for p in folds]
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.get("val", {}).keys()})
    agg: dict[str, dict] = {}
    for k in keys:
        vals = [float(r["val"][k]) for r in rows if k in r.get("val", {})]
        if not vals:
            continue
        agg[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "fold_values": vals,
        }
    summary = out_root / "v1_baseline_summary.json"
    summary.write_text(json.dumps({
        "model": "v1_production_gat",
        "n_folds": len(rows),
        "metrics": agg,
    }, indent=2))
    logger.info("[capture] wrote %s", summary)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("reevaluate", "wandb"), required=True)
    parser.add_argument("--v1-checkpoints", type=Path,
                        default=output_dir() / "checkpoints/production",
                        help="v1 best-ckpt root (mode=reevaluate).")
    parser.add_argument("--wandb-root", type=Path,
                        default=output_dir() / "wandb",
                        help="v1 W&B local runs root (mode=wandb).")
    parser.add_argument("--run-tag", type=str, default="production_gat",
                        help="Tag or name substring filter (mode=wandb).")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Where to write per-fold JSONs (default: results/v1_reference).")
    args = parser.parse_args()

    if args.mode == "reevaluate":
        mode_reevaluate(args)
    else:
        mode_wandb(args)

    write_summary(args.out_root or (V2_ROOT / "results" / "v1_reference"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
