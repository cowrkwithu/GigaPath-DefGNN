#!/usr/bin/env python3
"""How far do GigaPath-DefGNN's learned sampling offsets reach? (all test WSIs)

For every test WSI (pooled over the five folds; best-validation checkpoints,
as in 18_test_eval.py), capture each deformable layer's OffsetMLP output
Δ ∈ [N, K, 2] (per-slide normalized coordinates) and report, per layer:

* offset length in tile spacings (Δ rescaled by the slide's per-axis span,
  divided by the 256-px tile pitch);
* fraction of query points q = p + Δ outside the slide's tile bounding box;
* distance from each query point to its nearest tile, in tile spacings
  (how far the sampled position lies from any tissue).

Run:  python3 scripts/21_offset_stats.py
Out:  results/test_eval/offsets/offset_stats.json (+ .md summary)
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
_sys.path.insert(0, str(_ROOT / "src"))

import argparse                                             # noqa: E402
import importlib.util                                       # noqa: E402
import json                                                 # noqa: E402

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from scipy.spatial import cKDTree                           # noqa: E402

from src.training import GraphSlideDataModule              # noqa: E402

TILE_PX = 256


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=_Path, default=_ROOT / "results/test_eval/offsets/offset_stats.json")
    args = p.parse_args()

    spec = importlib.util.spec_from_file_location("test_eval", _ROOT / "scripts" / "18_test_eval.py")
    te = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(te)
    kind, ckpt_dir, _ = te.MODELS["defgnn"]

    rows = []
    for fold in args.folds:
        best, _, _ = te.callback_state(te.last_ckpt(kind, ckpt_dir, fold))
        model, fwd = te.build("defgnn")
        te.load_weights(model, best)
        model.eval().cuda()
        layers = [m for m in model.modules() if type(m).__name__ == "DeformableAttentionLayer"]
        captured: list[torch.Tensor] = []
        for layer in layers:
            layer.offset_mlp.register_forward_hook(lambda _m, _i, out: captured.append(out.float().cpu()))

        dm = GraphSlideDataModule(splits_csv=te.SPLITS, graphs_root=te.GRAPHS, fold=fold,
                                  num_workers=4, attach_tile_labels=False)
        dm.setup(stage="test")
        with torch.no_grad():
            for batch in dm.test_dataloader():
                captured.clear()
                batch = batch.cuda()
                a, _ = fwd(batch)
                with torch.autocast("cuda", dtype=torch.float16):
                    model(*a)
                # eval forward calls each OffsetMLP twice (once for bookkeeping);
                # both calls see identical inputs, so keep one per layer.
                per_layer = captured[::2] if len(captured) == 2 * len(layers) else captured
                xy = batch.pos.float().cpu().numpy()
                lo = xy.min(0)
                span = np.maximum(xy.max(0) - lo, 1.0)
                norm = (xy - lo) / span
                tree = cKDTree(norm * span / TILE_PX)          # tile-spacing units
                row = {"slide_id": str(batch.slide_id), "fold": fold,
                       "tumor_class": str(batch.slide_id).split("_")[0], "n_tiles": len(xy)}
                for li, off in enumerate(per_layer, start=1):
                    d = off.numpy()                            # [N, K, 2] normalized
                    length = np.linalg.norm(d * span / TILE_PX, axis=-1).ravel()
                    q = (norm[:, None, :] + d).reshape(-1, 2)
                    outside = np.mean((q < 0).any(1) | (q > 1).any(1))
                    nearest, _ = tree.query(q * span / TILE_PX)
                    row[f"layer{li}"] = {
                        "offset_tiles_median": float(np.median(length)),
                        "offset_tiles_p95": float(np.percentile(length, 95)),
                        "frac_query_outside_bbox": float(outside),
                        "query_to_nearest_tile_median": float(np.median(nearest)),
                        "frac_query_within_1_tile": float(np.mean(nearest <= 1.0)),
                    }
                rows.append(row)
        print(f"fold {fold}: {sum(r['fold'] == fold for r in rows)} WSIs", flush=True)
        del model
        torch.cuda.empty_cache()

    summary = {}
    for li in (1, 2):
        key = f"layer{li}"
        vals = {m: np.array([r[key][m] for r in rows if key in r]) for m in rows[0][key]}
        summary[key] = {m: {"median_over_wsis": float(np.median(v)),
                            "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}
                        for m, v in vals.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_wsis": len(rows), "summary": summary, "per_wsi": rows}, indent=1))
    lines = ["| Layer | Metric | Median over WSIs [IQR] |", "|---|---|---|"]
    for key, ms in summary.items():
        for m, s in ms.items():
            lines.append(f"| {key} | {m} | {s['median_over_wsis']:.3f} [{s['iqr'][0]:.3f}, {s['iqr'][1]:.3f}] |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
