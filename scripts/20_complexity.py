#!/usr/bin/env python3
"""Parameters, FLOPs, latency and peak memory for every aggregator (R1-2).

Answers Reviewer 1 point 2 (1st revision, applsci-4523976). All twelve
aggregators run on the same frozen Prov-GigaPath tile features, so the
encoder cost (identical for every model) is reported once and the table
compares the trainable heads.

* Parameters: trainable parameters of the aggregator head.
* FLOPs: one inference forward pass, counted by
  ``torch.utils.flop_counter.FlopCounterMode`` (2 FLOPs per multiply-add).
  It counts dense matrix products (mm / addmm / bmm / linear / SDPA), which
  dominate every model here. Sparse scatter aggregation (PyG message passing)
  and the ``torch_cluster.knn`` search used by GigaPath-DefGNN are NOT
  counted; their cost is O(E·d) and O(N·K·k_NN) respectively, and they are
  reported analytically in the manuscript.
* Latency and peak memory: fp16 autocast inference on one WSI, median of
  ``--repeats`` timed runs after one warm-up; peak memory is this process's
  ``max_memory_allocated``.

Two reference WSIs: the one closest to the median tile count (MCT_42_1,
26,126 tiles) and the largest (TRB_10_3, 93,973 tiles).

Run:  python3 scripts/20_complexity.py
Out:  results/complexity/complexity.json, results/complexity/complexity.md
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT))
from src.utils.env import output_dir  # noqa: E402
_sys.path.insert(0, str(_ROOT / "src"))

import argparse                                             # noqa: E402
import importlib.util                                       # noqa: E402
import json                                                 # noqa: E402
import statistics                                           # noqa: E402
import time                                                 # noqa: E402

import torch                                                # noqa: E402
from torch.utils.flop_counter import FlopCounterMode        # noqa: E402

from src.models.baselines import BASELINE_REGISTRY          # noqa: E402

GRAPHS = output_dir() / "graphs/dual_edge"
WSIS = {"median": "MCT_42_1", "largest": "TRB_10_3"}
ORDER = ["gcn", "gin", "graphsage", "gat", "abmil", "dsmil", "transmil",
         "clam_sb", "clam_mb", "clam_sb_inst", "clam_mb_inst", "acmil", "wikg", "defgnn"]
NAMES = {"gcn": "GCN", "gin": "GIN", "graphsage": "GraphSAGE", "gat": "GAT",
         "abmil": "ABMIL", "dsmil": "DSMIL", "transmil": "TransMIL",
         "clam_sb": "CLAM-SB", "clam_mb": "CLAM-MB", "clam_sb_inst": "CLAM-SB (inst. loss)",
         "clam_mb_inst": "CLAM-MB (inst. loss)", "acmil": "ACMIL",
         "wikg": "WiKG", "defgnn": "GigaPath-DefGNN"}
ENCODER_PARAMS = 1_134_953_984  # Prov-GigaPath tile encoder (frozen); counted from timm hf_hub:prov-gigapath/prov-gigapath


def _test_eval_module():
    spec = importlib.util.spec_from_file_location("test_eval", _ROOT / "scripts" / "18_test_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(name: str, te):
    """Model + forward-fn, built exactly as in the test evaluation."""
    if name in ("acmil", "wikg"):
        from src.training.dataset import baseline_forward_fn
        model = BASELINE_REGISTRY[name](embed_dim=1536, hidden_dim=256, num_classes=7)
        return model, baseline_forward_fn
    return te.build(name)


def measure(model, fwd, data, repeats: int) -> dict:
    model.eval().cuda()
    data = data.cuda()
    args, _ = fwd(data)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            model(*args)
        flops = fc.get_total_flops()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        times = []
        for i in range(repeats + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                model(*args)
            torch.cuda.synchronize()
            if i:  # first run is warm-up
                times.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() - base
    return {"gflops": flops / 1e9, "latency_ms": 1e3 * statistics.median(times),
            "peak_mem_mib": peak / 2**20}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=ORDER)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", type=_Path, default=_ROOT / "results/complexity/complexity.json")
    args = p.parse_args()

    # nn.TransformerEncoderLayer's inference fast path is a fused kernel the
    # FLOP counter cannot see; disable it so TransMIL's attention is counted.
    torch.backends.mha.set_fastpath_enabled(False)
    te = _test_eval_module()
    graphs = {k: torch.load(GRAPHS / f"{sid}.pt", weights_only=False) for k, sid in WSIS.items()}
    out = {"wsis": {k: {"slide_id": sid, "tiles": int(graphs[k].num_nodes),
                        "edges": int(graphs[k].edge_index.shape[1])} for k, sid in WSIS.items()},
           "encoder_params": ENCODER_PARAMS, "gpu": torch.cuda.get_device_name(0), "models": {}}
    for name in args.models:
        torch.manual_seed(0)
        model, fwd = build(name, te)
        row = {"params": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}
        for k, g in graphs.items():
            row[k] = measure(model, fwd, g, args.repeats)
        out["models"][name] = row
        print(f"{NAMES[name]:16s} params {row['params']:>9,d}  "
              + "  ".join(f"{k}: {row[k]['gflops']:8.2f} GFLOPs {row[k]['latency_ms']:8.1f} ms "
                          f"{row[k]['peak_mem_mib']:7.0f} MiB" for k in graphs), flush=True)
        del model
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.is_file():  # keep models not re-measured in this run
        prev = json.loads(args.out.read_text())
        out["models"] = {**prev.get("models", {}), **out["models"]}
    args.out.write_text(json.dumps(out, indent=1))
    args.models = [m for m in ORDER if m in out["models"]]
    w = out["wsis"]
    lines = [f"| Model | Trainable params | GFLOPs ({w['median']['tiles']:,} tiles) | "
             f"GFLOPs ({w['largest']['tiles']:,} tiles) | Latency ms (median / largest) | "
             "Peak MiB (median / largest) |", "|---|---|---|---|---|---|"]
    for name in args.models:
        r = out["models"][name]
        lines.append(f"| {NAMES[name]} | {r['params']:,} | {r['median']['gflops']:.1f} | "
                     f"{r['largest']['gflops']:.1f} | {r['median']['latency_ms']:.0f} / "
                     f"{r['largest']['latency_ms']:.0f} | {r['median']['peak_mem_mib']:.0f} / "
                     f"{r['largest']['peak_mem_mib']:.0f} |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
