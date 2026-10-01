#!/usr/bin/env bash
# Phase 2 ablation — K (number of deformable sampling offsets) sweep.
#
# Runs the deformable model with K ∈ {1, 2, 4, 8} (configurable via K_VALUES)
# across 5 patient-level CV folds with locked seeds, writing each run to
# `results/k_sweep/k<K>/fold_<i>/metrics.json`.
#
# K = 2 is the headline value (already in results/deformable/); re-running it
# here gives a clean sweep dataset where every cell of the table was produced
# by identical scripts + identical config + identical hardware in one batch.
# Pass K_VALUES="1 4 8" to skip the K=2 re-run if disk space matters.
#
# Memory caveat: K = 4 and K = 8 may OOM on the largest CATCH WSIs
# (~94 K tiles) on a 24 GiB RTX 3090. The script does not abort on
# per-fold OOM — it logs the failure and continues to the next (K, fold)
# combination. Aggregation skips folds with no metrics file.
#
# Usage:
#   bash scripts/08_k_sweep.sh                          # full 4-K × 5-fold sweep
#   K_VALUES="1 2 4" bash scripts/08_k_sweep.sh         # subset of K
#   FOLDS="0 1" bash scripts/08_k_sweep.sh              # subset of folds
#   K_VALUES=1 FOLDS=0 bash scripts/08_k_sweep.sh       # single cell (smoke)
#
# Expected wall-clock (RTX 3090, fp16):
#   K=1: ~30 h for 5 folds (slightly cheaper than K=2 headline at ~44 h)
#   K=2: already done in results/deformable/ — re-running takes ~44 h
#   K=4: ~70 h for 5 folds if it fits memory
#   K=8: ~120 h (likely OOM; will most cells fail)
# Full unfiltered sweep: 10+ days. The operator typically runs subsets.

set -uo pipefail

V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V2_ROOT"

CONFIG="${CONFIG:-configs/experiment_deformable.yaml}"
K_VALUES="${K_VALUES:-1 2 4 8}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EXTRA="${EXTRA:-}"
RUN_TAG="$(date '+%Y%m%d_%H%M%S')"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[k-sweep] === start $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "[k-sweep] K_VALUES='$K_VALUES'  FOLDS='$FOLDS'  config=$CONFIG  tag=$RUN_TAG"

for K in $K_VALUES; do
  for fold in $FOLDS; do
    cell_dir="results/k_sweep/k${K}/fold_${fold}"
    mkdir -p "$cell_dir"
    log_file="$cell_dir/full_stdout_${RUN_TAG}.log"
    ckpt_dir="$cell_dir/checkpoints"
    metrics_path="$cell_dir/metrics.json"

    if [[ -f "$metrics_path" ]] && [[ "${SKIP_DONE:-0}" == "1" ]]; then
      echo "[k-sweep] === skipping K=$K fold=$fold (metrics exists, SKIP_DONE=1) ==="
      continue
    fi

    echo "[k-sweep] === K=$K fold=$fold === $(date '+%H:%M:%S')"
    /usr/bin/time -v python3 scripts/04b_train_deformable.py \
        --config "$CONFIG" \
        --fold "$fold" \
        --num-offsets "$K" \
        --num-workers 4 \
        --wandb --wandb-project vetgigagraph_v2 \
        --metrics-out "$metrics_path" \
        --checkpoint-dir "$ckpt_dir" \
        $EXTRA \
        > "$log_file" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[k-sweep] !!! K=$K fold=$fold FAILED (exit $rc) — see $log_file" >&2
      echo "[k-sweep]     continuing to next cell"
    else
      bacc=$(python3 -c "import json; d=json.load(open('$metrics_path')); print(f'{d[\"val\"][\"val_balanced_accuracy\"]:.4f}')" 2>/dev/null || echo "?")
      echo "[k-sweep]     OK — val_bacc=$bacc"
    fi
  done
done

# Aggregate per-K summary
python3 - <<'PY'
import json, pathlib, statistics

root = pathlib.Path("results/k_sweep")
out_summary = root / "summary.json"
per_k = {}
for k_dir in sorted(root.glob("k*")):
    if not k_dir.is_dir():
        continue
    k = k_dir.name[1:]
    fold_files = sorted(k_dir.glob("fold_*/metrics.json"))
    rows = [json.loads(p.read_text()) for p in fold_files]
    val_metrics = [r["val"] for r in rows if r.get("val")]
    if not val_metrics:
        per_k[k] = {"n_folds": 0, "note": "no successful folds"}
        continue
    keys = sorted({kk for v in val_metrics for kk in v.keys()})
    agg = {}
    for kk in keys:
        vals = [v[kk] for v in val_metrics if kk in v and isinstance(v[kk], (int, float))]
        if not vals:
            continue
        agg[kk] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "fold_values": vals,
        }
    per_k[k] = {"n_folds": len(rows), "metrics": agg}

out_summary.parent.mkdir(parents=True, exist_ok=True)
out_summary.write_text(json.dumps({"sweep": "K (deformable num_offsets)",
                                    "per_K": per_k}, indent=2))
print(f"[k-sweep] wrote {out_summary}")
for k, summary in sorted(per_k.items()):
    if "metrics" not in summary:
        print(f"  K={k}: {summary['n_folds']} folds — {summary.get('note','')}")
        continue
    m = summary["metrics"].get("val_balanced_accuracy", {})
    print(f"  K={k}: {summary['n_folds']} folds  val_bacc={m.get('mean',float('nan')):.4f} ± {m.get('std',float('nan')):.4f}")
PY

echo "[k-sweep] === finished $(date '+%Y-%m-%d %H:%M:%S') ==="
