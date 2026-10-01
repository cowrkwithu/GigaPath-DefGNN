#!/usr/bin/env bash
# Phase 2 ablation — OffsetMLP depth × hidden-dim sweep.
#
# Cross-product:
#   offset_mlp_depth  ∈ {1, 2}
#   offset_mlp_hidden ∈ {32, 64, 128}
# → 6 cells × 5 folds = 30 runs.
#
# Tests whether OffsetMLP capacity matters. The headline (depth=1, hidden=64)
# is the minimal MLP. Larger depth/hidden = more capacity to predict offsets;
# does this help or overfit?
#
# Memory: OffsetMLP is fp32-promoted regardless of depth (per the stability
# guarantee in src/deformable_attention/layer.py); deeper/wider MLP adds
# negligible memory compared to the deformable sampling kernel itself.
# All 6 cells should fit at the K=2 headline memory budget.
#
# Wall-clock: ~44h per cell × 6 cells = ~264 h ≈ 11 days for the full sweep.
# Subset highly recommended; the most informative single contrast is
# (depth=1, hidden=64) vs (depth=2, hidden=128) — the extremes.
#
# Usage:
#   bash scripts/11_mlp_depth_sweep.sh                                    # full
#   MLP_DEPTHS="2" MLP_HIDDENS="128" bash scripts/11_mlp_depth_sweep.sh   # one extreme
#   MLP_DEPTHS="1 2" MLP_HIDDENS="64 128" FOLDS="0" bash …                # smoke

set -uo pipefail
V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V2_ROOT"

CONFIG="${CONFIG:-configs/experiment_deformable.yaml}"
MLP_DEPTHS="${MLP_DEPTHS:-1 2}"
MLP_HIDDENS="${MLP_HIDDENS:-32 64 128}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EXTRA="${EXTRA:-}"
RUN_TAG="$(date '+%Y%m%d_%H%M%S')"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[mlp-sweep] === start $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "[mlp-sweep] MLP_DEPTHS='$MLP_DEPTHS' × MLP_HIDDENS='$MLP_HIDDENS'  FOLDS='$FOLDS'"

for depth in $MLP_DEPTHS; do
  for hidden in $MLP_HIDDENS; do
    for fold in $FOLDS; do
      cell_dir="results/mlp_sweep/depth${depth}_hidden${hidden}/fold_${fold}"
      mkdir -p "$cell_dir"
      log_file="$cell_dir/full_stdout_${RUN_TAG}.log"
      ckpt_dir="$cell_dir/checkpoints"
      metrics_path="$cell_dir/metrics.json"

      if [[ -f "$metrics_path" ]] && [[ "${SKIP_DONE:-0}" == "1" ]]; then
        echo "[mlp-sweep] === skipping depth=$depth hidden=$hidden fold=$fold ==="
        continue
      fi

      echo "[mlp-sweep] === depth=$depth hidden=$hidden fold=$fold === $(date '+%H:%M:%S')"
      /usr/bin/time -v python3 scripts/04b_train_deformable.py \
          --config "$CONFIG" --fold "$fold" \
          --offset-mlp-depth "$depth" --offset-mlp-hidden "$hidden" \
          --num-workers 4 --wandb --wandb-project vetgigagraph_v2 \
          --metrics-out "$metrics_path" --checkpoint-dir "$ckpt_dir" \
          $EXTRA > "$log_file" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "[mlp-sweep] !!! depth=$depth hidden=$hidden fold=$fold FAILED (exit $rc) — see $log_file" >&2
      else
        bacc=$(python3 -c "import json; d=json.load(open('$metrics_path')); print(f'{d[\"val\"][\"val_balanced_accuracy\"]:.4f}')" 2>/dev/null || echo "?")
        echo "[mlp-sweep]     OK — val_bacc=$bacc"
      fi
    done
  done
done

python3 - <<'PY'
import json, pathlib, statistics
root = pathlib.Path("results/mlp_sweep")
per_cell = {}
for d in sorted(root.glob("depth*_hidden*")):
    if not d.is_dir():
        continue
    cell = d.name
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("fold_*/metrics.json"))]
    val_metrics = [r["val"] for r in rows if r.get("val")]
    if not val_metrics:
        per_cell[cell] = {"n_folds": 0, "note": "no successful folds"}
        continue
    agg = {}
    for kk in {kk for v in val_metrics for kk in v.keys()}:
        vals = [v[kk] for v in val_metrics if kk in v and isinstance(v[kk], (int, float))]
        if vals:
            agg[kk] = {"mean": statistics.mean(vals),
                       "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                       "fold_values": vals}
    per_cell[cell] = {"n_folds": len(rows), "metrics": agg}
out = root / "summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"sweep": "OffsetMLP depth × hidden",
                            "per_cell": per_cell}, indent=2))
print(f"[mlp-sweep] wrote {out}")
for k, s in sorted(per_cell.items()):
    if "metrics" not in s:
        print(f"  {k}: {s['n_folds']} folds — {s.get('note','')}"); continue
    m = s["metrics"].get("val_balanced_accuracy", {})
    print(f"  {k}: {s['n_folds']} folds  val_bacc={m.get('mean',float('nan')):.4f} ± {m.get('std',float('nan')):.4f}")
PY

echo "[mlp-sweep] === finished $(date '+%Y-%m-%d %H:%M:%S') ==="
