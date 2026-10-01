#!/usr/bin/env bash
# Run all 5 folds of the v2 deformable-attention baseline sequentially.
# Locked seeds {42, 123, 456, 789, 1024} are read from configs/experiment_deformable.yaml.
#
# Usage:
#   bash scripts/run_deformable_cv.sh                        # full 5-fold
#   bash scripts/run_deformable_cv.sh --fast-dev-run         # smoke (1 batch/fold)
#   FOLDS="0 1" bash scripts/run_deformable_cv.sh            # subset

set -uo pipefail

V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V2_ROOT"

CONFIG="${CONFIG:-configs/experiment_deformable.yaml}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EXTRA="${*:-}"
RUN_START=$(date '+%Y-%m-%d %H:%M:%S')
RUN_TAG="$(date '+%Y%m%d_%H%M%S')"

# Memory fragmentation guard — required for the deformable kernel on 24 GiB.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# W&B mode: 'online' requires WANDB_API_KEY; 'offline' writes locally; 'disabled'
# skips the import entirely. Default offline so 4-5 day runs don't need network
# or a key, and so the per-step curves are still available for post-hoc analysis.
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[run-cv] === start $RUN_START ==="
echo "[run-cv] config=$CONFIG folds='$FOLDS' tag=$RUN_TAG"

for fold in $FOLDS; do
  fold_dir="results/deformable/fold_${fold}"
  mkdir -p "$fold_dir"
  log_file="$fold_dir/full_stdout_${RUN_TAG}.log"
  ckpt_dir="$fold_dir/checkpoints"
  metrics_path="$fold_dir/metrics.json"

  echo "[run-cv] === fold $fold === $(date '+%H:%M:%S')"
  echo "[run-cv] log → $log_file"

  /usr/bin/time -v python3 scripts/04b_train_deformable.py \
      --config "$CONFIG" \
      --fold "$fold" \
      --num-workers 4 \
      --wandb \
      --wandb-project vetgigagraph_v2 \
      --metrics-out "$metrics_path" \
      --checkpoint-dir "$ckpt_dir" \
      $EXTRA \
      > "$log_file" 2>&1
  rc=$?
  echo "[run-cv] fold $fold exit=$rc at $(date '+%H:%M:%S')"

  if [[ $rc -ne 0 ]]; then
    echo "[run-cv] !!! fold $fold FAILED — see $log_file" >&2
    echo "[run-cv] continuing with next fold (other folds may still succeed)" >&2
  fi
done

# Aggregate across folds and write a summary the comparison script reads.
python3 - <<'PY'
import json, pathlib, statistics
root = pathlib.Path("results/deformable")
fold_files = sorted(root.glob("fold_*/metrics.json"))
if not fold_files:
    raise SystemExit("[run-cv] no fold metrics found; aggregation skipped.")
rows = [json.loads(p.read_text()) for p in fold_files]
val_metrics = [r["val"] for r in rows if r.get("val")]
keys = sorted({k for v in val_metrics for k in v.keys()})
agg = {}
for k in keys:
    vals = [v[k] for v in val_metrics if k in v]
    if not vals or not all(isinstance(x, (int, float)) for x in vals):
        continue
    agg[k] = {
        "mean": statistics.mean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "fold_values": vals,
    }
summary_path = root / "summary.json"
summary_path.write_text(json.dumps({
    "model": "vetgigagraph_v2_deformable",
    "n_folds": len(rows),
    "metrics": agg,
}, indent=2))
print(f"[run-cv] wrote {summary_path}")
PY

echo "[run-cv] === finished $(date '+%Y-%m-%d %H:%M:%S') ==="
