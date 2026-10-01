#!/usr/bin/env bash
# Phase 2 ablation — k_NN (soft K-NN neighbors per query) sweep.
#
# Runs the deformable model with k_NN ∈ {4, 8, 16} (configurable via KNN_VALUES)
# across 5 patient-level CV folds with locked seeds. K (num_offsets) stays at the
# headline value (2) so the comparison isolates the k_NN dimension.
#
# Memory note: k_NN scales the `feats[E, D]` gather in the deformable sampling
# kernel — peak memory is roughly linear in k_NN. k_NN = 16 may be tight on a
# 24 GiB GPU at K = 2 (the chunked kernel mitigates but doesn't eliminate the
# scale). Smoke (1 fold, 1 epoch) recommended first via:
#   KNN_VALUES=16 FOLDS=0 EXTRA="--max-epochs 1" bash scripts/09_knn_sweep.sh
#
# Usage:
#   bash scripts/09_knn_sweep.sh                              # full sweep
#   KNN_VALUES="4 8" bash scripts/09_knn_sweep.sh             # subset
#   FOLDS="0 1" bash scripts/09_knn_sweep.sh                  # subset of folds
#   KNN_VALUES=4 FOLDS=0 bash scripts/09_knn_sweep.sh         # single cell
#
# Expected wall-clock (RTX 3090, fp16, K=2 headline ~44h for 5 folds):
#   k_NN=4:  ~30-35 h for 5 folds (sparser sampling, cheaper)
#   k_NN=8:  ~44 h for 5 folds  (= headline — already in results/deformable/)
#   k_NN=16: ~60-80 h for 5 folds if it fits

set -uo pipefail
V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V2_ROOT"

CONFIG="${CONFIG:-configs/experiment_deformable.yaml}"
KNN_VALUES="${KNN_VALUES:-4 8 16}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EXTRA="${EXTRA:-}"
RUN_TAG="$(date '+%Y%m%d_%H%M%S')"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[knn-sweep] === start $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "[knn-sweep] KNN_VALUES='$KNN_VALUES'  FOLDS='$FOLDS'  config=$CONFIG"

for knn in $KNN_VALUES; do
  for fold in $FOLDS; do
    cell_dir="results/knn_sweep/knn${knn}/fold_${fold}"
    mkdir -p "$cell_dir"
    log_file="$cell_dir/full_stdout_${RUN_TAG}.log"
    ckpt_dir="$cell_dir/checkpoints"
    metrics_path="$cell_dir/metrics.json"

    if [[ -f "$metrics_path" ]] && [[ "${SKIP_DONE:-0}" == "1" ]]; then
      echo "[knn-sweep] === skipping k_NN=$knn fold=$fold (exists) ==="
      continue
    fi

    echo "[knn-sweep] === k_NN=$knn fold=$fold === $(date '+%H:%M:%S')"
    /usr/bin/time -v python3 scripts/04b_train_deformable.py \
        --config "$CONFIG" --fold "$fold" --knn-k "$knn" \
        --num-workers 4 --wandb --wandb-project vetgigagraph_v2 \
        --metrics-out "$metrics_path" --checkpoint-dir "$ckpt_dir" \
        $EXTRA > "$log_file" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[knn-sweep] !!! k_NN=$knn fold=$fold FAILED (exit $rc) — see $log_file" >&2
    else
      bacc=$(python3 -c "import json; d=json.load(open('$metrics_path')); print(f'{d[\"val\"][\"val_balanced_accuracy\"]:.4f}')" 2>/dev/null || echo "?")
      echo "[knn-sweep]     OK — val_bacc=$bacc"
    fi
  done
done

# Aggregate
python3 - <<'PY'
import json, pathlib, statistics
root = pathlib.Path("results/knn_sweep")
per_knn = {}
for d in sorted(root.glob("knn*")):
    if not d.is_dir():
        continue
    k = d.name[3:]
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("fold_*/metrics.json"))]
    val_metrics = [r["val"] for r in rows if r.get("val")]
    if not val_metrics:
        per_knn[k] = {"n_folds": 0, "note": "no successful folds"}
        continue
    agg = {}
    for kk in {kk for v in val_metrics for kk in v.keys()}:
        vals = [v[kk] for v in val_metrics if kk in v and isinstance(v[kk], (int, float))]
        if vals:
            agg[kk] = {"mean": statistics.mean(vals),
                       "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                       "fold_values": vals}
    per_knn[k] = {"n_folds": len(rows), "metrics": agg}
out = root / "summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"sweep": "k_NN (deformable knn_k)", "per_knn": per_knn}, indent=2))
print(f"[knn-sweep] wrote {out}")
for k, s in sorted(per_knn.items(), key=lambda x: int(x[0])):
    if "metrics" not in s:
        print(f"  k_NN={k}: {s['n_folds']} folds — {s.get('note','')}"); continue
    m = s["metrics"].get("val_balanced_accuracy", {})
    print(f"  k_NN={k}: {s['n_folds']} folds  val_bacc={m.get('mean', float('nan')):.4f} ± {m.get('std', float('nan')):.4f}")
PY

echo "[knn-sweep] === finished $(date '+%Y-%m-%d %H:%M:%S') ==="
