#!/usr/bin/env bash
# Phase 2 ablation — offset L2 penalty weight sweep.
#
# Runs the deformable model with offset_penalty_weight ∈ {0, 1e-5, 1e-4, 1e-3}
# across 5 patient-level CV folds with locked seeds. The L2 penalty is applied
# to the OffsetMLP outputs for the first 20 epochs by default; this sweep tests
# whether that regularization actually matters (does the OffsetMLP collapse
# without it? does a stronger penalty help focal-tumor stability?).
#
# 0     = no offset regularization (does OffsetMLP collapse to zero? a key risk
#         called out in the layer design — see src/deformable_attention/layer.py
#         "Why offset L2 penalty for the first 20 epochs?" comment).
# 1e-5  = headline / 10 (light push).
# 1e-4  = headline (default).
# 1e-3  = headline × 10 (strong push toward identity — should hurt if too strong).
#
# Memory note: this sweep does NOT touch the memory-bottleneck dimensions
# (K, k_NN). All 4 penalty values use identical memory; expected wall-clock
# is identical (~44 h for 5 folds each ≈ 22 h × 4 = ~7 days for the full sweep).
#
# Usage:
#   bash scripts/10_offset_penalty_sweep.sh                          # full
#   PENALTY_VALUES="0 1e-4" bash scripts/10_offset_penalty_sweep.sh  # subset

set -uo pipefail
V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V2_ROOT"

CONFIG="${CONFIG:-configs/experiment_deformable.yaml}"
PENALTY_VALUES="${PENALTY_VALUES:-0 1e-5 1e-4 1e-3}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EXTRA="${EXTRA:-}"
RUN_TAG="$(date '+%Y%m%d_%H%M%S')"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[penalty-sweep] === start $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "[penalty-sweep] PENALTY_VALUES='$PENALTY_VALUES'  FOLDS='$FOLDS'"

for p_val in $PENALTY_VALUES; do
  # filename-safe label: 0, 1e-5, 1e-4, 1e-3 → 0, 1e-05, 1e-04, 1e-03
  p_label=$(python3 -c "x='$p_val'; print(x if x=='0' else f'{float(x):.0e}'.replace('+0','').replace('-0','-0'))")
  for fold in $FOLDS; do
    cell_dir="results/offset_penalty_sweep/p${p_label}/fold_${fold}"
    mkdir -p "$cell_dir"
    log_file="$cell_dir/full_stdout_${RUN_TAG}.log"
    ckpt_dir="$cell_dir/checkpoints"
    metrics_path="$cell_dir/metrics.json"

    if [[ -f "$metrics_path" ]] && [[ "${SKIP_DONE:-0}" == "1" ]]; then
      echo "[penalty-sweep] === skipping p=$p_val fold=$fold ==="
      continue
    fi

    echo "[penalty-sweep] === p=$p_val fold=$fold === $(date '+%H:%M:%S')"
    /usr/bin/time -v python3 scripts/04b_train_deformable.py \
        --config "$CONFIG" --fold "$fold" \
        --offset-penalty-weight "$p_val" \
        --num-workers 4 --wandb --wandb-project vetgigagraph_v2 \
        --metrics-out "$metrics_path" --checkpoint-dir "$ckpt_dir" \
        $EXTRA > "$log_file" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[penalty-sweep] !!! p=$p_val fold=$fold FAILED (exit $rc) — see $log_file" >&2
    else
      bacc=$(python3 -c "import json; d=json.load(open('$metrics_path')); print(f'{d[\"val\"][\"val_balanced_accuracy\"]:.4f}')" 2>/dev/null || echo "?")
      echo "[penalty-sweep]     OK — val_bacc=$bacc"
    fi
  done
done

python3 - <<'PY'
import json, pathlib, statistics
root = pathlib.Path("results/offset_penalty_sweep")
per_p = {}
for d in sorted(root.glob("p*")):
    if not d.is_dir():
        continue
    k = d.name[1:]
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("fold_*/metrics.json"))]
    val_metrics = [r["val"] for r in rows if r.get("val")]
    if not val_metrics:
        per_p[k] = {"n_folds": 0, "note": "no successful folds"}
        continue
    agg = {}
    for kk in {kk for v in val_metrics for kk in v.keys()}:
        vals = [v[kk] for v in val_metrics if kk in v and isinstance(v[kk], (int, float))]
        if vals:
            agg[kk] = {"mean": statistics.mean(vals),
                       "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                       "fold_values": vals}
    per_p[k] = {"n_folds": len(rows), "metrics": agg}
out = root / "summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"sweep": "offset_penalty_weight",
                            "per_weight": per_p}, indent=2))
print(f"[penalty-sweep] wrote {out}")
for k, s in sorted(per_p.items()):
    if "metrics" not in s:
        print(f"  p={k}: {s['n_folds']} folds — {s.get('note','')}"); continue
    m = s["metrics"].get("val_balanced_accuracy", {})
    print(f"  p={k}: {s['n_folds']} folds  val_bacc={m.get('mean',float('nan')):.4f} ± {m.get('std',float('nan')):.4f}")
PY

echo "[penalty-sweep] === finished $(date '+%Y-%m-%d %H:%M:%S') ==="
