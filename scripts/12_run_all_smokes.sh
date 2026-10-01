#!/usr/bin/env bash
# Phase 2 ablation — smoke harness for ALL remaining ablation dimensions.
#
# Runs ONE FOLD × ONE EPOCH of each non-headline cell across:
#   k_NN ∈ {4, 16}              (k_NN = 8 is headline)
#   offset_penalty ∈ {0, 1e-3}  (penalty = 1e-4 is headline)
#   MLP depth × hidden:
#     (depth=2, hidden=64)       (depth=1, hidden=64 is headline)
#     (depth=2, hidden=128)       (extreme)
# Total: 6 smoke cells × ~30 min = ~3 hours wall-clock.
#
# Smokes provide:
#   (1) Memory feasibility (OOM check) before committing to multi-day sweeps.
#   (2) val_bacc after 1 epoch as a coarse signal — extremely cheap and
#       informative for ruling out dead-zone configurations.
#
# PRECONDITION: do not run while another training job is using the GPU
# (e.g., the K=1 5-fold sweep dispatched 2026-05-26 14:39:33 KST). The script
# refuses to start if any 04b_train_deformable.py process is running.
#
# Usage:
#   bash scripts/12_run_all_smokes.sh           # all 6 smoke cells
#   SKIP_KNN=1 bash scripts/12_run_all_smokes.sh   # skip k_NN block
#
# After smokes, dispatch the full sweeps that smoke OK via:
#   bash scripts/09_knn_sweep.sh
#   bash scripts/10_offset_penalty_sweep.sh
#   bash scripts/11_mlp_depth_sweep.sh

set -uo pipefail
V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V2_ROOT"

if pgrep -f "04b_train_deformable" > /dev/null; then
  echo "[smokes] !!! Another 04b_train_deformable.py process is running."
  echo "[smokes]     Refusing to start to avoid GPU contention."
  ps -ef | grep "04b_train_deformable" | grep -v grep >&2
  exit 1
fi

SKIP_KNN="${SKIP_KNN:-0}"
SKIP_PENALTY="${SKIP_PENALTY:-0}"
SKIP_MLP="${SKIP_MLP:-0}"

COMMON_EXTRA="--max-epochs 1 --no-default-callbacks"

started=$(date '+%Y-%m-%d %H:%M:%S')
echo "[smokes] === start $started ==="

if [[ "$SKIP_KNN" != "1" ]]; then
  echo "[smokes] --- k_NN smoke (k_NN = 4, 16) ---"
  KNN_VALUES="4 16" FOLDS="0" EXTRA="$COMMON_EXTRA" \
    bash scripts/09_knn_sweep.sh
fi

if [[ "$SKIP_PENALTY" != "1" ]]; then
  echo "[smokes] --- offset penalty smoke (p = 0, 1e-3) ---"
  PENALTY_VALUES="0 1e-3" FOLDS="0" EXTRA="$COMMON_EXTRA" \
    bash scripts/10_offset_penalty_sweep.sh
fi

if [[ "$SKIP_MLP" != "1" ]]; then
  echo "[smokes] --- MLP depth × hidden smoke (depth=2 × hidden=64, 128) ---"
  MLP_DEPTHS="2" MLP_HIDDENS="64 128" FOLDS="0" EXTRA="$COMMON_EXTRA" \
    bash scripts/11_mlp_depth_sweep.sh
fi

finished=$(date '+%Y-%m-%d %H:%M:%S')
echo "[smokes] === finished $finished ==="
echo ""
echo "[smokes] Per-cell logs under results/{knn_sweep,offset_penalty_sweep,mlp_sweep}/"
echo "[smokes] If all cells finished OK, dispatch full sweeps (sequential, ~weeks):"
echo "  bash scripts/09_knn_sweep.sh           # k_NN sweep"
echo "  bash scripts/10_offset_penalty_sweep.sh # penalty sweep"
echo "  bash scripts/11_mlp_depth_sweep.sh     # MLP depth × hidden sweep"
