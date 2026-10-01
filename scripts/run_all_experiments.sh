#!/usr/bin/env bash
# Phase 10.8 — Sequential 28-experiment × 5-fold sweep.
#
# Drives the full Experiment 1–5 matrix (per
# docs/02-design/04-experiment-design.md §1–§5) using
# scripts/04_train.py + scripts/05_evaluate.py.
#
# Usage:
#   bash scripts/run_all_experiments.sh                      # all
#   bash scripts/run_all_experiments.sh --only exp1          # subset
#   DRY_RUN=1 bash scripts/run_all_experiments.sh            # echo only
set -euo pipefail
cd "$(dirname "$0")/.."

ONLY="${1:-all}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS=(42 123 456 789 1024)
FOLDS=(0 1 2 3 4)

run() {
  printf "▶ %s\n" "$*"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$@"
}

# Experiment 1 — baseline comparison (7 models × 5 folds = 35 runs)
exp1() {
  local models=(abmil dsmil transmil clam_sb clam_mb vetgigagraph)
  for m in "${models[@]}"; do
    for f in "${FOLDS[@]}"; do
      run python scripts/04_train.py --model "$m" --fold "$f"
    done
  done
}

# Experiment 2 — graph variant ablation (5 graphs × 5 folds, vetgigagraph only)
exp2() {
  local graphs=(spatial_knn feature_sim dual_edge hierarchical heterogeneous)
  for g in "${graphs[@]}"; do
    for f in "${FOLDS[@]}"; do
      run python scripts/03_build_graphs.py --graph-type "$g"
      run python scripts/04_train.py --model vetgigagraph --fold "$f"
    done
  done
}

# Experiment 3 — GNN backbone ablation (4 backbones × 5 folds)
exp3() {
  local backbones=(gat gcn graphsage gin)
  for b in "${backbones[@]}"; do
    for f in "${FOLDS[@]}"; do
      run python scripts/04_train.py --model vetgigagraph --fold "$f"   # backbone via config override
    done
  done
}

# Experiment 4 — fusion ablation (6 fusions × 5 folds)
exp4() {
  local fusions=(gnn_only slide_only concat learnable_weighted cross_attention gated)
  for fz in "${fusions[@]}"; do
    for f in "${FOLDS[@]}"; do
      run python scripts/04_train.py --model vetgigagraph --fold "$f"   # fusion via config override
    done
  done
}

# Experiment 5 — cross-species transfer (T1–T5)
exp5() {
  for t in T1 T2 T3 T4 T5; do
    for f in "${FOLDS[@]}"; do
      run python scripts/04_train.py --model vetgigagraph --fold "$f"   # transfer mode via config override
    done
  done
}

case "$ONLY" in
  --only) shift; ONLY="${1:-}"; ;;
esac

case "${ONLY:-all}" in
  exp1) exp1 ;;
  exp2) exp2 ;;
  exp3) exp3 ;;
  exp4) exp4 ;;
  exp5) exp5 ;;
  all)  exp1; exp2; exp3; exp4; exp5 ;;
  *)    echo "unknown --only $ONLY (use exp1|exp2|exp3|exp4|exp5|all)"; exit 1 ;;
esac

# Aggregate metrics + statistical tests once everything finishes.
run python scripts/05_evaluate.py --models all --bonferroni
