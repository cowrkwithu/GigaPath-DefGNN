#!/usr/bin/env bash
# Phase 11.5 — End-to-end smoke orchestrator.
#
# Generates a tiny synthetic graph fixture in /tmp, runs ABMIL +
# VetGigaGraph training for 1 epoch each, and verifies the resulting
# split CSV. Should complete in < 5 min on CPU.
#
# Usage:
#   bash scripts/smoke_test.sh                # default tmp dir
#   bash scripts/smoke_test.sh --out /path    # custom output dir
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${1:-${SMOKE_OUT:-/tmp/vetgigagraph_smoke_$$}}"
CONFIG="tests/fixtures/configs/smoke.yaml"

if [[ "${1:-}" == "--out" ]]; then
  OUT_DIR="${2:?--out requires a value}"
fi

DATA_DIR="$OUT_DIR/data"
RUN_DIR="$OUT_DIR/runs"
mkdir -p "$RUN_DIR"

echo "▶ smoke fixture root: $OUT_DIR"
echo

echo "[1/4] Generate synthetic graph dataset"
python3 tests/fixtures/_make_smoke.py --out "$DATA_DIR"

SPLITS_CSV="$DATA_DIR/splits/cv5fold.csv"
GRAPHS_ROOT="$DATA_DIR/graphs/spatial_knn"

echo
echo "[2/4] Smoke split sanity (column schema only — full verify_splits unit-tested separately)"
python3 -c "
import pandas as pd
df = pd.read_csv('$SPLITS_CSV')
required = {'slide_id', 'patient_id', 'tumor_class', 'fold', 'split'}
missing = required - set(df.columns)
assert not missing, f'split CSV missing columns: {missing}'
assert sorted(df.fold.unique().tolist()) == [0, 1, 2, 3, 4], 'expected 5 folds'
assert set(df.split.unique()).issubset({'train', 'val', 'test'}), 'unknown split values'
print(f'OK: {len(df)} rows, 5 folds, schema valid')
"

echo
echo "[3/4] Train ABMIL (1 epoch, CPU, fold 0)"
python3 scripts/04_train.py \
  --config "$CONFIG" \
  --model abmil --fold 0 \
  --splits-csv "$SPLITS_CSV" \
  --graphs-root "$GRAPHS_ROOT" \
  --max-epochs 1 --mixed-precision false --gpus cpu \
  --out "$RUN_DIR" \
  --metrics-out "$RUN_DIR/abmil_fold0.json" \
  --checkpoint-dir "$RUN_DIR/ckpt_abmil" \
  --log-level WARNING

echo
echo "[4/4] Train VetGigaGraph (1 epoch, CPU, fold 0)"
python3 scripts/04_train.py \
  --config "$CONFIG" \
  --model vetgigagraph --fold 0 \
  --splits-csv "$SPLITS_CSV" \
  --graphs-root "$GRAPHS_ROOT" \
  --max-epochs 1 --mixed-precision false --gpus cpu \
  --out "$RUN_DIR" \
  --metrics-out "$RUN_DIR/vetgigagraph_fold0.json" \
  --checkpoint-dir "$RUN_DIR/ckpt_vgg" \
  --log-level WARNING

echo
echo "✅ Smoke pipeline passed."
echo "   metrics: $RUN_DIR/{abmil,vetgigagraph}_fold0.json"
echo "   ckpts:   $RUN_DIR/ckpt_{abmil,vgg}/fold_0.ckpt"
