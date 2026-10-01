#!/usr/bin/env bash
# 1st-revision (applsci-4523976) GPU queue: new baselines, GIN diagnostics,
# GigaPath-DefGNN graph-construction ablation, GigaPath-DefGNN headline re-run. Resumable: a run whose metrics
# file exists is skipped. Every run uses the locked per-fold seeds.
#
# Usage: bash scripts/rev1_queue.sh [stage ...]   (default: all stages)
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${VETGIGA_OUTPUT_DIR:-/data/cia_outputs}"   # data root (see .env.example)

R=$OUT/checkpoints/rev1
SEEDS=(42 123 456 789 1024)
STAGES="${*:-acmil wikg gin defgnn_graph defgnn_rerun res20x}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

LOGDIR=$OUT/logs/rev1
# Headline deformable config with knn_chunk_size 1024 -> 16384: same k-NN
# neighbours (fp32 max diff 9.5e-7), ~12x faster training steps.
DEFORM_CONFIG=configs/rev1/experiment_deformable_fastknn.yaml
log() { echo "[rev1 $(date '+%m-%d %H:%M:%S')] $*"; }

train_v1() {  # name model fold [extra args...]
  local name=$1 model=$2 f=$3; shift 3
  local out="$R/$name/fold_$f"
  # fold_<f>.ckpt is written only after fit() returns, so it marks completion.
  # (04_train.py writes its config sidecar next to --metrics-out at start-up.)
  [[ -f "$out/fold_$f.ckpt" ]] && { log "skip $name fold $f"; return; }
  mkdir -p "$out"; log "start $name fold $f"
  python3 scripts/04_train.py --model "$model" --fold "$f" --seed "${SEEDS[$f]}" \
    --checkpoint-dir "$out" --metrics-out "$out/fold_metrics.json" "$@" > "$out/train.log" 2>&1
  log "done  $name fold $f (exit $?)"
}

train_deform() {  # name graphs_root fold
  local name=$1 graphs=$2 f=$3
  local out="$R/$name/fold_$f"
  [[ -f "$out/metrics.json" ]] && { log "skip $name fold $f"; return; }
  mkdir -p "$out"; log "start $name fold $f"
  python3 scripts/04b_train_deformable.py --config "$DEFORM_CONFIG" \
    --fold "$f" --graphs-root "$graphs" --num-workers 4 \
    --checkpoint-dir "$out/checkpoints" --metrics-out "$out/metrics.json" > "$out/train.log" 2>&1
  log "done  $name fold $f (exit $?)"
}

prep_20x() {  # 20x tiles -> features -> dual-edge graphs (Reviewer 2, point 3)
  # Encode each slide as soon as its tiling is complete (metadata.json is
  # written last), overlapping GPU feature extraction with CPU tiling.
  # 02_extract_features.py re-encodes whatever it is given, so pass only
  # slides that are tiled and not yet encoded.
  local ready rc watcher
  mkdir -p $OUT/features_20x $OUT/graphs_20x
  # Build graphs while features are still being extracted. --watch exits at
  # 350 graphs; if a slide is dropped at 20x it never gets there, so the
  # watcher is stopped explicitly after a final non-watch pass below.
  python3 scripts/03_build_graphs.py --graph-type dual_edge --features-root $OUT/features_20x \
    --graphs-root $OUT/graphs_20x --watch --poll-interval 120 >> "$LOGDIR/graphs_20x.log" 2>&1 &
  watcher=$!
  log "20x graph watcher started (pid $watcher)"
  while true; do
    ready=$(for m in $OUT/tiles_20x/*/metadata.json; do
               s=$(basename "$(dirname "$m")")
               [[ -f $OUT/features_20x/$s.h5 ]] || echo "$s"
             done)
    if [[ -n "$ready" ]]; then
      log "20x features: encoding $(echo "$ready" | wc -w) newly tiled slides"
      python3 scripts/02_extract_features.py --tiles-root $OUT/tiles_20x \
        --features-root $OUT/features_20x --slides $ready >> "$LOGDIR/features_20x.log" 2>&1
      rc=$?; [[ $rc -ne 0 ]] && log "20x features: extractor exit $rc"
    elif ! pgrep -f "01_preprocess[.]py --downsample 2" > /dev/null; then
      break
    else
      log "waiting for 20x tiling ($(ls $OUT/tiles_20x/*/metadata.json | wc -l)/350 tiled, $(ls $OUT/features_20x/*.h5 | wc -l) encoded)"
      sleep 600
    fi
  done
  log "20x: $(ls $OUT/tiles_20x/*/metadata.json | wc -l) tiled, $(ls $OUT/features_20x/*.h5 | wc -l) encoded"
  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
  # Final pass: every remaining feature file, no age grace (extraction is done).
  python3 scripts/03_build_graphs.py --graph-type dual_edge --features-root $OUT/features_20x \
    --graphs-root $OUT/graphs_20x --min-age-seconds 0 >> "$LOGDIR/graphs_20x.log" 2>&1
  rc=$?
  log "20x graphs: $(ls $OUT/graphs_20x/dual_edge/*.pt | wc -l) files (exit $rc)"
}

run_evals() {  # held-out evaluation + statistics on an otherwise idle GPU
  local models=$1 tag=$2
  log "start evals $tag"
  PYTORCH_ALLOC_CONF=expandable_segments:True python3 scripts/18_test_eval.py --models $models --skip-missing \
    >> "$LOGDIR/test_eval_$tag.log" 2>&1
  log "done  evals $tag (exit $?)"
  if [[ $tag == main ]]; then
    PYTORCH_ALLOC_CONF=expandable_segments:True python3 scripts/20_complexity.py >> "$LOGDIR/complexity_idle.log" 2>&1
    log "done  complexity re-measure (exit $?)"
  fi
  python3 scripts/19_test_stats.py > "$LOGDIR/test_stats_$tag.log" 2>&1
  log "done  stats $tag (exit $?)"
}

for stage in $STAGES; do
  case $stage in
    evals) run_evals "gat_spatial gat_feature gat_featknn defgnn_spatial defgnn_feature defgnn_featknn defgnn_dual gin_fix gin_fp32 gin_layernorm gin_mean" main; continue ;;
    evals20x) run_evals "gat_20x defgnn_20x" 20x; continue ;;
  esac
  [[ $stage == res20x ]] && prep_20x
  if [[ $stage == featknn ]]; then  # wait for scripts/25_extract_feature_knn_graphs.py
    until [[ $(ls $OUT/graphs/feature_knn/*.pt 2>/dev/null | wc -l) -ge 350 ]]; do
      log "waiting for feature k-NN graphs ($(ls $OUT/graphs/feature_knn/*.pt 2>/dev/null | wc -l)/350)"; sleep 300
    done
  fi
  for f in 0 1 2 3 4; do
    case $stage in
      acmil) train_v1 acmil acmil "$f" ;;
      wikg)  train_v1 wikg wikg "$f" ;;
      # Published GCN / GIN runs were aborted by the old NaNGuard at the first
      # fp16 GradScaler overflow (epochs 8-10 / 1); retrain with the fixed guard.
      gcn_fix) train_v1 gcn_fix vetgigagraph "$f" --config configs/rev1/gcn_fix.yaml ;;
      gin_fix) train_v1 gin_fix vetgigagraph "$f" --config configs/rev1/gin_fix.yaml ;;
      clam_inst)  # CLAM with the official instance-level clustering loss
        train_v1 clam_sb_inst clam_sb_inst "$f"
        train_v1 clam_mb_inst clam_mb_inst "$f" ;;
      gin)
        train_v1 gin_fp32      vetgigagraph "$f" --config configs/rev1/gin_fp32.yaml --mixed-precision false
        train_v1 gin_layernorm vetgigagraph "$f" --config configs/rev1/gin_layernorm.yaml
        train_v1 gin_mean      vetgigagraph "$f" --config configs/rev1/gin_mean.yaml ;;
      defgnn_graph)
        train_deform defgnn_spatial $OUT/graphs/spatial_knn "$f"
        train_deform defgnn_feature $OUT/graphs/feature_sim "$f" ;;
      res20x)  # headline GAT and GigaPath-DefGNN on 20x (0.5 um/px) tiles
        train_v1 gat_20x vetgigagraph "$f" --graphs-root $OUT/graphs_20x/dual_edge
        train_deform defgnn_20x $OUT/graphs_20x/dual_edge "$f" ;;
      featknn)  # true feature-only k-NN graph (k = 5), the feature branch of the dual-edge graph
        train_v1 gat_featknn vetgigagraph "$f" --graphs-root $OUT/graphs/feature_knn
        train_deform defgnn_featknn $OUT/graphs/feature_knn "$f" ;;
      defgnn_rerun)  # headline config again, now saving final weights (fold_<f>.ckpt)
        train_deform defgnn_dual $OUT/graphs/dual_edge "$f" ;;
      *) log "unknown stage $stage"; exit 1 ;;
    esac
  done
done
log "queue finished: $STAGES"
