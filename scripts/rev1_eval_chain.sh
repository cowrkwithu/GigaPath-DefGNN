#!/usr/bin/env bash
# Held-out evaluation chain for the 1st revision (resumable: 18_test_eval.py
# skips keys already present in its --out file).
#   1. GPU: graph-ablation models (can share the GPU with 20x feature extraction)
#   2. CPU fp32: GIN variants (too large to share the GPU); serialized after 1
#      because both hold a ~17 GB graph cache in RAM
#   3. merge the CPU predictions into predictions.json, then 19_test_stats.py
set -u
cd "$(dirname "$0")/.."
OUT="${VETGIGA_OUTPUT_DIR:-/data/cia_outputs}"   # data root (see .env.example)
LOGDIR=$OUT/logs/rev1
log() { echo "[eval-chain $(date '+%m-%d %H:%M:%S')] $*"; }

log "start GPU evals"
PYTORCH_ALLOC_CONF=expandable_segments:True python3 scripts/18_test_eval.py \
  --models gat_spatial defgnn_spatial gat_feature defgnn_feature gat_featknn defgnn_featknn defgnn_dual \
  --skip-missing >> "$LOGDIR/test_eval_main_retry.log" 2>&1
log "done  GPU evals (exit $?)"

log "start GIN CPU evals"
nice -n 5 python3 scripts/18_test_eval.py --device cpu --models gin_fix gin_fp32 gin_layernorm gin_mean \
  --skip-missing --out results/test_eval/predictions_gin_cpu.json >> "$LOGDIR/test_eval_gin_cpu.log" 2>&1
log "done  GIN CPU evals (exit $?)"

python3 - <<'EOF'
import json, os
from pathlib import Path
main, gin = Path("results/test_eval/predictions.json"), Path("results/test_eval/predictions_gin_cpu.json")
res = json.loads(main.read_text())
add = json.loads(gin.read_text())
res.update(add)  # the CPU fp32 GIN results are authoritative
tmp = main.with_suffix(".json.tmp")
tmp.write_text(json.dumps(res))
os.replace(tmp, main)
print(f"merged {len(add)} GIN entries -> {len(res)} total")
EOF
log "merge exit $?"

python3 scripts/19_test_stats.py > "$LOGDIR/test_stats_main.log" 2>&1
log "done  stats main (exit $?)"
