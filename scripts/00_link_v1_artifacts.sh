#!/usr/bin/env bash
# Link v1-origin data artifacts (tiles / features / dual-edge graphs / locked
# patient-level CV splits) into the repo's data/ paths so the Hydra config
# paths resolve.
#
# Post-Phase-B (2026-05-31): the src/ tree is now unified, so the historical
# `src/shared → ../pw-vetGigagraph/src` symlink step has been removed. Only
# data symlinks remain — those still need to be re-created on every fresh
# checkout because the underlying files live on a 15 TB machine-local
# partition (/data/cia_outputs) and are not committed to git.
#
# Idempotent: re-running just re-creates the symlinks. Skips silently if
# data directories are absent (operator must run preprocessing first).

set -euo pipefail

V1_OUTPUTS="${V1_OUTPUTS:-${VETGIGA_OUTPUT_DIR:-/data/cia_outputs}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[link] data source: $V1_OUTPUTS"
echo "[link] repo:        $REPO_ROOT"

if [[ ! -d "$V1_OUTPUTS" ]]; then
  echo "[link] WARNING: data source ($V1_OUTPUTS) not found; run preprocessing first." >&2
fi

# data/ — symlink the underlying splits / tiles / features / graphs trees so
# the Hydra config paths resolve.
mkdir -p "$REPO_ROOT/data"
for sub in splits tiles features graphs; do
  src="$V1_OUTPUTS/$sub"
  dst="$REPO_ROOT/data/$sub"
  if [[ -d "$src" && ! -e "$dst" ]]; then
    echo "[link] data/$sub → $src"
    ln -s "$src" "$dst"
  elif [[ -L "$dst" ]]; then
    echo "[link] data/$sub already linked"
  fi
done

# Frozen v1 reference metrics are tracked in-repo (results/v1_reference/) so
# they survive any rebuild of the underlying preprocessing pipeline. No
# linking needed here; this comment is retained as documentation.
ref_dir="$REPO_ROOT/results/v1_reference"
if [[ ! -f "$ref_dir/v1_baseline_summary.json" && -d "$V1_OUTPUTS/logs" ]]; then
  echo "[link] hint: capture v1 reference metrics with scripts/_capture_v1_reference.py (TODO)."
fi

echo "[link] done."
