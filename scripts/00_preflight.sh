#!/usr/bin/env bash
# Phase 10.1 — Pre-flight environment + data prep checks.
#
# Wraps the verification matrix in `docs/02-design/02-data-spec.md` §5.
# Exits non-zero on first failure with a pointer to the offending row.
#
# Usage:
#   bash scripts/00_preflight.sh
#
# Required passes:
#   1. Python ≥ 3.10
#   2. pip install editable (-e .) succeeds
#   3. requirements.txt installable
#   4. CUDA + GPU available
#   5. GigaPath tile encoder loads from HF Hub
#   6. ≥ 800 GB free disk on /data
#   7. CATCH dataset present (≥ 333 .svs files)
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf "\n[%s] %s\n" "$1" "$2"; }
ok() { printf "  ✅ %s\n" "$1"; }
fail() { printf "  ❌ %s\n" "$1"; exit 1; }

step "1/7" "Python ≥ 3.10"
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  && ok "$(python3 --version)" \
  || fail "Python 3.10+ required"

step "2/7" "Editable install (pip install -e .)"
python3 -c 'import src; print(src.__file__)' >/dev/null \
  && ok "src/ importable" \
  || fail "Run: pip install -e ."

step "3/7" "Required packages"
python3 - <<'PY' || fail "missing dependencies — run: pip install -r requirements.txt"
import importlib, sys
for mod in (
    "torch", "torch_geometric", "openslide", "timm",
    "pytorch_lightning", "wandb", "torchstain",
    "h5py", "pandas", "scipy", "sklearn", "matplotlib",
):
    importlib.import_module(mod)
print("ok")
PY
ok "all packages importable"

step "4/7" "CUDA + GPU"
python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' \
  && ok "GPU detected" \
  || fail "CUDA not available — train will fall back to CPU but will be very slow"

step "5/7" "GigaPath HF gate (skipped unless RUN_HEAVY=1)"
if [ "${RUN_HEAVY:-0}" = "1" ]; then
  python3 -c 'import timm; m = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, num_classes=0); n = sum(p.numel() for p in m.parameters()); print(f"params: {n:,}"); assert n > 1.0e9' \
    && ok "GigaPath weights load (~1.13B params)" \
    || fail "GigaPath load failed — check HF_TOKEN and HF gate-acceptance"
else
  ok "skipped (set RUN_HEAVY=1 to actually load GigaPath)"
fi

step "6/7" "≥ 800 GB free disk on /data"
free_gb=$(df -BG --output=avail /data 2>/dev/null | tail -1 | tr -d 'G ' || echo 0)
if [ "${free_gb:-0}" -ge 800 ]; then
  ok "${free_gb} GB available"
else
  fail "/data has < 800 GB (got ${free_gb} GB)"
fi

step "7/7" "CATCH dataset present (≥ 333 .svs)"
RAW="${VETGIGA_RAW_DIR:-/data/cancerImagingArchive}"
n_svs=$(find "$RAW" -maxdepth 3 -name '*.svs' 2>/dev/null | wc -l || echo 0)
if [ "$n_svs" -ge 333 ]; then
  ok "${n_svs} WSI files found"
else
  fail "Found ${n_svs} .svs files under $RAW/ (need ≥ 333)"
fi

printf "\n✅ All preflight checks passed.\n"
