#!/usr/bin/env bash
# Phase 10.9 — Aggregate verifier — runs Step 2 → 6 in order.
#
# Stops on first failure. Used by the Phase 11 smoke test and the
# Phase 12 final-report sign-off.
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf "\n[%s] %s\n" "$1" "$2"; }

step "2/5" "Tile output (Module A)"
python3 scripts/_verify_step_2.py "$@"

step "3/5" "HDF5 features (Module B)"
python3 scripts/_verify_step_3.py "$@"

step "4/5" "Graphs (Module C)"
python3 scripts/_verify_step_4.py "$@"

step "5/5" "Training runs (Module E)"
python3 scripts/_verify_step_5.py "$@"

step "6/6" "Deliverables (figures + tables)"
python3 scripts/_verify_step_6.py "$@"

printf "\n✅ All verification steps passed.\n"
