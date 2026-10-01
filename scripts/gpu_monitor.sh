#!/usr/bin/env bash
# gpu_monitor.sh — sample NVIDIA GPU stats at a fixed interval.
#
# Low-overhead time series of GPU utilization, memory, temperature, power,
# and running compute processes. Useful while long-running ML jobs
# (feature extraction, graph construction, training) are on the GPU and
# you want a one-line-per-sample log alongside the workload.
#
# Usage:
#   bash gpu_monitor.sh                       # 30 s interval, runs until Ctrl-C
#   bash gpu_monitor.sh -i 10                 # 10 s interval
#   bash gpu_monitor.sh --once                # single snapshot, then exit
#   bash gpu_monitor.sh -n 60 -i 60           # one hour at 60 s intervals
#   bash gpu_monitor.sh -l /data/cia_outputs/logs/gpu.log
#   bash gpu_monitor.sh -i 5 | grep -E 'util=100'    # streaming filter
#
# Output (one sample per line):
#   [2026-05-06T10:30:00+09:00] util=100% mem=10422/24576MiB(42%) \
#       free=13954MiB temp=71C power=325/370W \
#       procs=[3297794:python3(10426MiB),2426618:python3(412MiB)]
#
# Exit codes:
#   0  normal exit (count reached, or signal received cleanly)
#   2  nvidia-smi missing
#   3  GPU index not present
set -uo pipefail

INTERVAL=30
COUNT=0          # 0 = run forever
LOG_FILE=""
GPU_ID=0

usage() {
  # Print the leading comment block (lines starting with "# ") as help text.
  awk '
    NR == 1 { next }                  # skip shebang
    /^[^#]/ { exit }                  # stop at first non-comment line
    { sub(/^# ?/, ""); print }
  ' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--interval)  INTERVAL="$2"; shift 2 ;;
    -n|--count)     COUNT="$2";    shift 2 ;;
    --once)         COUNT=1;       shift   ;;
    -l|--log)       LOG_FILE="$2"; shift 2 ;;
    -g|--gpu)       GPU_ID="$2";   shift 2 ;;
    -h|--help)      usage 0 ;;
    *) printf 'unknown arg: %s\n' "$1" >&2; usage 1 ;;
  esac
done

command -v nvidia-smi >/dev/null 2>&1 || {
  printf 'nvidia-smi not found in PATH — is the NVIDIA driver installed?\n' >&2
  exit 2
}

# Verify the GPU index exists; otherwise nvidia-smi errors are noisy.
if ! nvidia-smi --id="$GPU_ID" --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
  printf 'GPU index %s not available on this host.\n' "$GPU_ID" >&2
  exit 3
fi

# Strip surrounding whitespace from a CSV field.
trim() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<<"$1"; }

sample_line() {
  local ts gpu_csv util mem_u mem_t mem_f temp pwr pwr_lim pct procs
  ts="$(date -Iseconds)"

  # Single nvidia-smi call for the gauge fields.
  gpu_csv="$(nvidia-smi --id="$GPU_ID" \
    --query-gpu=utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,power.draw,power.limit \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$gpu_csv" ]]; then
    printf '[%s] (nvidia-smi gauge query failed)\n' "$ts"
    return
  fi
  IFS=',' read -r util mem_u mem_t mem_f temp pwr pwr_lim <<<"$gpu_csv"
  util="$(trim "$util")"
  mem_u="$(trim "$mem_u")"
  mem_t="$(trim "$mem_t")"
  mem_f="$(trim "$mem_f")"
  temp="$(trim "$temp")"
  pwr="$(trim "$pwr")"
  pwr_lim="$(trim "$pwr_lim")"
  if [[ "$mem_t" =~ ^[0-9]+$ && "$mem_t" -gt 0 ]]; then
    pct=$(( mem_u * 100 / mem_t ))
  else
    pct="?"
  fi

  # Compute processes (separate query — graphics-only PIDs are excluded).
  procs="$(nvidia-smi --id="$GPU_ID" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' 'NF>=3 {
        gsub(/^[ \t]+|[ \t]+$/, "", $1)
        gsub(/^[ \t]+|[ \t]+$/, "", $2)
        gsub(/^[ \t]+|[ \t]+$/, "", $3)
        if (out != "") out = out ","
        out = out $1 ":" $2 "(" $3 "MiB)"
      } END { print out }')"
  [[ -z "$procs" ]] && procs="(none)"

  printf '[%s] util=%s%% mem=%s/%sMiB(%s%%) free=%sMiB temp=%sC power=%s/%sW procs=[%s]\n' \
    "$ts" "$util" "$mem_u" "$mem_t" "$pct" "$mem_f" "$temp" "$pwr" "$pwr_lim" "$procs"
}

emit() {
  local line
  line="$(sample_line)"
  printf '%s\n' "$line"
  if [[ -n "$LOG_FILE" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    printf '%s\n' "$line" >> "$LOG_FILE"
  fi
}

trap 'exit 0' INT TERM

hdr="# gpu_monitor.sh started=$(date -Iseconds) interval=${INTERVAL}s count=${COUNT:-inf} gpu=$GPU_ID log=${LOG_FILE:-stdout}"
printf '%s\n' "$hdr"
if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  printf '%s\n' "$hdr" >> "$LOG_FILE"
fi

i=0
while :; do
  emit
  i=$((i + 1))
  if [[ "$COUNT" -gt 0 && "$i" -ge "$COUNT" ]]; then
    break
  fi
  sleep "$INTERVAL"
done
