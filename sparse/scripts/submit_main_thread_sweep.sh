#!/bin/bash
# Generates one job script per thread count from main_thread_sweep_template.sh
# (replacing the {{THREADS}} placeholders) and submits each with sbatch.
#
# Usage: ./submit_main_thread_sweep.sh [thread counts...]
#   e.g. ./submit_main_thread_sweep.sh 1 2 4 8
#   With no arguments, sweeps the default list below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/main_thread_sweep_template.sh"
TMP_DIR="$SCRIPT_DIR/tmp"

THREAD_COUNTS=("$@")
if [ ${#THREAD_COUNTS[@]} -eq 0 ]; then
  THREAD_COUNTS=(1 2 4 8 16 32 64 128)
fi

mkdir -p "$TMP_DIR" "$SCRIPT_DIR/logs"

for t in "${THREAD_COUNTS[@]}"; do
  cpus=$((2 * t)); [ "$cpus" -gt 256 ] && cpus=256
  job_script="$TMP_DIR/main_thread_sweep_t${t}.sh"
  sed -e "s/{{THREADS}}/${t}/g" -e "s/{{CPUS}}/${cpus}/g" "$TEMPLATE" > "$job_script"

  # Submit from SCRIPT_DIR so the relative logs/ path in #SBATCH --output works.
  job_id=$(cd "$SCRIPT_DIR" && sbatch --parsable "$job_script")
  echo "Submitted job $job_id with $t thread(s) ($job_script)"
done
