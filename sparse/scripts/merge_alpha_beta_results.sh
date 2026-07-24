#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
OUT="${1:-$RESULTS_DIR/msmarco_full_alpha_beta_sweep.csv}"

shopt -s nullglob
files=("$RESULTS_DIR"/raw_a*_b*.csv)
if [ ${#files[@]} -eq 0 ]; then
  echo "No raw_a*_b*.csv files found in $RESULTS_DIR -- nothing to merge." >&2
  exit 1
fi

head -n 1 "${files[0]}" > "$OUT"
for f in "${files[@]}"; do
  tail -n +2 "$f" >> "$OUT"
done

echo "Merged ${#files[@]} file(s) into $OUT"
