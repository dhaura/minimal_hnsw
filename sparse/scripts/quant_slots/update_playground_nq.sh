#!/bin/bash
# Rebuild the nq_splade SparseHNSW series in spknn-playground from the
# slotted-layout full sweep (slots only, 448 B; best point per recall), then
# regenerate the merged CSVs, figures and speedup_vs_baselines.csv.
#
#   bash update_playground_nq.sh <archive_name>     e.g. archive_v5_pre_slots
#
# Input: results/quant_slots/step6_full_sweep_nq_splade (slots-only runs, so
# PeakRSSGB is already the slotted footprint). Only these new points form the
# SparseHNSW front; the old per-config CSVs stay in configs/ and are not read.

set -euo pipefail
ARCHIVE=${1:?archive name for the current playground results, e.g. archive_v5_pre_slots}
HERE="$(cd "$(dirname "$0")" && pwd)"
SWEEP="$HERE/../results/quant_slots/step6_full_sweep_nq_splade"
PG=/global/homes/d/dhaura/repos/SpKNN/spknn-playground
OUT=$PG/results/nq_splade_perlmutter
DATA=$SCRATCH/datasets/SpKNN/nq_splade/sparse_hnsw

[ -e "$OUT/$ARCHIVE" ] && { echo "archive $OUT/$ARCHIVE exists; pick a new name" >&2; exit 1; }
mkdir -p "$OUT/$ARCHIVE" "$DATA/$ARCHIVE"
cp -a "$OUT/all_points.csv" "$OUT/pareto.csv" "$OUT/speedup_vs_baselines.csv" "$OUT/figures" "$OUT/figures_zoom" "$OUT/$ARCHIVE/"
cp -a "$DATA/sparse_hnsw_results.csv" "$DATA/$ARCHIVE/"

for f in "$SWEEP"/slots_nq_splade_*.csv; do
    cp "$f" "$DATA/sparse_hnsw_results_$(basename "$f")"
done
python3 "$HERE/../combine_hnsw_envelope.py" "$DATA" | tail -3

SPKNN_DATASET=nq_splade bash "$PG/common/make_figures_perlmutter.sh" | tail -3
python3 "$HERE/make_speedup_table.py" "$OUT/pareto.csv" "$OUT/speedup_vs_baselines.csv" \
    --baselines SEISMIC,SINDI,kANNolo,PyANNS,GrassRMA \
    --thresholds 0.80,0.85,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99
