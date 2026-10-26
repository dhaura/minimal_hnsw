#!/bin/bash
# Rebuild the msmarco_full SparseHNSW series in spknn-playground from the
# slotted-layout results (slots only, best point per recall), then regenerate
# the merged CSVs, figures and speedup_vs_baselines.csv.
#
#   bash update_playground_msmarco.sh <archive_name>     e.g. archive_v8_pre_low_recall
#
# Inputs: results/quant_slots/step2_full_sweep (via prepare_playground_slots.py,
# peak memory from step3) and results/quant_slots/step5_low_recall (slots-only
# runs, so their PeakRSSGB is already the slotted footprint).

set -euo pipefail
ARCHIVE=${1:?archive name for the current playground results, e.g. archive_v8_pre_low_recall}
HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/../results/quant_slots"
PG=/global/homes/d/dhaura/repos/SpKNN/spknn-playground
OUT=$PG/results/msmarco_full_perlmutter
DATA=$SCRATCH/datasets/SpKNN/msmarco_full/sparse_hnsw

[ -e "$OUT/$ARCHIVE" ] && { echo "archive $OUT/$ARCHIVE exists; pick a new name" >&2; exit 1; }
mkdir -p "$OUT/$ARCHIVE" "$DATA/$ARCHIVE"
cp -a "$OUT/all_points.csv" "$OUT/pareto.csv" "$OUT/speedup_vs_baselines.csv" "$OUT/figures" "$OUT/figures_zoom" "$OUT/$ARCHIVE/"
cp -a "$DATA/sparse_hnsw_results.csv" "$DATA/$ARCHIVE/"

# Slotted sweep points (corrected peak memory) and low-recall points as inputs.
python3 "$HERE/prepare_playground_slots.py" "$DATA"
for f in "$RES"/step5_low_recall/low_recall_*.csv; do
    cp "$f" "$DATA/sparse_hnsw_results_slots_lowrecall_$(basename "$f" | sed 's/^low_recall_//')"
done
# Only slotted inputs: the old per-config CSVs stay in configs/ and are not read.
python3 "$HERE/../combine_hnsw_envelope.py" "$DATA" | tail -3

bash "$PG/common/make_figures_perlmutter.sh" | tail -3
python3 "$HERE/make_speedup_table.py" "$OUT/pareto.csv" "$OUT/speedup_vs_baselines.csv" \
    --thresholds 0.80,0.85,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99
