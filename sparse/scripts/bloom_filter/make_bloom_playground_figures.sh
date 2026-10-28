#!/bin/bash
# Comparison with the Bloom gate ON, in a NEW playground folder
# (results/<dataset>_perlmutter_bloom); existing results are only read.
#
#   bash make_bloom_playground_figures.sh <sweep_dir>                  # msmarco_full
#   DATASET=nq_splade bash make_bloom_playground_figures.sh <sweep_dir>
#   OUT_NAME / BLOOM_DATA_NAME pick a different playground folder / data folder
#   (defaults <dataset>_perlmutter_bloom / sparse_hnsw_bloom).
#
# 1. Bloom series: every gate-on, every-row point of the sweep (all thresholds,
#    patience, ...; gate-off and short-row-bypass rows are left out) folded into
#    one best-QPS-per-recall front named SparseHNSW+Bloom,
#    built in its own data folder ($DATA_ROOT/sparse_hnsw_bloom).
# 2. Merge with the existing baselines and the current SparseHNSW (no gate)
#    front, unchanged and not re-run.
# 3. Figures (figures_bloom, recall >= 0.75) and the speedup table at each recall.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SWEEP=${1:?usage: make_bloom_playground_figures.sh <sweep_dir> (e.g. ../results/bloom_filter/threshold_sweeps/perlmutter_msmarco_full)}
PG=/global/homes/d/dhaura/repos/SpKNN/spknn-playground
DATASET=${DATASET:-msmarco_full}
OUT=$PG/results/${OUT_NAME:-${DATASET}_perlmutter_bloom}
DATA_ROOT=$SCRATCH/datasets/SpKNN/$DATASET
BLOOM_DATA=$DATA_ROOT/${BLOOM_DATA_NAME:-sparse_hnsw_bloom}

module load python/3.11-24.1.0 2>/dev/null || true
source "$PG/common/bench_env_perlmutter.sh" >/dev/null
source "$SPKNN_VENV/bin/activate"

mkdir -p "$OUT" "$BLOOM_DATA"
rm -f "$BLOOM_DATA"/sparse_hnsw_results_bloom_*.csv
for f in "$SWEEP"/bloom_"$DATASET"_*.csv; do
    python3 - "$f" "$BLOOM_DATA/sparse_hnsw_results_$(basename "$f")" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
keep = [r for r in rows if " T=" in r["Params"] and "bypass=" not in r["Params"]]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(keep)
PY
done
python3 "$HERE/../combine_hnsw_envelope.py" "$BLOOM_DATA" --model "SparseHNSW+Bloom" | tail -3

CSVS=("$BLOOM_DATA/sparse_hnsw_results.csv" "$DATA_ROOT/sparse_hnsw/sparse_hnsw_results.csv")
for pat in "$DATA_ROOT"/sindi/sindi_results*.csv "$DATA_ROOT"/grassRMA/grassRMA_results*.csv \
           "$DATA_ROOT"/kannolo/kannolo_results*.csv "$DATA_ROOT"/seismic/seismic_results*.csv \
           "$DATA_ROOT"/pyanns/pyanns_results*.csv; do
    [ -s "$pat" ] && CSVS+=("$pat")
done
printf '  input %s\n' "${CSVS[@]}"

cd "$PG"
python3 common/merge_results.py -o "$OUT/all_points.csv" "${CSVS[@]}"
python3 common/merge_results.py -o "$OUT/pareto.csv" --pareto "${CSVS[@]}"
python3 "$HERE/../quant_slots/make_speedup_table.py" "$OUT/pareto.csv" "$OUT/speedup_vs_baselines.csv" \
    --ours "SparseHNSW+Bloom" --baselines SparseHNSW,SEISMIC,kANNolo,PyANNS,GrassRMA,SINDI \
    --thresholds 0.80,0.85,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99

# Figures: the no-gate series is dropped and SparseHNSW+Bloom is shown as
# SparseHNSW; only recall >= 0.75 is plotted, one marker per 1% recall bin
# (best QPS per bin kept, so no threshold value is lost).
python3 - "$OUT/pareto.csv" "$OUT/pareto_bloom_as_sparsehnsw.csv" "$OUT/.plot_input.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
keep = [dict(r, Model="SparseHNSW") if r["Model"] == "SparseHNSW+Bloom" else r
        for r in rows if r["Model"] != "SparseHNSW"]
for path, out in ((sys.argv[2], keep), (sys.argv[3], [r for r in keep if float(r["Recall"]) >= 0.75])):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(out)
PY
rm -rf "${OUT:?}/figures_bloom"
python3 common/plot_results.py -i "$OUT/.plot_input.csv" -o "$OUT/figures_bloom" --xmin 0.75 --bin 1
rm -f "${OUT:?}/.plot_input.csv"
