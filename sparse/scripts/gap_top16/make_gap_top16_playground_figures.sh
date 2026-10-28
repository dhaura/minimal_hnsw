#!/bin/bash
# Playground comparison for the gap / top-16 full sweep, in a NEW playground folder
# (results/<dataset>_perlmutter_gap_top16); existing results are only read.
#
#   bash make_gap_top16_playground_figures.sh                         # msmarco_full
#   DATASET=nq_splade bash make_gap_top16_playground_figures.sh
#
# 1. Fronts from run_gap_top16_sweep_perlmutter.sh, each folded into one
#    best-QPS-per-recall front in its own data folder ($DATA_ROOT/sparse_hnsw_gt16_<series>):
#      all      every row of the sweep (no filter, gap, top-16, both), as SparseHNSW
#      gap / top16 / top16gap   SparseHNSW+Gap, SparseHNSW+Top16, SparseHNSW+Top16+Gap
# 2. Two comparisons, each with merged points, Pareto, speedup tables and figures
#    (recall >= 0.75):
#      vs_baselines   the "all" front vs the existing baselines (not re-run)
#      vs_sparsehnsw  the three variant fronts vs the existing playground SparseHNSW
#                     front ($DATA_ROOT/sparse_hnsw, not re-run)

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATASET=${DATASET:-msmarco_full}
SWEEP=${1:-$HERE/../results/gap_top16/full_sweep_$DATASET}
PG=/global/homes/d/dhaura/repos/SpKNN/spknn-playground
OUT=$PG/results/${DATASET}_perlmutter_gap_top16
DATA_ROOT=$SCRATCH/datasets/SpKNN/$DATASET

module load python/3.11-24.1.0 2>/dev/null || true
source "$PG/common/bench_env_perlmutter.sh" >/dev/null
source "$SPKNN_VENV/bin/activate"
mkdir -p "$OUT"

VARIANTS=()
for series in all gap top16 top16gap; do
    case $series in
        all) MODEL=SparseHNSW ;;           gap) MODEL=SparseHNSW+Gap ;;
        top16) MODEL=SparseHNSW+Top16 ;;   top16gap) MODEL=SparseHNSW+Top16+Gap ;;
    esac
    D=$DATA_ROOT/sparse_hnsw_gt16_$series
    mkdir -p "$D"
    rm -f "$D"/sparse_hnsw_results_gaptop16_*.csv "$D"/configs/sparse_hnsw_results_gaptop16_*.csv
    for f in "$SWEEP"/gaptop16_"$DATASET"_*.csv; do
        python3 - "$f" "$D/sparse_hnsw_results_$(basename "$f")" "$series" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
want = sys.argv[3]
def series(p):
    top16, gap = " layout=list16" in p, " gap=" in p
    return ("top16" if top16 else "") + ("gap" if gap else "") or "base"
keep = [r for r in rows if want == "all" or series(r["Params"]) == want]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(keep)
PY
    done
    python3 "$HERE/../combine_hnsw_envelope.py" "$D" --model "$MODEL" | tail -2
    if [ "$series" = all ]; then ALL="$D/sparse_hnsw_results.csv"; else VARIANTS+=("$D/sparse_hnsw_results.csv"); fi
done
BASELINES=()
for pat in "$DATA_ROOT"/sindi/sindi_results*.csv "$DATA_ROOT"/grassRMA/grassRMA_results*.csv \
           "$DATA_ROOT"/kannolo/kannolo_results*.csv "$DATA_ROOT"/seismic/seismic_results*.csv \
           "$DATA_ROOT"/pyanns/pyanns_results*.csv; do
    [ -s "$pat" ] && BASELINES+=("$pat")
done

# compare <name> <ours, comma list> <against, comma list> <csv>...
compare() {
    local name=$1 ours=$2 against=$3; shift 3
    printf "  $name input %s\n" "$@"
    python3 common/merge_results.py -o "$OUT/all_points_$name.csv" "$@"
    python3 common/merge_results.py -o "$OUT/pareto_$name.csv" --pareto "$@"
    for o in ${ours//,/ }; do
        python3 "$HERE/../quant_slots/make_speedup_table.py" "$OUT/pareto_$name.csv" \
            "$OUT/speedup_${name}_${o//+/_}.csv" --ours "$o" --baselines "$against" \
            --thresholds 0.80,0.85,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99
    done
    # Figures: recall >= 0.75 only, one marker per 1% recall bin (best QPS per bin).
    python3 - "$OUT/pareto_$name.csv" "$OUT/.plot_input.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
    w.writerows(r for r in rows if float(r["Recall"]) >= 0.75)
PY
    rm -rf "${OUT:?}/figures_$name"
    python3 common/plot_results.py -i "$OUT/.plot_input.csv" -o "$OUT/figures_$name" --xmin 0.75 --bin 1
    rm -f "${OUT:?}/.plot_input.csv"
}

cd "$PG"
compare vs_baselines SparseHNSW SEISMIC,kANNolo,PyANNS,GrassRMA,SINDI "$ALL" "${BASELINES[@]}"
compare vs_sparsehnsw SparseHNSW+Gap,SparseHNSW+Top16,SparseHNSW+Top16+Gap SparseHNSW \
    "$DATA_ROOT/sparse_hnsw/sparse_hnsw_results.csv" "${VARIANTS[@]}"
echo "-> $OUT"
