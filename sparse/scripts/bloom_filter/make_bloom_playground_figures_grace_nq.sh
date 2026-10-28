#!/bin/bash
# Grace nq_splade playground comparison with the Bloom gate ON, in a NEW
# folder (results/nq_splade_grace_bloom); existing results are only READ.
#
#   bash make_bloom_playground_figures_grace_nq.sh [sweep_dir]
#
# Differs from the msmarco_full Grace script only in where the baselines come
# from.  msmarco_full had a pre-merged results/<ds>_grace/all_points.csv; for
# nq_splade the five baselines were run fresh into the per-dataset layout
# $SPKNN_OUT_ROOT/<method>/<method>_results_*.csv, which is the layout the
# original Perlmutter script expects, so they are globbed per method.
#
# As on msmarco_full, the no-gate SparseHNSW series is rebuilt from THIS
# sweep's own T=-1 rows (same build, same jobs) rather than any older run.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATASET=${DATASET:-nq_splade}
SWEEP=${1:-$HERE/../results/bloom_filter/threshold_sweeps/grace_$DATASET}
PG=/scratch/user/dhaura/repos/spknn-playground
OUT=$PG/results/${DATASET}_grace_bloom
DATA_ROOT=/scratch/user/dhaura/datasets/SpKNN/$DATASET
BLOOM_DATA=$DATA_ROOT/sparse_hnsw_bloom_grace
NOGATE_DATA=$DATA_ROOT/sparse_hnsw_nogate_grace

module purge
module load GCCcore/13.2.0 Python/3.11.5
source "$PG/common/bench_env.sh" >/dev/null
source "$SPKNN_VENV/bin/activate"

[ -d "$SWEEP" ] || { echo "FATAL no sweep dir: $SWEEP" >&2; exit 1; }
rm -rf "${BLOOM_DATA:?}" "${NOGATE_DATA:?}"
mkdir -p "$OUT" "$BLOOM_DATA" "$NOGATE_DATA"

for f in "$SWEEP"/bloom_"$DATASET"_*.csv; do
    [ -s "$f" ] || continue
    python3 - "$f" "$BLOOM_DATA/sparse_hnsw_results_$(basename "$f")" \
                   "$NOGATE_DATA/sparse_hnsw_results_$(basename "$f")" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for path, keep in ((sys.argv[2], [r for r in rows if " T=" in r["Params"]]),
                   (sys.argv[3], [r for r in rows if " T=" not in r["Params"]])):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(keep)
PY
done
python3 "$HERE/../combine_hnsw_envelope.py" "$BLOOM_DATA"  --model "SparseHNSW+Bloom" | tail -2
python3 "$HERE/../combine_hnsw_envelope.py" "$NOGATE_DATA" --model "SparseHNSW"       | tail -2

CSVS=("$BLOOM_DATA/sparse_hnsw_results.csv" "$NOGATE_DATA/sparse_hnsw_results.csv")
for pat in "$DATA_ROOT"/sindi/sindi_results*.csv "$DATA_ROOT"/grassRMA/grassRMA_results*.csv \
           "$DATA_ROOT"/kannolo/kannolo_results*.csv "$DATA_ROOT"/seismic/seismic_results*.csv \
           "$DATA_ROOT"/pyanns/pyanns_results*.csv; do
    [ -s "$pat" ] && CSVS+=("$pat")
done
printf '  input %s\n' "${CSVS[@]}"
[ "${#CSVS[@]}" -ge 3 ] || { echo "FATAL only ${#CSVS[@]} inputs; baselines missing under $DATA_ROOT" >&2; exit 1; }

cd "$PG"
python3 common/merge_results.py -o "$OUT/all_points.csv" "${CSVS[@]}"
python3 common/merge_results.py -o "$OUT/pareto.csv" --pareto "${CSVS[@]}"
python3 "$HERE/../quant_slots/make_speedup_table.py" "$OUT/pareto.csv" "$OUT/speedup_vs_baselines.csv" \
    --ours "SparseHNSW+Bloom" --baselines SparseHNSW,SEISMIC,kANNolo,PyANNS,GrassRMA,SINDI \
    --thresholds 0.80,0.85,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99

python3 - "$OUT/pareto.csv" "$OUT/pareto_bloom_as_sparsehnsw.csv" "$OUT/.plot_input.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
keep = [dict(r, Model="SparseHNSW") if r["Model"] == "SparseHNSW+Bloom" else r
        for r in rows if r["Model"] != "SparseHNSW"]
for path, out in ((sys.argv[2], keep), (sys.argv[3], [r for r in keep if float(r["Recall"]) >= 0.75])):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(out)
PY
rm -rf "${OUT:?}/figures_bloom"
python3 common/plot_results.py -i "$OUT/.plot_input.csv" -o "$OUT/figures_bloom" --xmin 0.75 --bin 1
rm -f "${OUT:?}/.plot_input.csv"
echo "########## figures -> $OUT/figures_bloom ##########"
