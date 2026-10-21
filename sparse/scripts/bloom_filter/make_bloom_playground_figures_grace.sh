#!/bin/bash
# GRACE port of make_bloom_playground_figures.sh.  Comparison with the Bloom
# gate ON, in a NEW playground folder (results/<dataset>_grace_bloom);
# existing playground results are only ever READ.
#
#   bash make_bloom_playground_figures_grace.sh [sweep_dir]        # msmarco_full
#
# Differences from the Perlmutter script, all forced by the Grace data layout:
#  * paths: $PG and the bench_env are the Grace ones; no $SCRATCH.
#  * the no-gate SparseHNSW series is rebuilt from THIS sweep's own T=-1 rows
#    instead of $DATA_ROOT/sparse_hnsw/.  On Grace that directory is empty, and
#    the only other Grace SparseHNSW front (results/<ds>_grace, Aug, job
#    19271186) predates the slot-row layout and u8 quantization -- no "slots="
#    in its Params, one config, 48.7k QPS at 95.5% recall where this build does
#    ~176k.  Using it would credit the Bloom gate with a ~2.5x "speedup" over a
#    SparseHNSW that no longer exists.  Same-job T=-1 is the honest reference.
#  * the other five baselines (SEISMIC, kANNolo, PyANNS, GrassRMA, SINDI) are
#    read from results/<dataset>_grace/all_points.csv, which is already the
#    merged Grace run at 48 threads, K=10, 6980 queries.
#
# 1. Bloom series: every gate-on row of the sweep (all thresholds, patience,
#    ...) folded into one best-QPS-per-recall front "SparseHNSW+Bloom".
# 2. No-gate series: the sweep's T=-1 rows, same treatment, "SparseHNSW".
# 3. Figures (figures_bloom, recall >= 0.75) and the speedup table.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATASET=${DATASET:-msmarco_full}
SWEEP=${1:-$HERE/../results/bloom_filter/grace_bloom_sweep_$DATASET}
PG=/scratch/user/dhaura/repos/spknn-playground
OUT=$PG/results/${DATASET}_grace_bloom
DATA_ROOT=/scratch/user/dhaura/datasets/SpKNN/$DATASET
BLOOM_DATA=$DATA_ROOT/sparse_hnsw_bloom_grace
NOGATE_DATA=$DATA_ROOT/sparse_hnsw_nogate_grace
GRACE_BASE=$PG/results/${DATASET}_grace/all_points.csv

module purge
module load GCCcore/13.2.0 Python/3.11.5
source "$PG/common/bench_env.sh" >/dev/null
source "$SPKNN_VENV/bin/activate"

[ -d "$SWEEP" ] || { echo "FATAL no sweep dir: $SWEEP" >&2; exit 1; }
[ -r "$GRACE_BASE" ] || { echo "FATAL no Grace baselines: $GRACE_BASE" >&2; exit 1; }

# Derived scratch only (never existing results); *_grace suffix, guarded.
rm -rf "${BLOOM_DATA:?}" "${NOGATE_DATA:?}"
mkdir -p "$OUT" "$BLOOM_DATA" "$NOGATE_DATA"

# 1+2. split each sweep CSV into gate-on / gate-off halves
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

# 3. the five external baselines, straight out of the merged Grace run
python3 - "$GRACE_BASE" "$OUT/.baselines.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
keep = [r for r in rows if r["Model"] not in ("SparseHNSW", "SparseHNSW+Bloom")]
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(keep)
print("baselines kept:", sorted({r["Model"] for r in keep}))
PY

CSVS=("$BLOOM_DATA/sparse_hnsw_results.csv" "$NOGATE_DATA/sparse_hnsw_results.csv" "$OUT/.baselines.csv")
printf '  input %s\n' "${CSVS[@]}"

cd "$PG"
python3 common/merge_results.py -o "$OUT/all_points.csv" "${CSVS[@]}"
python3 common/merge_results.py -o "$OUT/pareto.csv" --pareto "${CSVS[@]}"
python3 "$HERE/../quant_slots/make_speedup_table.py" "$OUT/pareto.csv" "$OUT/speedup_vs_baselines.csv" \
    --ours "SparseHNSW+Bloom" --baselines SparseHNSW,SEISMIC,kANNolo,PyANNS,GrassRMA,SINDI \
    --thresholds 0.80,0.85,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99

# Figures: as on Perlmutter, the no-gate series is dropped and SparseHNSW+Bloom
# is shown as SparseHNSW; recall >= 0.75, one marker per 1% recall bin.
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
rm -f "${OUT:?}/.plot_input.csv" "${OUT:?}/.baselines.csv"
echo "########## figures -> $OUT/figures_bloom ##########"
