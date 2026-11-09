#!/bin/bash
# Figures and speedup tables for the final method sweep (run_final_sweep_perlmutter.sh): SparseHNSW
# against the baselines (CSVs in results/baselines/) and its features against plain SparseHNSW.
# Inputs are only read; each run replaces its own output folder.
#
#   bash make_final_figures.sh                      # msmarco_full -> results/final_sweep/figures_msmarco_full
#   DATASET=nq_splade bash make_final_figures.sh
#   # CSVs from another sweep with the same columns:
#   SWEEP=<csv dir> GLOB='<pattern>.csv' OUT=<dir> bash make_final_figures.sh
#
# 1. Each SparseHNSW series folded into one best-QPS-per-recall front ($OUT/fronts/<series>), from
#    seeded and unseeded rows (the lowrec preset), without patience stops, gate layouts off / inline16:
#      all       every such row, as SparseHNSW (the method's best front)
#      off / gap / inline16 / inline16gap   SparseHNSW-off, SparseHNSW+Gap, +Inline16, +Inline16+Gap
# 2. Two comparisons, each with merged points, Pareto, speedup tables and figures (recall >= 0.75):
#      vs_baselines  the "all" front vs every baseline CSV in results/baselines/<dataset>/<method>/
#      vs_off        the three feature fronts vs off (no gate, no stop) from the same jobs
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$(cd "$HERE/.." && pwd)"   # sparse/scripts
DATASET=${DATASET:-msmarco_full}
SWEEP=${SWEEP:-$S/results/final_sweep/full_sweep_$DATASET}
GLOB=${GLOB:-final_${DATASET}_*.csv}
OUT=${OUT:-$S/results/final_sweep/figures_$DATASET}
BASE_ROOT=${BASE_ROOT:-$S/results/baselines/$DATASET}

module load python/3.11-24.1.0 2>/dev/null || true
VENV=${SPKNN_VENV:-$SCRATCH/benchmarks/minimal_hnsw/bench-venv-perlmutter}
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"   # else the module's python (has matplotlib)
compgen -G "$SWEEP/$GLOB" >/dev/null || { echo "no sweep CSVs: $SWEEP/$GLOB" >&2; exit 1; }
rm -rf "${OUT:?}"
mkdir -p "$OUT"

declare -A MODEL=([all]=SparseHNSW [off]=SparseHNSW-off [gap]=SparseHNSW+Gap
                  [inline16]=SparseHNSW+Inline16 [inline16gap]=SparseHNSW+Inline16+Gap)
VARIANTS=()
for series in all off gap inline16 inline16gap; do
    D=$OUT/fronts/$series
    mkdir -p "$D"
    for f in "$SWEEP"/$GLOB; do
        python3 - "$f" "$D/sparse_hnsw_results_$(basename "$f")" "$series" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
want = sys.argv[3]
def series(p):
    kv = dict(t.split("=", 1) for t in p.split())
    lay = kv.get("layout", "")
    if "pat" in kv or lay not in ("", "inline16"):
        return None
    return (lay + ("gap" if "gap" in kv else "")) or "off"
keep = [r for r in rows if series(r["Params"]) and (want == "all" or series(r["Params"]) == want)]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(keep)
PY
    done
    python3 "$S/combine_hnsw_envelope.py" "$D" --model "${MODEL[$series]}" | tail -2
    case $series in
        all) ALL="$D/sparse_hnsw_results.csv" ;;
        off) OFF="$D/sparse_hnsw_results.csv" ;;
        *) VARIANTS+=("$D/sparse_hnsw_results.csv") ;;
    esac
done
BASELINES=()
for m in sindi grassRMA kannolo seismic pyanns; do
    for f in "$BASE_ROOT"/$m/*.csv; do [ -s "$f" ] && BASELINES+=("$f"); done
done
echo "baselines: ${#BASELINES[@]} CSVs from $BASE_ROOT"

# compare <name> <ours, comma list> <against, comma list> <csv>...
compare() {
    local name=$1 ours=$2 against=$3; shift 3
    python3 "$S/plotting/merge_results.py" -o "$OUT/all_points_$name.csv" "$@"
    python3 "$S/plotting/merge_results.py" -o "$OUT/pareto_$name.csv" --pareto "$@"
    for o in ${ours//,/ }; do
        python3 "$S/quant_slots/make_speedup_table.py" "$OUT/pareto_$name.csv" \
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
    python3 "$S/plotting/plot_results.py" -i "$OUT/.plot_input.csv" -o "$OUT/figures_$name" --xmin 0.75 --bin 1
    rm -f "${OUT:?}/.plot_input.csv"
}

compare vs_baselines SparseHNSW SEISMIC,kANNolo,PyANNS,GrassRMA,SINDI "$ALL" "${BASELINES[@]}"
compare vs_off SparseHNSW+Gap,SparseHNSW+Inline16,SparseHNSW+Inline16+Gap SparseHNSW-off "$OFF" "${VARIANTS[@]}"
echo "-> $OUT"
