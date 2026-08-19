#!/bin/bash
#SBATCH --job-name=ab_figures
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/scripts"

module load python/3.11-24.1.0 2>/dev/null

source /global/homes/d/dhaura/repos/SpKNN/spknn-playground/common/bench_env_perlmutter.sh
source "$SPKNN_VENV/bin/activate"

AB="$SPKNN_OUT_ROOT/alpha_beta"
OUT="$SPKNN_PLAYGROUND/results/${SPKNN_DATASET}_perlmutter/alpha_beta"
mkdir -p "$OUT"

shopt -s nullglob
CSVS=("$AB"/${CSV_GLOB:-alpha_beta_sweep*.csv})
if [ ${#CSVS[@]} -eq 0 ]; then
    echo "FATAL: no ${CSV_GLOB:-alpha_beta_sweep*.csv} under $AB" >&2
    exit 1
fi
echo "inputs:"; for f in "${CSVS[@]}"; do echo "  $f ($(($(wc -l < "$f") - 1)) rows)"; done

python3 - "${CSVS[@]}" <<'PY' || exit 1
import csv, os, re, sys, collections
gens, seen = collections.defaultdict(list), collections.Counter()
for p in sys.argv[1:]:
    m = re.search(r"_v(\d+)\.csv$", os.path.basename(p))
    gens["v" + m.group(1) if m else "base"].append(os.path.basename(p))
    for r in csv.DictReader(open(p)):
        d = dict(re.findall(r"(\w+)=([0-9.]+)", r["Params"]))
        seen[(d.get('alpha'), d.get('beta'), d.get('ef'))] += 1
if len(gens) > 1:
    print("FATAL: the selected CSVs span more than one sweep generation; "
          "every (alpha,beta,ef) would be plotted once per code version.", file=sys.stderr)
    for g, files in sorted(gens.items()):
        print(f"  {g}: {', '.join(files)}", file=sys.stderr)
    print("Narrow CSV_GLOB to one generation.", file=sys.stderr)
    sys.exit(1)
dupes = sum(1 for n in seen.values() if n > 1)
gen = next(iter(gens))
print(f"  ok: generation={gen}, {len(seen)} unique (alpha,beta,ef) points"
      + (f", {dupes} measured twice (overlapping grid seam -- expected)" if dupes else ""))
PY

echo
echo "=================== iso-recall analysis ==================="
python3 analyze_alpha_beta.py "${CSVS[@]}" \
    --targets "${TARGETS:-0.90,0.95,0.98,0.99}" \
    --baseline "${BASELINE:-0.8/3}" | tee "$OUT/analysis.txt"

echo
echo "=================== figures ==================="
python3 plot_alpha_beta_perlmutter.py "${CSVS[@]}" \
    -o "$OUT" \
    --candidates "${CANDIDATES:-auto}" \
    --baseline "${BASELINE:-0.8/3}" \
    --label "${SPKNN_DATASET} (${SPKNN_NDOCS} docs, ${BENCH_THREADS} threads, M=${M:-32} efC=${EFC:-200})"

echo
echo "wrote $OUT"
ls -la "$OUT"
