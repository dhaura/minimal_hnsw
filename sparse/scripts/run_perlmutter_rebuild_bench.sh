#!/bin/bash
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=06:00:00
#SBATCH --job-name=pm_rebuild_bench
#SBATCH --output=logs/pm_rebuild_bench_%j.out

set -u
REPO=$SCRATCH/repos/sparse_hnsw/minimal_hnsw
SCRIPTS=$REPO/sparse/scripts
D1M=$REPO/sparse/data/msmarco_1M
DFULL=$REPO/sparse/data/msmarco_full

M=16; EFC=200; EF=150
DOC_PRUNE=0.35; QUERY_PRUNE=0.5; N_CAND=20

export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:/opt/AMD/aocc-compiler-4.1.0/lib:${LD_LIBRARY_PATH:-}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export MKL_NUM_THREADS=1

numa_for() {   # numa_for <threads>
  command -v numactl >/dev/null || return 0
  if [ "$1" -le 64 ]; then echo "numactl --cpunodebind=0-3 --interleave=0-3"
  else echo "numactl --interleave=all"; fi
}

echo "host=$(hostname)"
echo "cpu=$(lscpu | sed -n 's/^Model name: *//p')"
echo "avx=$(lscpu | grep -o -E 'avx[0-9a-z_]*' | sort -u | tr '\n' ' ')"
echo "params: M=$M efC=$EFC ef=$EF | sindi doc=$DOC_PRUNE query=$QUERY_PRUNE ncand=$N_CAND"

##############################################################################
echo
echo "#################### PHASE 1: compiler A/B (msmarco_1M) ####################"
##############################################################################
AB_THREADS=64
export OMP_NUM_THREADS=$AB_THREADS
NUMA=$(numa_for $AB_THREADS)
echo "threads=$AB_THREADS launch='$NUMA'"
echo

AB_TSV=$(mktemp)

run_ab() {   # run_ab <build> <bin> <label>
  local build=$1 bin=$2 label=$3 out idx qt rc
  case $bin in
    sparse_hnsw_demo)
      out=$($NUMA "$REPO/$build/bin/$bin" $M $EFC $EF 1 0 0 1 0 \
            "$D1M/base_1M.csr" "$D1M/queries.dev.csr" "$D1M/base_1M.dev.gt" \
            /dev/null 2>&1) ;;
    grassRMA_demo)
      out=$($NUMA "$REPO/$build/bin/$bin" $M $EFC $EF \
            "$D1M/base_1M.csr" "$D1M/queries.dev.csr" "$D1M/base_1M.dev.gt" 2>&1) ;;
  esac
  idx=$(grep -oP 'points to the index in \K[0-9]+' <<<"$out" | tail -1)
  qt=$(grep -oP 'Total Query time: \K[0-9]+'      <<<"$out" | tail -1)
  rc=$(grep -oP 'Recall@k: \K[0-9.]+'             <<<"$out" | tail -1)
  printf '%-14s %-18s index_us=%-12s query_us=%-11s recall=%s\n' \
         "$label" "$bin" "${idx:-FAIL}" "${qt:-FAIL}" "${rc:-FAIL}"
  # only the first (A) gcc run feeds the decision; A' is the drift control
  echo -e "${build}\t${bin}\t${label}\t${qt:-0}" >> "$AB_TSV"
}

for bin in sparse_hnsw_demo grassRMA_demo; do
  run_ab build-gnu   "$bin" "gnu-14.3"
  run_ab build-intel "$bin" "icpx-2025.3"
  run_ab build-aocc  "$bin" "aocc-4.1"
  run_ab build-gnu   "$bin" "gnu-14.3(A2)"
  echo "---------------------------------------------------------------------"
done

WINNER=$(awk -F'\t' '
  $3 == "gnu-14.3(A2)" { next }            # drift control, not a candidate
  $4 > 0 { tot[$1] += $4 }
  END { best=""; for (b in tot) if (best=="" || tot[b] < bt) { best=b; bt=tot[b] }
        print best }' "$AB_TSV")
[ -n "$WINNER" ] || WINNER=build-gnu

echo
echo "-- drift control (the two gcc runs should agree; if not, treat the A/B as noise) --"
awk -F'\t' '$3 ~ /^gnu/ { printf "  %-18s %-14s query_us=%s\n", $2, $3, $4 }' "$AB_TSV"
echo
echo "WINNING TOOLCHAIN: $WINNER  (lowest summed query time over both binaries)"
BIN=$REPO/$WINNER/bin

##############################################################################
echo
echo "#################### PHASE 2: msmarco_full, all three ####################"
##############################################################################
FULL_THREADS=128
export OMP_NUM_THREADS=$FULL_THREADS
NUMA=$(numa_for $FULL_THREADS)
echo "build=$WINNER threads=$FULL_THREADS launch='$NUMA'"

BASE=$DFULL/base_full.csr
QUERIES=$DFULL/queries.dev.csr
GT=$DFULL/base_full.dev.gt
mkdir -p "$SCRIPTS/results"

echo; echo "########## sparse_hnsw ##########"
$NUMA stdbuf -oL -eL "$BIN/sparse_hnsw_demo" $M $EFC $EF 1 0 0 1 0 \
  "$BASE" "$QUERIES" "$GT" "$SCRIPTS/results/msmarco_full_sparse_hnsw_${SLURM_JOB_ID}.csv"

echo; echo "########## grassRMA ##########"
$NUMA stdbuf -oL -eL "$BIN/grassRMA_demo" $M $EFC $EF "$BASE" "$QUERIES" "$GT"

echo; echo "########## SINDI (vsag, znver3) ##########"
$NUMA stdbuf -oL -eL "$BIN/sindi_demo" \
  $DOC_PRUNE $QUERY_PRUNE $N_CAND "$BASE" "$QUERIES" "$GT"

rm -f "$AB_TSV"
echo; echo "==================== done ===================="
