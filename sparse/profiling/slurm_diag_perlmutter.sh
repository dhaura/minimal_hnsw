#!/bin/bash
# High-recall diagnostic campaign, one dataset per job:
#   1. sparse_diag_sweep  -- decompose the recall-0.98 loss into traversal-miss
#                            vs shortlist-miss (decides graph work vs refine work)
#   2. hugepages A/B      -- (RUN_HP=1 only) same sparse_profile config from the
#                            normal and craype-hugepages2M builds, bracketed A/B/A
#
# Submit from this directory:
#   DATASET=msmarco_full RUN_HP=1 sbatch -J diag_ms slurm_diag_perlmutter.sh
#   DATASET=nq_splade              sbatch -J diag_nq slurm_diag_perlmutter.sh
#
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/diagseed_%x_%j.out

set -uo pipefail
REPO=$SCRATCH/repos/sparse_hnsw/minimal_hnsw
BIN=$REPO/build/bin
BIN_HP=$REPO/build-hp/bin
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:/opt/AMD/aocc-compiler-4.1.0/lib:${LD_LIBRARY_PATH:-}

DATASET=${DATASET:-msmarco_full}
RUN_HP=${RUN_HP:-0}
if [ "$DATASET" = "msmarco_full" ]; then
    DATA=$REPO/sparse/data/msmarco_full
    BASE=$DATA/base_full.csr; QUERIES=$DATA/queries.dev.csr; GT=$DATA/base_full.dev.gt
    ALPHA=0.85; BETA=8       # the frontier's best at recall 0.98
else
    DATA=$REPO/sparse/data/nq_splade
    BASE=$DATA/base_nq.csr; QUERIES=$DATA/queries.test.csr; GT=$DATA/base_nq.test.gt
    ALPHA=0.8; BETA=8
fi
M=32; EFC=200; THREADS=64
export OMP_NUM_THREADS=$THREADS OMP_PLACES=cores OMP_PROC_BIND=close
LAUNCH="numactl --cpunodebind=0-3 --interleave=0-3"
OUT=$REPO/sparse/profiling/logs
mkdir -p "$OUT"

echo "### dataset=$DATASET node=$(hostname) alpha=$ALPHA beta=$BETA M=$M efC=$EFC"

echo; echo "########## 1. diag: traversal-miss vs shortlist-miss at high recall ##########"
$LAUNCH stdbuf -oL -eL $BIN/sparse_diag_sweep \
    $M $EFC "200,400,800,1600,3200" 1 0 0 $ALPHA $BETA \
    "$BASE" "$QUERIES" "$GT" \
    "$OUT/diag_${DATASET}_${SLURM_JOB_ID}.csv" \
    "$OUT/diag_${DATASET}_${SLURM_JOB_ID}_decomp.csv" \
    "SparseHNSW_q8" 1 1 2000 1

if [ "$RUN_HP" = "1" ]; then
    echo; echo "########## 2. hugepages A/B (bracketed A, B, A) ##########"
    module load craype-hugepages2M 2>/dev/null
    export HUGETLB_MORECORE=yes HUGETLB_VERBOSE=1
    HP_ARGS="$M $EFC 800 1 0 0 $ALPHA 3 $BASE $QUERIES $GT batch@$THREADS,batch@1 0 cold 1"
    export PROF_BUILD_THREADS=$THREADS
    echo "-------- A: normal build --------"
    $LAUNCH stdbuf -oL -eL $BIN/sparse_profile $HP_ARGS
    echo "-------- B: hugepages build --------"
    $LAUNCH stdbuf -oL -eL $BIN_HP/sparse_profile $HP_ARGS
    echo "-------- A2: normal build (drift check) --------"
    $LAUNCH stdbuf -oL -eL $BIN/sparse_profile $HP_ARGS
fi

echo; echo "########## Done ##########"
