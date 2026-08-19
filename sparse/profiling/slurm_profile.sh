#!/bin/bash
# Run the whole profiling ladder on the Grace cluster (TAMU HPRC).
#
# Grace compute node: 2 x Intel Xeon Gold 6248R (Cascade Lake), 24 cores each,
# SMT off -> 48 cores, 2 NUMA domains, 32 KB L1d / 1 MB L2 per core, 36 MB L3
# per socket. This is NOT the Perlmutter AMD Zen3 node the previous revision of
# this toolkit targeted; see archive/v1/ for that version.
#
# Submit from this directory so the relative logs/ path resolves:
#   mkdir -p logs && sbatch slurm_profile.sh
#
#SBATCH --job-name=sparse_profile
#SBATCH --partition=medium
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
#SBATCH --output=logs/profile_%j.out

module load GCC/13.2.0

REPO=$SCRATCH/repos/minimal_hnsw
BIN=$REPO/build-release/bin
PROF=$REPO/sparse/profiling
DATA=$REPO/sparse/data/msmarco_full
BASE=$DATA/base_full.csr
QUERIES=$DATA/queries.dev.csr
GT=$DATA/base_full.dev.gt

# The configuration under study, identical to what sparse_hnsw_demo races.
M=16
EFC=200
EF=150
ALPHA=0.8
BETA=3
THREADS=48
HNSW_ARGS="$M $EFC $EF 1 0 0 $ALPHA $BETA $BASE $QUERIES $GT"

# Grace's job environment ships OMP_NUM_THREADS=1, so omp_get_max_threads()
# reports 1 no matter how many cores are allocated. Every rung here names its
# thread count explicitly (bench_scale's list, sparse_profile's mode@threads,
# PROF_BUILD_THREADS), so nothing depends on the default -- but set it anyway so
# a mode written without @threads does not silently run on one core.
export OMP_NUM_THREADS=$THREADS

# nproc honours OMP_NUM_THREADS, so read the allocation from SLURM instead.
echo "### node=$(hostname)  cores_alloc=${SLURM_CPUS_PER_TASK:-?}  cores_online=$(getconf _NPROCESSORS_ONLN)"
echo "### cpu=$(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"
echo "### THP=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
echo "### perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)"
echo "### config: M=$M efC=$EFC ef=$EF alpha=$ALPHA beta=$BETA threads=$THREADS"
numactl --hardware | head -8

if command -v perf >/dev/null 2>&1; then
    HAVE_PERF=1
else
    HAVE_PERF=0
    echo
    echo "### perf NOT available on this cluster -- E2 (hardware counters) and"
    echo "### E6b (cycle hotspots) will be skipped. Every other rung is"
    echo "### wall-clock / in-code-counter based and runs normally."
fi

# Single-core rungs: pin to one core with its memory bound to the LOCAL NUMA
# node, so the micro-benchmark measures the kernel rather than the interconnect.
# (Core 8 is on node 0; on Grace even cores are node 0, odd cores node 1.)
PIN1="numactl --physcpubind=8 --membind=0"

echo; echo "########## E3+E4: ablation and working-set sweep (1 core, local mem) ##########"
# Every micro-benchmark prunes with the index's ALPHA first. The traversal only
# ever streams pruned rows (~215 B vs ~507 B at alpha=0.8), so timing the raw
# matrix would characterise a kernel that never runs.
#
# One process handles the whole sweep -- loading the 9 GB base costs minutes and
# must not be paid per sweep point. Row counts are chosen so the PRUNED working
# set walks past 1 MB L2, the 6 MB STLB reach and 36 MB L3. The last entry (0)
# is the full matrix, which is also the E3 ablation point.
$PIN1 $BIN/bench_distance $BASE $QUERIES 1000000 0 10000,40000,160000,640000,2500000 1 $ALPHA
echo "-------- E3 proper: full working set, 6 variants, 3 reps --------"
$PIN1 $BIN/bench_distance $BASE $QUERIES 2000000 0 0 3 $ALPHA

if [ "$HAVE_PERF" = "1" ]; then
    echo; echo "########## E2/E3 counters: per-variant groups (1 core) ##########"
    for V in 1 3 4 6; do
        echo "======== variant $V ========"
        $PROF/perf_groups.sh $PIN1 $BIN/bench_distance $BASE $QUERIES 2000000 $V 0 1 $ALPHA
    done
fi

echo; echo "########## E5: thread scaling, stream ceiling vs dense kernel ##########"
# Interleaved across both memory controllers, which is how the production runs
# are launched. 24 = one full socket, 48 = both.
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
numactl --interleave=0-1 $BIN/bench_scale $BASE $QUERIES 400000 1,2,4,8,16,24,48 $ALPHA
unset OMP_PROC_BIND

echo; echo "########## E6 + E3b + E1: the real driver (one index build) ##########"
# msmarco_full takes ~8 min to build at 48 threads, so every mode shares a single
# build. Ladder order: replay (the premise check) -> repeat -> batch.
export PROF_BUILD_THREADS=$THREADS
export OMP_PROC_BIND=close
numactl --cpunodebind=0-1 --interleave=0-1 stdbuf -oL -eL \
  $BIN/sparse_profile $HNSW_ARGS \
  "replay@1:2000,repeat@1:2000,batch@1,batch@$THREADS" 0 cold

if [ "$HAVE_PERF" = "1" ]; then
    echo; echo "########## E2: counters on the real search (gated) ##########"
    OMP_NUM_THREADS=1 $PROF/perf_groups.sh $BIN/sparse_profile $HNSW_ARGS batch@1
    echo; echo "########## E6b: what the non-distance time IS ##########"
    OMP_NUM_THREADS=1 $PROF/perf_hotspots.sh $BIN/sparse_profile $HNSW_ARGS batch@1
fi

echo; echo "########## Done ##########"

# Plot in a subshell with its own toolchain: matplotlib on Grace is built
# against GCC/12.3.0, which conflicts with the GCC/13.2.0 the binaries need.
# Never pipe `module load` -- Lmod's `module` is a shell function, and a pipe
# runs it in a subshell where the environment changes are silently discarded.
(
    module purge
    module load GCC/12.3.0 matplotlib/3.7.2
    python3 $PROF/plot_profile.py logs/profile_${SLURM_JOB_ID}.out \
            -o logs/plots_${SLURM_JOB_ID} --pdf
)
