#!/bin/bash
# Run the whole profiling ladder on NERSC Perlmutter (AMD Zen3).
#
# Perlmutter CPU node: 2 x AMD EPYC 7763 (Zen3 "Milan"), 64 cores each = 128
# cores / 256 hyperthreads, NPS4 -> EIGHT NUMA domains of 16 cores + 2 memory
# channels (~51 GB/s) each; 32 KB L1d / 512 KB L2 per core, 32 MB L3 per CCX.
#
# Submit from this directory so the relative logs/ path resolves:
#   mkdir -p logs && sbatch slurm_profile_perlmutter.sh
#
#SBATCH --job-name=sparse_profile
#SBATCH --account=m4012
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128   # 64 physical cores x 2 hyperthreads -> 64 threads
#SBATCH --output=logs/profile_%j.out

REPO=$SCRATCH/repos/sparse_hnsw/minimal_hnsw
BIN=$REPO/build/bin
PROF=$REPO/sparse/profiling
DATA=$REPO/sparse/data/msmarco_full
BASE=$DATA/base_full.csr
QUERIES=$DATA/queries.dev.csr
GT=$DATA/base_full.dev.gt

export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:/opt/AMD/aocc-compiler-4.1.0/lib:${LD_LIBRARY_PATH:-}

M=${M:-16}
EFC=${EFC:-200}
EF=${EF:-150}
ALPHA=${ALPHA:-0.8}
BETA=${BETA:-3}
THREADS=${THREADS:-64}
HNSW_ARGS="$M $EFC $EF 1 0 0 $ALPHA $BETA $BASE $QUERIES $GT"

export OMP_NUM_THREADS=$THREADS

echo "### node=$(hostname)  cores_alloc=${SLURM_CPUS_PER_TASK:-?}  cores_online=$(getconf _NPROCESSORS_ONLN)"
echo "### cpu=$(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"
echo "### avx=$(lscpu | grep -o -E 'avx[0-9a-z_]*' | sort -u | tr '\n' ' ')"
echo "### THP=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
echo "### perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null)"
echo "### config: M=$M efC=$EFC ef=$EF alpha=$ALPHA beta=$BETA threads=$THREADS"
echo "### binaries from $BIN"
numactl --hardware | head -20

if command -v perf >/dev/null 2>&1; then
    HAVE_PERF=1
    echo "### perf $(perf --version 2>&1 | awk '{print $NF}') available -- E2/E6b will run"
else
    HAVE_PERF=0
    echo "### perf NOT found -- E2/E6b skipped (unexpected on Perlmutter)"
fi

PIN1="numactl --physcpubind=8 --membind=0"

echo; echo "########## E3+E4: ablation and working-set sweep (1 core, local mem) ##########"

$PIN1 $BIN/bench_distance $BASE $QUERIES 1000000 0 1000,2500,10000,40000,150000,600000,2500000 1 $ALPHA
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

export OMP_PLACES=cores
export OMP_PROC_BIND=spread
numactl --cpunodebind=0-3 --interleave=0-3 \
    $BIN/bench_scale $BASE $QUERIES 400000 1,2,4,8,16,32,64 $ALPHA
echo "-------- whole node (128 threads, all 8 domains) --------"
numactl --interleave=all \
    $BIN/bench_scale $BASE $QUERIES 400000 128 $ALPHA
unset OMP_PROC_BIND

echo; echo "########## E6 + E3b + E1: the real driver (one index build) ##########"

export PROF_BUILD_THREADS=$THREADS
export OMP_PROC_BIND=close
numactl --cpunodebind=0-3 --interleave=0-3 stdbuf -oL -eL \
  $BIN/sparse_profile $HNSW_ARGS \
  "replay@1:2000,repeat@1:2000,batch@1,batch@$THREADS" 0 cold

if [ "$HAVE_PERF" = "1" ]; then
    echo; echo "########## E2: counters on the real search (gated) ##########"
    OMP_NUM_THREADS=1 $PROF/perf_groups.sh $BIN/sparse_profile $HNSW_ARGS batch@1
    echo; echo "########## E6b: what the non-distance time IS ##########"
    OMP_NUM_THREADS=1 $PROF/perf_hotspots.sh $BIN/sparse_profile $HNSW_ARGS batch@1
fi

echo; echo "########## Done ##########"

(
    module load python/3.11-24.1.0 2>/dev/null
    VENV=$SCRATCH/benchmarks/SpKNN/bench-venv-perlmutter
    if [ -x "$VENV/bin/python3" ]; then
        "$VENV/bin/python3" $PROF/plot_profile.py logs/profile_${SLURM_JOB_ID}.out \
                -o logs/plots_${SLURM_JOB_ID} --pdf
    else
        echo "venv not found at $VENV -- plot manually with plot_profile.py" >&2
    fi
)
