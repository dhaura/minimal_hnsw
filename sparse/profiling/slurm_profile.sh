#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --output=logs/profile_%j.out

module load intel

BIN=$SCRATCH/repos/sparse_hnsw/minimal_hnsw/build/bin
PROF=$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/profiling
DATA=$SCRATCH/repos/sparse_hnsw/minimal_hnsw/sparse/data/msmarco_full
BASE=$DATA/base_full.csr
QUERIES=$DATA/queries.dev.csr
GT=$DATA/base_full.dev.gt
HNSW_ARGS="16 200 150 1 0 0 1 0 $BASE $QUERIES $GT"

echo "### node=$(hostname)  THP=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
echo "### perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"

echo; echo "########## E3: hot/cold ablation (single core, 3 reps) ##########"
taskset -c 8 $BIN/bench_distance $BASE $QUERIES 2000000 0 0 3

echo; echo "########## E4: working-set sweep (variant 3, cold merge) ##########"
for N in 4000 16000 64000 250000 1000000; do
    taskset -c 8 $BIN/bench_distance $BASE $QUERIES 1000000 3 $N
done

echo; echo "########## E3 counters: per-variant groups (single core) ##########"
for V in 1 3 4; do
    echo "======== variant $V ========"
    $PROF/perf_groups.sh taskset -c 8 $BIN/bench_distance $BASE $QUERIES 2000000 $V
done

echo; echo "########## E5: thread scaling, stream ceiling vs merge ##########"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
$BIN/bench_scale $BASE $QUERIES 400000 1,2,4,8,16,32,64,128

echo; echo "########## E0+E1+E2: production search, batch mode ##########"
# Build with all cores, search single-threaded. NOTE: do NOT taskset this
# binary -- it would pin all 128 build threads onto one core. OMP_PROC_BIND
# pins the single search thread.
export PROF_BUILD_THREADS=128
OMP_NUM_THREADS=1 $PROF/perf_groups.sh $BIN/sparse_profile $HNSW_ARGS batch

echo; echo "########## E1 at scale: production search, 64 threads ##########"
OMP_NUM_THREADS=64 $BIN/sparse_profile $HNSW_ARGS batch

echo; echo "########## E3b: repeat-query cold/warm on the real path ##########"
OMP_NUM_THREADS=1 $PROF/perf_groups.sh $BIN/sparse_profile $HNSW_ARGS repeat 2000 cold

# Validates the premise of everything above: that distance() owns the time.
# Run this FIRST when reading the log -- if the share were low, the whole
# distance-kernel analysis would be optimizing a minority of the runtime.
echo; echo "########## E6: does distance() actually own the time? ##########"
OMP_NUM_THREADS=1 $BIN/sparse_profile $HNSW_ARGS replay 2000

echo; echo "########## E6b: what the non-distance time IS ##########"
OMP_NUM_THREADS=1 $PROF/perf_hotspots.sh $BIN/sparse_profile $HNSW_ARGS batch

echo; echo "########## Done ##########"

module load python 2>/dev/null && \
  python3 $PROF/plot_profile.py logs/profile_${SLURM_JOB_ID}.out \
          -o logs/plots_${SLURM_JOB_ID} --pdf
