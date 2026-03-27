#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/hnswlib_1M_t1_%j.out

export OMP_NUM_THREADS=1
srun -u -n 1 perf record -F 99 -g -- $SCRATCH/repos/minimal_hnsw/build/bin/hnswlib_demo 16 200 150 $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_base.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_query.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_groundtruth.ivecs fvecs
perf report --stdio > $SCRATCH/repos/minimal_hnsw/dense/perf/perf_report_hnswlib_1M_v1.txt
