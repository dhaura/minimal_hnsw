#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=cpu
#SBATCH --output=logs/hnsw_1M_100_%j.out

srun -u -n 1 perf record -F 99 -g -- $SCRATCH/repos/minimal_hnsw/build/bin/hnsw_demo 16 200 150 1 0 0 $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_base.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_query.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_groundtruth.ivecs fvecs $SCRATCH/repos/minimal_hnsw/dense/output
perf report --stdio > $SCRATCH/repos/minimal_hnsw/dense/perf/perf_report_1M_v2.txt

source $SCRATCH/repos/minimal_hnsw/venv/bin/activate
python3 $SCRATCH/repos/minimal_hnsw/dense/scripts/plot_distribution.py $SCRATCH/repos/minimal_hnsw/dense/output --output $SCRATCH/repos/minimal_hnsw/dense/output/plots --bins 60
