# minimal_hnsw

A minimal implementation of Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search.

## Project Structure

```
minimal_hnsw/
├── dense/           # Dense vector HNSW implementation
│   ├── hnsw.h                # HNSW header file
│   ├── hnsw.cpp              # HNSW implementation
│   ├── hilbert_ordering.h    # Hilbert curve dataset reordering header
│   ├── hilbert_ordering.cpp  # Faiss PCA projection + Hilbert curve ordering
│   ├── main.cpp              # Demo application
│   ├── hnswlib_main.cpp      # hnswlib benchmark driver
│   ├── scripts/              # Run scripts and plotting utilities
│   ├── perf/                 # Captured perf reports
│   ├── output/               # Timing stats, distributions, plots
│   └── CMakeLists.txt
├── sparse/          # Sparse vector HNSW implementation
│   ├── sparse_hnsw.h/.cpp    # Sparse HNSW index
│   ├── csr_matrix.h          # Compact CSR matrix loader (uint16 indices, fp16 values)
│   ├── prune.h               # Mass-ratio pruning (alpha)
│   ├── quant_csr.h           # uint8-quantized traversal copy
│   ├── inverted_seed.h       # Per-term top-doc seed table
│   ├── bench_common.h        # Shared list parsing / CSV writing for the drivers
│   ├── main.cpp              # Single-configuration demo
│   ├── sweep_main.cpp        # ef x beta sweep over one build
│   ├── decouple_main.cpp     # Search-side alpha sweep over one build
│   ├── diag_sweep_main.cpp   # Recall-loss decomposition
│   ├── grassRMA_main.cpp     # grassRMA benchmark driver
│   ├── sindi_main.cpp        # SINDI (vsag) benchmark driver
│   ├── sindi_sweep_main.cpp  # SINDI parameter sweep
│   ├── profiling/            # Profiling ladder
│   ├── scripts/              # SLURM launchers, dataset prep, analysis
│   └── CMakeLists.txt
└── CMakeLists.txt   # Root CMake configuration
```

## Building the Project

### Requirements
- CMake 3.10 or higher
- C++11 compatible compiler (C++17 for the sparse benchmark drivers)

### Build and Run Instructions (dense)

```bash
# Clone hnswlib (for benchmarking) and faiss (for the PCA used by hilbert ordering)
cd dense
git clone https://github.com/nmslib/hnswlib.git
git clone https://github.com/facebookresearch/faiss.git
cd ../

# Build faiss
cmake -S $SCRATCH/repos/minimal_hnsw/dense/faiss -B $SCRATCH/repos/minimal_hnsw/dense/faiss/build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_EXTRAS=OFF -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$SCRATCH/repos/minimal_hnsw/dense/faiss/install
cmake --build $SCRATCH/repos/minimal_hnsw/dense/faiss/build -j4
cmake --install $SCRATCH/repos/minimal_hnsw/dense/faiss/build

# Configure and build
module load intel # For MKL support
cmake -S $SCRATCH/repos/minimal_hnsw -B $SCRATCH/repos/minimal_hnsw/build -DFAISS_ROOT=$SCRATCH/repos/minimal_hnsw/dense/faiss/install
cmake --build $SCRATCH/repos/minimal_hnsw/build -j4

# Download the dataset
cd dense
mkdir data
cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzvf sift.tar.gz

# Run the demo
cd ../../build
./bin/hnsw_demo 16 200 150 1 0 0 1 0 $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_base.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_query.fvecs $SCRATCH/repos/minimal_hnsw/dense/data/sift/sift_groundtruth.ivecs fvecs $SCRATCH/repos/minimal_hnsw/dense/output/timing_stats/mkl_stats.csv

# Usage
# ./bin/hnsw_demo <M> <ef_construction> <ef> <use_heuristic> <extend_candidates> <keep_pruned> <use_mkl> <mklThreshold> <input_filepath> <query_filepath> <gt_filepath> <file_type> <stat_file>
```

## Dense HNSW Implementation

The dense folder contains a simple HNSW implementation for dense vectors with the following features:

- **Basic HNSW structure**: Multi-layer graph with hierarchical navigation
- **Point insertion**: Add vectors to the index with automatic layer assignment
- **K-NN search**: Find k nearest neighbors for a query vector
- **Euclidean distance**: Uses L2 distance metric

### Usage Example

```cpp
#include "hnsw.h"

// Create index: dimension=2, M=16, ef_construction=200
HNSW index(2, 16, 200, 1000);

// Add points
std::vector<float> point = {1.0f, 2.0f};
index.addPoint(point, 0);

// Search for k nearest neighbors
std::vector<float> query = {1.0f, 2.0f};
auto results = index.searchKNN(query, 5);
```

## Sparse HNSW Implementation

The sparse folder contains an HNSW implementation for high-dimensional sparse vectors stored in CSR format, targeting inner-product (MIPS) retrieval on MS MARCO and NQ-SPLADE. It shares the dense algorithm structure but replaces the vector storage, the distance kernel and the search-side knobs:

- **CSR-backed storage**: Vectors are never materialized densely; the index stores only row ids and reads `(index, value)` pairs straight out of the shared `CSRMatrix`
- **Compact 4-byte entries**: Column indices are `uint16_t` (valid while `ncol <= 65536`, checked at load) and values are IEEE `fp16`, halving the bytes streamed per distance call
- **Dense-scatter distance**: The query row is scattered into a dense fp32 buffer once per query, then gathered per candidate; returns `1 - <q, p>`
- **Mass-ratio pruning (`alpha`)**: Per row, keep the fewest largest-magnitude entries reaching `alpha` of the row weight. `beta > 1` re-scores the top `k*beta` candidates against the unpruned matrix
- **Scalar quantization (`quantize`)**: Traversal scores against a uint8 copy of the matrix; the `beta` refine pass still uses fp16
- **Inverted-index seeding (`seed_top_k`, `seed_terms`, `seed_per_term`)**: Enter layer 0 at the top documents of the query's heaviest terms instead of descending from the entry point alone
- **Adaptive termination (`patience`)**: Stop layer 0 after N consecutive expansions that do not improve the running top-k
- **Flat neighbor lists**: Level 0 uses a fixed-size contiguous block per node; upper layers share one packed buffer with per-node offsets
- **Parallel construction and search**: OpenMP-parallel `addPointsBatch` / `searchKNNBatch`, with one mutex per element guarding its neighbor lists and a global lock for the entry point
- **Prefetching**: Software prefetch of visited bits, `indptr`, and neighbor vectors, driven by a neighbor filtering loop
- **Neighbor selection**: Both naive top-M and the HNSW heuristic, with optional `extend_candidates` and `keep_pruned`

### Build

```bash
REPO=$SCRATCH/repos/sparse_hnsw/minimal_hnsw

# Clone grassRMA (for benchmarking)
cd $REPO/sparse
git clone https://github.com/Leslie-Chung/GrassRMA.git

module swap PrgEnv-gnu PrgEnv-intel

cmake -S $REPO -B $REPO/build -DCMAKE_BUILD_TYPE=Release
cmake --build $REPO/build -j16
```

Binaries land in `build/bin`: `sparse_hnsw_demo`, `sparse_hnsw_sweep`,
`sparse_decouple_sweep`, `sparse_diag_sweep`, `grassRMA_demo`, and the
profiling ladder (`bench_distance`, `bench_scale`, `sparse_profile`).

The SINDI baseline needs vsag, plus a **GCC** build tree of this repo:

```bash
cd $REPO/sparse
git clone https://github.com/antgroup/vsag.git
cmake -S vsag -B vsag/build-release -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_TOOLS=OFF \
      -DENABLE_WERROR=OFF -DNUM_BUILDING_JOBS=16 \
      -DCMAKE_INSTALL_PREFIX=$PWD/vsag/install-znver3
cmake --build vsag/build-release --parallel 16
cmake --install vsag/build-release

cmake -S $REPO -B $REPO/build-gnu -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
cmake --build $REPO/build-gnu --target sindi_demo sindi_sweep -j16
```

### Datasets

|              | `msmarco_full`       | `nq_splade`        |
| ------------ | -------------------- | ------------------ |
| base         | `base_full.csr`      | `base_nq.csr`      |
| queries      | `queries.dev.csr`    | `queries.test.csr` |
| ground truth | `base_full.dev.gt`   | `base_nq.test.gt`  |
| docs / queries / dim | 8,841,823 / 6,980 / 30,109 | 2,680,893 / 3,452 / 30,522 |

```bash
# msmarco_full (big-ann sparse track). base_1M / base_small are drop-in smaller
# variants from the same bucket.
mkdir -p $REPO/sparse/data/msmarco_full && cd $REPO/sparse/data/msmarco_full
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_full.csr.gz
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/queries.dev.csr.gz
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_full.dev.gt
gunzip base_full.csr.gz queries.dev.csr.gz

# nq_splade: documents.tar.gz + queries.tar.gz from
# https://huggingface.co/datasets/tuskanny/seismic-nq-splade
mkdir -p $REPO/sparse/data/nq_splade && cd $REPO/sparse/data/nq_splade
tar -xzf documents.tar.gz && tar -xzf queries.tar.gz
module load python/3.11-24.1.0
python3 ../../scripts/convert_seismic_json_to_csr.py --json 'documents/*.json' --out base_nq.csr
python3 ../../scripts/convert_seismic_json_to_csr.py --json queries/queries.test.json --out queries.test.csr
python3 ../../scripts/compute_gt.py --base base_nq.csr --queries queries.test.csr --out base_nq.test.gt --k 10
```

### Running

```bash
REPO=$SCRATCH/repos/sparse_hnsw/minimal_hnsw
DATA=$REPO/sparse/data

# 64 threads on one half of the node. For the full node use
# OMP_NUM_THREADS=128 and `numactl --interleave=all`.
export OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=spread MKL_NUM_THREADS=1
LAUNCH="numactl --cpunodebind=0-3 --interleave=0-3"

# Usage
# ./bin/sparse_hnsw_demo <M> <ef_construction> <ef> <use_heuristic> <extend_candidates>
#                        <keep_pruned> <alpha> <beta> <input_filepath> <query_filepath>
#                        <gt_filepath> <results_csv_path>
#                        [quantize=0] [seed_top_k=0] [seed_terms=0] [seed_per_term=1]

# msmarco_full
$LAUNCH $REPO/build/bin/sparse_hnsw_demo 32 200 200 1 0 0 0.85 3 \
  $DATA/msmarco_full/base_full.csr \
  $DATA/msmarco_full/queries.dev.csr \
  $DATA/msmarco_full/base_full.dev.gt \
  results_msmarco_full.csv 1 8 8 4

# nq_splade
$LAUNCH $REPO/build/bin/sparse_hnsw_demo 32 200 200 1 0 0 0.85 3 \
  $DATA/nq_splade/base_nq.csr \
  $DATA/nq_splade/queries.test.csr \
  $DATA/nq_splade/base_nq.test.gt \
  results_nq_splade.csv 1 8 8 4
```

`k` is read from the ground-truth file. `alpha=1.0` disables pruning, and the
`beta` refine pass with it; `quantize`/`seed_*` are off when 0.

Baselines, same dataset paths:

```bash
# ./bin/grassRMA_demo <M> <ef_construction> <ef> <input_filepath> <query_filepath> <gt_filepath>
$LAUNCH $REPO/build/bin/grassRMA_demo 32 200 200 \
  $DATA/nq_splade/base_nq.csr $DATA/nq_splade/queries.test.csr $DATA/nq_splade/base_nq.test.gt

# ./bin/sindi_demo <doc_prune_ratio> <query_prune_ratio> <n_candidate>
#                  <input_filepath> <query_filepath> <gt_filepath>
#                  [term_prune_ratio=0] [window_size=50000] [use_reorder=1] [use_quantization=0|1|fp16]
# n_candidate=0 means n_candidate=k. Note: build-gnu, not build.
$LAUNCH $REPO/build-gnu/bin/sindi_demo 0.35 0.5 20 \
  $DATA/msmarco_full/base_full.csr $DATA/msmarco_full/queries.dev.csr $DATA/msmarco_full/base_full.dev.gt
```

### Scripts (SLURM, Perlmutter)

`sparse/scripts/*_perlmutter.sh` source `bench_env_perlmutter.sh`, which sets
threads, pinning and NUMA policy and resolves `SPKNN_BASE` / `SPKNN_QUERIES` /
`SPKNN_GT` from `SPKNN_DATASET` (`msmarco_full` or `nq_splade`):

```bash
cd $REPO/sparse/scripts
sbatch --export=ALL,SPKNN_DATASET=msmarco_full run_alpha_beta_sweep_perlmutter.sh
sbatch --export=ALL,SPKNN_DATASET=nq_splade    run_alpha_beta_sweep_perlmutter.sh
```

| script                              | sweeps                                                                |
| ----------------------------------- | --------------------------------------------------------------------- |
| `run_alpha_beta_sweep_perlmutter.sh` | alpha x beta x ef, one index build per alpha (`sparse_hnsw_sweep`)     |
| `run_hnsw_sweep_perlmutter.sh`       | ef at a single alpha/beta                                             |
| `run_decouple_perlmutter.sh`         | search-side alpha over one build; `PRESET=baseline\|highrecall`        |
| `run_sindi_sweep_perlmutter.sh`      | SINDI doc/query prune x n_candidate (`build-gnu/bin/sindi_sweep`)      |

### Usage Example

```cpp
#include "sparse_hnsw.h"

// Load the base and query matrices in CSR format
CSRMatrix* base  = new CSRMatrix("base_full.csr", true);
CSRMatrix* query = new CSRMatrix("queries.dev.csr", true);

sparse_hnsw::SPARSE_HNSW index(base->ncol, base, /*M=*/32, /*ef_construction=*/200,
                               base->nrow, /*use_heuristic=*/true,
                               /*extend_candidates=*/false, /*keep_pruned=*/false,
                               /*alpha=*/0.85f, /*beta=*/3);

// alpha < 1: build and traverse on the pruned matrix, refine against the original
index.setPrunedDataMatrix(index.pruneMatrix(base));

// Insert all rows of the base matrix (OpenMP-parallel)
index.addPointsBatch(base->nrow);

// Optional search-side knobs
index.enableQuantizedTraversal();
index.buildSeedTable(/*top_k=*/8);
index.setSeedParams(/*terms=*/8, /*per_term=*/4);
index.setPatience(256);

// Search: k nearest neighbors for every query row (OpenMP-parallel)
std::vector<uint32_t> labels;
index.searchKNNBatch(query, query->nrow, /*k=*/10, /*ef=*/200, labels);

// Or a single query row
auto results = index.searchKNN(/*query_id=*/0, query, 10, 200);
```

## Future Work

- **SIMD sparse intersection**: Vectorize the merge loop (AVX-512 gather/conflict detection) instead of the scalar branchless walk
- **Locality-improving reordering**: Finish the PCA + Hilbert curve pre-ordering (currently stubbed out in `sparse/main.cpp`) so neighbor lists touch nearby cache lines
- **Support for `ncol > 65536`**: Widen `IndiceDataPair::indice` or block-partition the column space
- **Serialization/deserialization**: Persist a built index instead of rebuilding per run
- **Additional distance metrics**: Currently inner product only (sparse) and L2 (dense)
