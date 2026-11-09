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
├── sparse/          # Sparse vector HNSW implementation (SPLADE, inner product)
│   ├── sparse_hnsw.h/.cpp    # Sparse HNSW index
│   ├── csr_matrix.h          # Compact CSR matrix loader (uint16 indices, fp16 values)
│   ├── prune.h               # Mass-ratio pruning (alpha)
│   ├── quant_csr.h           # u8 traversal rows: packed (QuantCSR) or fixed 320 B slots (QuantSlots)
│   ├── inverted_seed.h       # Per-term top-doc seed table
│   ├── dist_log.h            # Per-distance-call log (compile-time, SPARSE_HNSW_DIST_LOG)
│   ├── bench_common.h        # Shared list parsing / CSV writing for the drivers
│   ├── main.cpp              # Single-configuration demo
│   ├── sweep_main.cpp        # ef x beta x seeds x stop x gate sweep over one build
│   ├── decouple_main.cpp     # Search-side alpha sweep over one build
│   ├── diag_sweep_main.cpp   # Recall-loss decomposition
│   ├── access_profile_main.cpp # Row-access distribution profile
│   ├── baselines/            # Competitor methods, built and run from this repo
│   │   ├── fetch_baselines.sh    # Clones GrassRMA / vsag / pyanns at pinned commits + patches
│   │   ├── patches/              # Local fixes to the baseline sources (applied by the scripts)
│   │   ├── grassRMA_main.cpp, sindi_main.cpp, sindi_sweep_main.cpp   # C++ drivers
│   │   ├── seismic/ kannolo/ pyanns/ grassRMA/   # Python drivers + Perlmutter runners
│   │   └── common/               # Python harness, venv build script, smoke test
│   ├── profiling/            # Profiling ladder (sparse_profile, perf scripts)
│   ├── scripts/              # SLURM launchers and analysis, one folder per study
│   │   ├── bench_env_perlmutter.sh   # Threads, pinning, NUMA, dataset paths (sourced by every runner)
│   │   └── plotting/             # Result merging / Pareto / figure tools
│   ├── scripts/results/      # Saved results and summary .md files, same folders
│   │                         #   (results/baselines/<dataset>/<method>/: baseline result CSVs)
│   ├── .clang-format         # Code style for the sparse sources
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

- **CSR-backed storage**: Vectors are never materialized densely; the index stores only row ids and reads `(index, value)` pairs straight out of the shared `CSRMatrix`.
- **Compact 4-byte entries**: Column indices are `uint16_t` (valid while `ncol <= 65536`, checked at load) and values are IEEE `fp16`, halving the bytes streamed per distance call.
- **Dense-scatter distance**: The query row is scattered into a dense fp32 buffer once per query, then gathered per candidate; returns `1 - <q, p>`.
- **Mass-ratio pruning (`alpha`)**: Per row, keep the fewest largest entries reaching `alpha` of the row weight. `beta > 1` re-scores the top `k*beta` candidates against the unpruned matrix.
- **Scalar quantization (`quantize`)**: Traversal scores against u8 rows (`[u16 n][fp16 scale][n x u16 idx][n x u8 code]`); the `beta` refine pass still uses fp16.
- **Fixed-stride row slots (`slot_size`, default 320 B)**: Each u8 row sits in a 64 B-aligned slot at `base + id * slot_size`, so a distance call needs no offset lookup; longer rows live in an overflow blob.
- **Inverted-index seeding (`seed_top_k`, `seed_terms`, `seed_per_term`)**: Enter layer 0 at the top documents of the query's heaviest terms instead of descending from the entry point alone.
- **Inline top-K gate (`inline<K>`)**: Slots store each row's K heaviest terms first; once the result list is full, layer 0 skips a neighbor whose top-K terms miss the query, before fetching the rest of its row.
- **Layer-0 stops**: score gap (`g<G>`: stop when the next candidate scores below `(1 - G)` x the k-th best so far) or patience (`<P>`: stop after P expansions that do not improve the top k).
- **Flat neighbor lists**: Level 0 uses a fixed-size contiguous block per node (upper layers share one packed buffer with per-node offsets).
- **Parallel construction and search**: OpenMP-parallel `addPointsBatch` / `searchKNNBatch`, with one mutex per element guarding its neighbor lists and a global lock for the entry point.
- **Prefetching**: Software prefetch of visited bits and neighbor rows while layer 0 filters a node's neighbor list.

### Build

All steps run on a Perlmutter **login node** (they need network access).

**1. Baseline sources.** Clones GrassRMA, vsag and pyanns into `sparse/` at the
commits behind the paper results and applies `sparse/baselines/patches/`. An
existing checkout is only checked, never modified.

```bash
REPO=$SCRATCH/repos/sparse_hnsw/minimal_hnsw
bash $REPO/sparse/baselines/fetch_baselines.sh
```

**2. SparseHNSW (and `grassRMA_demo`), Intel compiler.** Run the `module swap`
on its own line: piping it (`module swap ... | tail`) loses the swap and CMake
then configures a GCC build. A configured tree shows `OpenMP_CXX_LIB_NAMES=iomp5`
in `build/CMakeCache.txt`.

```bash
module swap PrgEnv-gnu PrgEnv-intel
cmake -S $REPO -B $REPO/build -DCMAKE_BUILD_TYPE=Release
cmake --build $REPO/build -j16
```

Binaries land in `build/bin`: `sparse_hnsw_demo`, `sparse_hnsw_sweep`,
`sparse_decouple_sweep`, `sparse_diag_sweep`, `sparse_access_profile`,
`sparse_hnsw_demo_distlog`, `grassRMA_demo`, and the profiling ladder
(`bench_distance`, `bench_scale`, `sparse_profile`).

**3. SINDI: vsag plus a GCC build tree** (default PrgEnv-gnu shell, GCC 14):

```bash
cd $REPO/sparse
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

`build-gnu/bin/sindi_sweep` must link `libgomp` only (`ldd`); the SINDI runner
refuses a binary that also pulls in `libiomp5`.

**4. Python baselines venv** (SEISMIC, kANNolo, GrassRMA's `sparse_hnswlib`,
PyANNS), at `$SCRATCH/benchmarks/minimal_hnsw/bench-venv-perlmutter` (set
`SPKNN_VENV` to change it; an existing venv there is replaced):

```bash
bash $REPO/sparse/baselines/common/build_bench_venv_perlmutter.sh
```

Every version is pinned. SEISMIC 0.5.1 and kANNolo 0.7.1 are built from their
PyPI source releases plus `patches/pyseismic_lsr-0.5.1.patch` (a regenerated
`Cargo.lock`: its locked git dependency no longer exists upstream) and
`patches/kannolo-0.7.1.patch` (`lto = "fat"`), with maturin 1.14.1 and Rust
`nightly-2026-08-04`. On Perlmutter this reproduces the extension modules behind
the paper results byte for byte. The Rust toolchain goes to `$SCRATCH/.rustup`.

**5. Smoke test.** Runs every method (SparseHNSW, SINDI, GrassRMA, kANNolo,
SEISMIC, PyANNS) at two settings on `msmarco_small` and checks each one returns
ranked results. A few minutes with 8 threads on a login node, or as a job:

```bash
BENCH_THREADS=8 bash $REPO/sparse/baselines/common/smoke_test_perlmutter.sh
# or: cd $REPO/sparse/baselines/common && sbatch smoke_test_perlmutter.sh
```

### Datasets

|              | `msmarco_full`       | `nq_splade`        | `msmarco_v2_splade`     |
| ------------ | -------------------- | ------------------ | ----------------------- |
| base         | `base_full.csr`      | `base_nq.csr`      | `base_v2_splade.csr`    |
| queries      | `queries.dev.csr`    | `queries.test.csr` | `queries.$QSET.csr`     |
| ground truth | `base_full.dev.gt`   | `base_nq.test.gt`  | `base_v2_splade.$QSET.gt` |
| docs / queries / dim | 8,841,823 / 6,980 / 30,109 | 2,680,893 / 3,452 / 30,522 | 138,364,198 / 3,903 / 30,522 |
| nnz          | 1.12e9 (0.26x uint32) | 4.12e8 | **1.75e10 (4.07x uint32)** |

All three are SPLADE++ CoCondenser-EnsembleDistil
(`naver/splade-cocondenser-ensembledistil`) - the big-ann sparse track and the
Pyserini `splade-pp-ed` corpora are the same model, so v1 and v2 results are
model-comparable. They are *not* index-comparable: the big-ann packaging
compacts the BERT vocabulary to 30,109 live dimensions and ships raw float
weights, while the Pyserini corpora keep all 30,522 and ship integer impacts
(documents x100, queries x1000). Never mix a query file across the two.

```bash
# msmarco_full (big-ann sparse track). base_1M / base_small are drop-in smaller
# variants from the same bucket.
mkdir -p $REPO/sparse/data/msmarco_full && cd $REPO/sparse/data/msmarco_full
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_full.csr.gz
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/queries.dev.csr.gz
wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_full.dev.gt
gunzip base_full.csr.gz queries.dev.csr.gz

# msmarco_v2_splade: SPLADE++ ED pre-encoded corpus (~70 GB tar) + pre-encoded
# topics from castorini/eval. Budget ~350 GB and about 2 h end to end.
mkdir -p $REPO/sparse/data/msmarco_v2_splade && cd $REPO/sparse/data/msmarco_v2_splade
curl -O https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco_v2_passage_splade_pp_ed.tar
tar -xf msmarco_v2_passage_splade_pp_ed.tar          # 1000 shards of *.jsonl.gz
curl -sSL -o bert_vocab.txt https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
# Topics live in castorini/eval (the repo anserini-tools was renamed to).
T=https://raw.githubusercontent.com/castorini/eval/master/topics
for q in dev dev2; do curl -sSLO $T/topics.msmarco-v2-passage.$q.splade-pp-ed.tsv.gz; done
for q in dl21 dl22 dl23; do curl -sSLO $T/topics.$q.splade-pp-ed.tsv.gz; done
# Converts topics -> queries.*.csr, 1000 shards -> base_v2_splade.csr, then
# computes exact ground truth for every query set. ~30 min on one node.
sbatch $REPO/sparse/scripts/msmarco_v2/build_msmarco_v2_splade_perlmutter.sh
# LIMIT_SHARDS=N converts only the first N shards (138,364 passages each) if you
# want a smaller prototype without touching the full corpus.

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

# Usage (run the binary with no arguments for the full list, incl. the distlog options)
# ./bin/sparse_hnsw_demo <M> <ef_construction> <ef> <alpha> <beta> <input_filepath>
#                        <query_filepath> <gt_filepath> <results_csv_path>
#                        [quantize=0] [seed_top_k=0] [seed_terms=0] [seed_per_term=1] ...

# msmarco_full
$LAUNCH $REPO/build/bin/sparse_hnsw_demo 32 200 200 0.85 3 \
  $DATA/msmarco_full/base_full.csr \
  $DATA/msmarco_full/queries.dev.csr \
  $DATA/msmarco_full/base_full.dev.gt \
  results_msmarco_full.csv 1 8 8 4

# nq_splade
$LAUNCH $REPO/build/bin/sparse_hnsw_demo 32 200 200 0.85 3 \
  $DATA/nq_splade/base_nq.csr \
  $DATA/nq_splade/queries.test.csr \
  $DATA/nq_splade/base_nq.test.gt \
  results_nq_splade.csv 1 8 8 4
```

```bash
# msmarco_v2_splade
export OMP_NUM_THREADS=128 OMP_PLACES=cores OMP_PROC_BIND=spread MKL_NUM_THREADS=1
numactl --interleave=all $REPO/build/bin/sparse_hnsw_demo 32 200 200 0.85 3 \
  $DATA/msmarco_v2_splade/base_v2_splade.csr \
  $DATA/msmarco_v2_splade/queries.dev.csr \
  $DATA/msmarco_v2_splade/base_v2_splade.dev.gt \
  results_msmarco_v2.csv 1 8 8 4
```

The paper configuration (u8 slots, seeds, inline16 gate, score-gap stop) runs
through `sparse_hnsw_sweep`, which builds once and sweeps ef x beta x stop rule x
gate over the same index; see the header of `sparse/sweep_main.cpp` for the list
formats:

```bash
# ... <model> <repeats> <warmup> <quantize> <seed_top_k> <seed_spec> <stop_list> <slot_size> <gate_layouts>
$LAUNCH $REPO/build/bin/sparse_hnsw_sweep 32 200 50,100,200,400 0.85 2,3,4,8 \
  $DATA/msmarco_full/base_full.csr $DATA/msmarco_full/queries.dev.csr \
  $DATA/msmarco_full/base_full.dev.gt sweep_msmarco_full.csv \
  SparseHNSW 5 1 1 8 8:4 0,g0.1 -320 off,inline16
```

`stop_list` 0 = no stop, `g0.1` = score gap 0.1; `slot_size` -320 = slots only;
`gate_layouts` off = no gate, `inline16` = top-16 gate.

Or through the sweep script, which resolves every path from `SPKNN_DATASET` and
refuses to start if the dataset needs `--interleave=all` and did not get it:

```bash
sbatch --cpus-per-task=256 \
  --export=ALL,SPKNN_DATASET=msmarco_v2_splade,PRESET=frontier \
  $REPO/sparse/scripts/run_hnsw_sweep_perlmutter.sh
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

The Python baselines (`sparse/baselines/{seismic,kannolo,pyanns,grassRMA}/*_ex.py`)
need the job environment and the venv; `-h` lists each driver's options. Without
the environment the PyANNS and GrassRMA modules fail to import (it preloads the
system `libstdc++`):

```bash
module load python/3.11-24.1.0
source $REPO/sparse/scripts/bench_env_perlmutter.sh
source $SPKNN_VENV/bin/activate
python3 $REPO/sparse/baselines/kannolo/kannolo_mt_ex.py -h
```

### Scripts (SLURM, Perlmutter)

Every runner sources `sparse/scripts/bench_env_perlmutter.sh`, which sets
threads, pinning and NUMA policy and resolves `SPKNN_BASE` / `SPKNN_QUERIES` /
`SPKNN_GT` from `SPKNN_DATASET` (`msmarco_full`, `nq_splade` or
`msmarco_v2_splade`). Runners find the repo at `$SCRATCH/repos/sparse_hnsw/minimal_hnsw`;
set `SPKNN_HNSW_REPO` for another clone. Submit from the runner's folder (logs go
to its `logs/`):

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
| `sketch_theory/run_inline_sweep_perlmutter.sh` | off / gap / inline<K> / inline<K>+gap over one build (`sparse_hnsw_sweep`) |
| `adaptive_stopping/run_stop_online_perlmutter.sh` | score-gap and patience stops online                     |
| `quant_slots/run_*_perlmutter.sh`    | packed vs slotted u8 rows, slot size, memory                          |
| `../baselines/<method>/run_<method>_perlmutter.sh` | SEISMIC, kANNolo, PyANNS, GrassRMA (Python drivers; `_v2_` = msmarco_v2) |

Each study folder in `sparse/scripts/` has its analysis scripts next to its
launchers; results and a summary `.md` go to the folder of the same name in
`sparse/scripts/results/`.

### Usage Example

```cpp
#include "sparse_hnsw.h"

// Load the base and query matrices in CSR format
CSRMatrix* base  = new CSRMatrix("base_full.csr", true);
CSRMatrix* query = new CSRMatrix("queries.dev.csr", true);

sparse_hnsw::SPARSE_HNSW index(base->ncol, base, /*M=*/32, /*ef_construction=*/200,
                               base->nrow, /*alpha=*/0.85f, /*beta=*/3);

// alpha < 1: build and traverse on the pruned matrix, refine against the original
index.setPrunedDataMatrix(index.pruneMatrix(base));

// Insert all rows of the base matrix (OpenMP-parallel)
index.addPointsBatch(base->nrow);

// Optional search-side knobs
index.enableQuantizedTraversal();          // u8 rows in 320 B slots
index.buildSeedTable(/*top_k=*/8);
index.setSeedParams(/*terms=*/8, /*per_term=*/4);
index.enableInlineGate(/*k=*/16);          // needs slots
index.setStopGap(0.1);                     // or setPatience(P)

// Search: k nearest neighbors for every query row (OpenMP-parallel)
std::vector<uint32_t> labels;
index.searchKNNBatch(query, query->nrow, /*k=*/10, /*ef=*/200, labels);

// Or a single query row
auto results = index.searchKNN(/*query_id=*/0, query, 10, 200);
```
