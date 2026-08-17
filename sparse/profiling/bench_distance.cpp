// E3 (ablation) + E4 (working-set sweep) of the profiling ladder.
//
// Isolates the memory cost of the PRODUCTION distance kernel from its ALU cost.
// The production kernel is now SPARSE_HNSW::distanceDense(): the query row is
// scattered once per query into a dense fp32 array of length `dim`, and each
// document row is then a straight-line SIMD loop that gathers q_dense[col] and
// multiplies by the fp16 value. There is no sorted-list merge anywhere on the
// search path any more: the beta>1 refine pass was switched to the same dense
// kernel (2026-08-12), so the merge is now BUILD-ONLY (insertion, where the
// inserted row itself is the query). Variant 6 keeps it as a historical
// before/after baseline, not as a live search-path kernel.
//
// That change matters for this rung. The old merge advanced its two pointers
// from data it had just loaded (a serial, data-dependent chain that starved
// memory-level parallelism). The dense loop has NO such dependency: every
// iteration's address is known as soon as p[i].indice lands, so the hardware
// can run far ahead. What it adds instead is a second random access stream --
// the gather into the 120 KB q_dense buffer (dim=30109 x 4 B), which is L2-
// resident but not L1-resident. V1 is what prices that in.
//
// The regimes:
//   1 HOT dense         : full kernel, doc rows L1-resident   -> T_compute
//   2 COLD stream       : touch the same bytes, no ALU        -> T_load (max MLP)
//   3 COLD dense        : the production kernel               -> T_total
//   4 COLD dense + pf   : batched prefetch (as searchLayer)   -> async gather
//   5 COLD gather+dense : memcpy batch to scratch, then dense -> staged pipeline
//   6 COLD merge        : the OLD sparse merge kernel         -> old vs new
//   7 COLD quant u8     : the CURRENT production kernel       -> V3/V7 = byte diet
//
// Read-out:
//   memory-attributable share = (V3 - V1) / V3
//   fusion penalty            = V3 - (V1 + V2)   (positive => the kernel's own
//                                                 structure is destroying MLP)
//   pipelining bound          = max(V1, V2)      (best case for perfect overlap)
//   kernel speedup            = V6 / V3          (what the dense rewrite bought)
//
// Usage:
//   bench_distance <base.csr> <queries.csr> [ncalls=2000000] [variant=0]
//                  [maxrow=0] [reps=1]
//     variant : 0 = all, 1..6 = single variant (use with perf_groups.sh so
//               counters are attributable to one regime)
//     maxrow  : restrict random ids to the first maxrow rows -> E4 sweep of
//               the working-set size. Accepts a comma-separated LIST, run
//               back-to-back in one process (e.g. 4000,16000,64000,250000,0);
//               0 or omitted means the whole base. Loading msmarco_full costs
//               minutes, so the sweep must not pay it once per point.
//     reps    : repeat everything; take the best rep on shared/noisy nodes
//     alpha   : mass-ratio pruning, applied to the base matrix before any
//               timing. THIS SHOULD MATCH THE INDEX'S alpha. The traversal only
//               ever streams pruned rows (~215 B at alpha=0.8 vs ~507 B
//               unpruned), so profiling the raw matrix measures a kernel that
//               never actually runs. 1.0 = no pruning.
#include <unistd.h>
#include "csr_matrix.h"
#include "prune.h"
#include "quant_csr.h"
#include "perf_ctl.h"
#include <chrono>
#include <random>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

static const int HOPS_BATCH = 32;   // ~= level-0 degree (2*M with M=16)

// The fp16 dense kernel: SPARSE_HNSW::distanceDense(), verbatim.
__attribute__((always_inline))
static inline float dense(const float* qd, const IndiceDataPair* p, uint32_t pn) {
    float res = 0.0f;
    #pragma omp simd reduction(+:res)
    for (uint32_t i = 0; i < pn; ++i) {
        res += static_cast<float>(p[i].data) * qd[p[i].indice];
    }
    return 1.0f - res;
}

// The OLD kernel: branchless sorted-list merge. No longer on any search path
// (SPARSE_HNSW::distance() is now build-only); kept as the before/after
// baseline for what the dense rewrite bought.
__attribute__((always_inline))
static inline float merge(const IndiceDataPair* q, uint32_t qn,
                          const IndiceDataPair* p, uint32_t pn) {
    const IndiceDataPair* qe = q + qn;
    const IndiceDataPair* pe = p + pn;
    float res = 0;
    while (q < qe && p < pe) {
        const int32_t qc = q->indice;
        const int32_t pc = p->indice;
        res += (qc == pc) ? static_cast<float>(q->data) * static_cast<float>(p->data) : 0.0f;
        q += (qc <= pc);
        p += (pc <= qc);
    }
    return 1.0f - res;
}

// THE CURRENT PRODUCTION KERNEL: SPARSE_HNSW::distanceQuant()
__attribute__((always_inline))
static inline float quant(const float* qd, const uint8_t* p) {
    uint16_t n16;
    _Float16 scale16;
    std::memcpy(&n16, p, sizeof(n16));
    std::memcpy(&scale16, p + 2, sizeof(scale16));
    const uint32_t pn = n16;
    const uint16_t* idx = reinterpret_cast<const uint16_t*>(p + sparse_hnsw::QuantCSR::kHeader);
    const uint8_t* code = p + sparse_hnsw::QuantCSR::kHeader + 2 * static_cast<size_t>(pn);
    float res = 0.0f;
    #pragma omp simd reduction(+:res)
    for (uint32_t i = 0; i < pn; ++i) {
        res += static_cast<float>(code[i]) * qd[idx[i]];
    }
    return 1.0f - res * static_cast<float>(scale16);
}

// Pure streaming read of a row: no ALU on the values, high ILP. Isolates memory.
__attribute__((always_inline))
static inline uint64_t stream_row(const IndiceDataPair* p, uint32_t pn) {
    uint64_t acc = 0;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
    for (uint32_t i = 0; i < pn; ++i) acc += w[i];
    return acc;
}

template <typename F>
static void timeit(const char* name, uint64_t ncalls, double bytes, F&& f) {
    perf_ctl::enable();
    auto t0 = std::chrono::steady_clock::now();
    volatile double sink = f();
    auto t1 = std::chrono::steady_clock::now();
    perf_ctl::disable();
    (void)sink;
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("%-24s %8.1f ns/call   %7.2f GB/s   (%.2f s)\n",
           name, sec * 1e9 / ncalls, bytes / sec / 1e9, sec);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <base.csr> <queries.csr> [ncalls=2000000] "
                        "[variant=0..7] [maxrow=0[,maxrow...]] [reps=1] [alpha=1.0]\n", argv[0]);
        return 1;
    }
    uint64_t NCALL = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 2000000ULL;
    int only       = (argc > 4) ? atoi(argv[4]) : 0;
    int reps       = (argc > 6) ? atoi(argv[6]) : 1;
    float alpha    = (argc > 7) ? strtof(argv[7], nullptr) : 1.0f;

    // Working-set sizes to sweep, in order. 0 = the whole base matrix.
    std::vector<long> ws_list;
    {
        std::string spec = (argc > 5) ? argv[5] : "0";
        size_t pos = 0;
        while (pos <= spec.size()) {
            size_t comma = spec.find(',', pos);
            if (comma == std::string::npos) comma = spec.size();
            std::string tok = spec.substr(pos, comma - pos);
            pos = comma + 1;
            if (!tok.empty()) ws_list.push_back(atol(tok.c_str()));
        }
        if (ws_list.empty()) ws_list.push_back(0);
    }

    CSRMatrix* Dp = new CSRMatrix(argv[1], true);
    CSRMatrix Q(argv[2], true);
    printf("base nrow=%ld nnz=%ld avg_nnz=%.1f  data=%.0f MB\n",
           (long)Dp->nrow, (long)Dp->nnz, (double)Dp->nnz / Dp->nrow, Dp->nnz * 4.0 / 1e6);

    // Prune to match the index. Done before any timing, and the unpruned copy is
    // released immediately so it neither inflates the footprint nor pollutes the
    // caches the benchmark is about to measure.
    if (alpha < 1.0f) {
        const int64_t nnz_before = Dp->nnz;
        auto p0 = std::chrono::steady_clock::now();
        CSRMatrix* pruned = sparse_hnsw::pruneMatrixWithAlpha(Dp, alpha);
        delete Dp;
        Dp = pruned;
        auto p1 = std::chrono::steady_clock::now();
        printf("pruned alpha=%.3f in %.1f s: nnz %ld -> %ld (keep_frac=%.4f), "
               "avg_row %.0f -> %.0f B, data=%.0f MB\n",
               alpha, std::chrono::duration<double>(p1 - p0).count(),
               (long)nnz_before, (long)Dp->nnz, (double)Dp->nnz / nnz_before,
               (double)nnz_before / Dp->nrow * 4.0,
               (double)Dp->nnz / Dp->nrow * 4.0, Dp->nnz * 4.0 / 1e6);
    } else {
        printf("alpha=1.0: no pruning (NOTE: the real search streams PRUNED rows "
               "-- pass the index's alpha to compare)\n");
    }
    CSRMatrix& D = *Dp;

    // One representative query row (resident across all calls, like a real query).
    const uint32_t qid = 0;
    const IndiceDataPair* qv = Q.indices_data + Q.indptr[qid];
    const uint32_t qn = static_cast<uint32_t>(Q.indptr[qid + 1] - Q.indptr[qid]);

    // Scatter it, exactly as SearchScratch::scatterQuery does once per query.
    // This buffer is the dense kernel's second access stream: dim x 4 B, too
    // big for L1 (32 KB) and, on both target parts, still inside L2 --
    // 1 MB/core on Grace's Cascade Lake, 512 KB/core on Perlmutter's Zen3.
    // Report the actual L2 rather than a hardcoded one: which side of the L2
    // boundary q_dense lands on is exactly how V1 is meant to be read.
    const int dim = static_cast<int>(D.ncol);
    std::vector<float> q_dense(static_cast<size_t>(dim), 0.0f);
    for (uint32_t i = 0; i < qn; ++i) q_dense[qv[i].indice] = static_cast<float>(qv[i].data);

    sparse_hnsw::QuantCSR QC;
    QC.build(D);
    printf("quantized copy: %.0f MB (%.1f B/row) vs fp16 %.0f MB (%.1f B/row) "
           "-> %.3fx fewer row bytes\n",
           QC.blob.size() / 1e6, (double)QC.blob.size() / D.nrow,
           D.nnz * 4.0 / 1e6, (double)D.nnz * 4.0 / D.nrow,
           (D.nnz * 4.0 / D.nrow) / ((double)QC.blob.size() / D.nrow));
    long l1b = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    long l2b = sysconf(_SC_LEVEL2_CACHE_SIZE);
    printf("query nnz=%u  dim=%d  q_dense=%.0f KB (L1=%ld KB, L2=%ld KB)\n",
           qn, dim, dim * 4.0 / 1024,
           l1b > 0 ? l1b / 1024 : 32, l2b > 0 ? l2b / 1024 : 512);

    auto bytes_of = [&](const std::vector<uint32_t>& v, uint64_t n) {
        double b = 0;
        for (uint64_t i = 0; i < n; ++i) {
            uint32_t id = v[i % v.size()];
            b += (D.indptr[id + 1] - D.indptr[id]) * 4.0;
        }
        return b;
    };

    perf_ctl::init();
    const float* qd = q_dense.data();

    // Gather scratch for variant 5 (staged two-phase pipeline).
    uint32_t maxnnz = 0;
    for (int64_t r = 0; r < D.nrow; ++r) {
        uint32_t n = static_cast<uint32_t>(D.indptr[r + 1] - D.indptr[r]);
        if (n > maxnnz) maxnnz = n;
    }
    std::vector<IndiceDataPair> gbuf(static_cast<size_t>(HOPS_BATCH) * maxnnz);
    memset(gbuf.data(), 0, gbuf.size() * sizeof(IndiceDataPair));  // pre-fault pages

    for (long maxrow : ws_list) {
    // Fresh ids per working set; the seed is fixed so the sweep points differ
    // only in how far the random rows spread, not in which sequence is drawn.
    std::mt19937 rng(1234);
    long nr = (maxrow > 0 && maxrow < D.nrow) ? maxrow : D.nrow;
    std::uniform_int_distribution<uint32_t> pick(0, nr - 1);
    printf("working set: %ld rows = %.1f MB\n\n", nr, (D.indptr[nr] - D.indptr[0]) * 4.0 / 1e6);

    // Random doc ids, consumed in HOPS_BATCH-sized "neighbor lists".
    std::vector<uint32_t> ids(NCALL);
    for (auto& x : ids) x = pick(rng);

    // Hot set: 16 rows (~8 KB) -> resident in the 32 KiB L1d.
    std::vector<uint32_t> hot(16);
    for (auto& x : hot) x = pick(rng);

    double cold_bytes = bytes_of(ids, NCALL);
    double hot_bytes  = bytes_of(hot, NCALL);

    for (int rep = 0; rep < reps; ++rep) {
        if (reps > 1) printf("--- rep %d/%d ---\n", rep + 1, reps);

        // 1. HOT DENSE: full ALU work + the q_dense gather, zero DRAM pressure
        //    on the doc rows -> T_compute.
        if (!only || only == 1) timeit("1 HOT dense", NCALL, hot_bytes, [&] {
            double s = 0;
            for (uint64_t i = 0; i < NCALL; ++i) {
                uint32_t id = hot[i & 15];
                s += dense(qd, D.indices_data + D.indptr[id], D.indptr[id + 1] - D.indptr[id]);
            }
            return s;
        });

        // 2. COLD STREAM: same bytes, no ALU -> T_load at maximum MLP.
        if (!only || only == 2) timeit("2 COLD stream (no ALU)", NCALL, cold_bytes, [&] {
            uint64_t s = 0;
            for (uint64_t i = 0; i < NCALL; ++i) {
                uint32_t id = ids[i];
                s += stream_row(D.indices_data + D.indptr[id], D.indptr[id + 1] - D.indptr[id]);
            }
            return (double)s;
        });

        // 3. COLD DENSE: the production kernel -> T_total.
        if (!only || only == 3) timeit("3 COLD dense (no pf)", NCALL, cold_bytes, [&] {
            double s = 0;
            for (uint64_t i = 0; i < NCALL; ++i) {
                uint32_t id = ids[i];
                s += dense(qd, D.indices_data + D.indptr[id], D.indptr[id + 1] - D.indptr[id]);
            }
            return s;
        });

        // 4. COLD DENSE + batched prefetch, mimicking searchLayer's filtered loop.
        if (!only || only == 4) timeit("4 COLD dense (batch pf)", NCALL, cold_bytes, [&] {
            double s = 0;
            for (uint64_t b = 0; b + HOPS_BATCH <= NCALL; b += HOPS_BATCH) {
                for (int j = 0; j < HOPS_BATCH; ++j) {   // prefetch the whole "neighbor list"
                    const char* v = (const char*)(D.indices_data + D.indptr[ids[b + j]]);
                    __builtin_prefetch(v, 0, 2);
                    __builtin_prefetch(v + 64, 0, 2);
                    __builtin_prefetch(v + 128, 0, 2);
                    __builtin_prefetch(v + 192, 0, 2);
                    __builtin_prefetch(v + 256, 0, 2);
                    __builtin_prefetch(v + 320, 0, 2);
                    __builtin_prefetch(v + 384, 0, 2);
                    __builtin_prefetch(v + 448, 0, 2);
                }
                for (int j = 0; j < HOPS_BATCH; ++j) {
                    uint32_t id = ids[b + j];
                    s += dense(qd, D.indices_data + D.indptr[id], D.indptr[id + 1] - D.indptr[id]);
                }
            }
            return s;
        });

        // 5. COLD GATHER->DENSE: explicit two-phase staging.
        //    Phase 1 memcpys the batch's rows into a small resident scratch
        //    (copy addresses known upfront -> full MLP). Phase 2 runs the dense
        //    kernel out of that scratch.
        if (!only || only == 5) timeit("5 COLD gather+dense", NCALL, cold_bytes, [&] {
            double s = 0;
            std::vector<uint32_t> off(HOPS_BATCH + 1);
            for (uint64_t b = 0; b + HOPS_BATCH <= NCALL; b += HOPS_BATCH) {
                uint32_t o = 0;
                for (int j = 0; j < HOPS_BATCH; ++j) {           // gather phase
                    uint32_t id = ids[b + j];
                    uint32_t st = static_cast<uint32_t>(D.indptr[id]);
                    uint32_t n  = static_cast<uint32_t>(D.indptr[id + 1]) - st;
                    off[j] = o;
                    memcpy(gbuf.data() + o, D.indices_data + st, n * sizeof(IndiceDataPair));
                    o += n;
                }
                off[HOPS_BATCH] = o;
                for (int j = 0; j < HOPS_BATCH; ++j) {           // compute phase
                    s += dense(qd, gbuf.data() + off[j], off[j + 1] - off[j]);
                }
            }
            return s;
        });

        // 6. COLD MERGE: the pre-rewrite kernel on identical inputs. V6/V3 is
        //    the speedup the dense rewrite actually bought on this machine.
        if (!only || only == 6) timeit("6 COLD merge (old)", NCALL, cold_bytes, [&] {
            double s = 0;
            for (uint64_t i = 0; i < NCALL; ++i) {
                uint32_t id = ids[i];
                s += merge(qv, qn, D.indices_data + D.indptr[id], D.indptr[id + 1] - D.indptr[id]);
            }
            return s;
        });

        // 7. COLD QUANT: the CURRENT production kernel on the same rows, from
        //    the uint8 layout. V3/V7 is what the byte diet bought; note its
        //    bytes/call differ from the other variants by construction -- that
        //    IS the point, so its GB/s is reported against its own traffic.
        if (!only || only == 7) {
            double qbytes = 0;
            for (uint64_t i = 0; i < NCALL; ++i) qbytes += QC.rowBytes(ids[i]);
            timeit("7 COLD quant u8 (production)", NCALL, qbytes, [&] {
                double s = 0;
                for (uint64_t i = 0; i < NCALL; ++i) {
                    s += quant(q_dense.data(), QC.blob.data() + QC.row_off[ids[i]]);
                }
                return s;
            });
        }
    }

    printf("\navg bytes/call: cold=%.0f hot=%.0f  (v5 scratch max=%zu KB, ~%.0f KB avg/batch)\n\n",
           cold_bytes / NCALL, hot_bytes / NCALL,
           gbuf.size() * sizeof(IndiceDataPair) / 1024,
           HOPS_BATCH * (cold_bytes / NCALL) / 1024);
    fflush(stdout);
    }   // working-set sweep
    return 0;
}
