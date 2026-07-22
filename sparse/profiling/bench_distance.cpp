// E3 (ablation) + E4 (working-set sweep) of the profiling ladder.
//
// Isolates the memory cost of SPARSE_HNSW::distance() from its ALU cost:
// the same merge kernel and CSR layout are run under five access regimes.
// Because load and compute overlap inside the fused merge loop, "load time"
// and "compute time" are only defined counterfactually -- each variant
// removes one cost and keeps the other:
//
//   1 HOT merge         : full merge, doc rows L1-resident   -> T_compute
//   2 COLD stream       : touch the same bytes, no merge     -> T_load (max MLP)
//   3 COLD merge        : the production kernel              -> T_total
//   4 COLD merge + pf   : batched prefetch (as searchLayer)  -> async gather
//   5 COLD gather+merge : memcpy batch to scratch, then merge-> dense-style pipeline
//
// Read-out:
//   memory-attributable share = (V3 - V1) / V3
//   fusion penalty            = V3 - (V1 + V2)   (MLP destroyed by the merge's
//                                                 data-dependent pointer advance)
//   pipelining bound          = max(V1, V2)      (best case for perfect overlap)
//
// Usage:
//   bench_distance <base.csr> <queries.csr> [ncalls=2000000] [variant=0]
//                  [maxrow=0] [reps=1]
//     variant : 0 = all, 1..5 = single variant (use with perf_groups.sh so
//               counters are attributable to one regime)
//     maxrow  : restrict random ids to the first maxrow rows -> E4 sweep of
//               the working-set size (e.g. 4000 16000 64000 250000 1000000)
//     reps    : repeat everything; take the best rep on shared/noisy nodes
#include "csr_matrix.h"
#include "perf_ctl.h"
#include <chrono>
#include <random>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static const int HOPS_BATCH = 32;   // ~= level-0 degree (2*M with M=16)

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

// Pure streaming read of a row: no merge, high ILP. Isolates memory.
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
                        "[variant=0..5] [maxrow=0] [reps=1]\n", argv[0]);
        return 1;
    }
    uint64_t NCALL = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 2000000ULL;
    int only       = (argc > 4) ? atoi(argv[4]) : 0;
    long maxrow    = (argc > 5) ? atol(argv[5]) : 0;
    int reps       = (argc > 6) ? atoi(argv[6]) : 1;

    CSRMatrix D(argv[1], true);
    CSRMatrix Q(argv[2], true);
    printf("base nrow=%ld nnz=%ld avg_nnz=%.1f  data=%.0f MB\n",
           (long)D.nrow, (long)D.nnz, (double)D.nnz / D.nrow, D.nnz * 4.0 / 1e6);

    // One representative query row (resident across all calls, like a real query).
    const uint32_t qid = 0;
    const IndiceDataPair* qv = Q.indices_data + Q.indptr[qid];
    const uint32_t qn = Q.indptr[qid + 1] - Q.indptr[qid];
    printf("query nnz=%u\n", qn);

    std::mt19937 rng(1234);
    long nr = (maxrow > 0 && maxrow < D.nrow) ? maxrow : D.nrow;
    std::uniform_int_distribution<uint32_t> pick(0, nr - 1);
    printf("working set: %ld rows = %.1f MB\n\n", nr, (D.indptr[nr] - D.indptr[0]) * 4.0 / 1e6);

    // Random doc ids, grouped into HOPS_BATCH-sized "neighbor lists".
    std::vector<uint32_t> ids(NCALL);
    for (auto& x : ids) x = pick(rng);

    // Hot set: 16 rows (~8 KB) -> resident in the 32 KiB L1d.
    std::vector<uint32_t> hot(16);
    for (auto& x : hot) x = pick(rng);

    auto bytes_of = [&](const std::vector<uint32_t>& v, uint64_t n) {
        double b = 0;
        for (uint64_t i = 0; i < n; ++i) {
            uint32_t id = v[i % v.size()];
            b += (D.indptr[id + 1] - D.indptr[id]) * 4.0;
        }
        return b;
    };

    perf_ctl::init();
    double cold_bytes = bytes_of(ids, NCALL);
    double hot_bytes  = bytes_of(hot, NCALL);

    // Gather scratch for variant 5 (dense-style two-phase pipeline).
    uint32_t maxnnz = 0;
    for (int64_t r = 0; r < D.nrow; ++r) {
        uint32_t n = static_cast<uint32_t>(D.indptr[r + 1] - D.indptr[r]);
        if (n > maxnnz) maxnnz = n;
    }
    std::vector<IndiceDataPair> gbuf(static_cast<size_t>(HOPS_BATCH) * maxnnz);
    memset(gbuf.data(), 0, gbuf.size() * sizeof(IndiceDataPair));  // pre-fault pages

    for (int rep = 0; rep < reps; ++rep) {
        if (reps > 1) printf("--- rep %d/%d ---\n", rep + 1, reps);

        // 1. HOT MERGE: full ALU work, zero memory pressure -> T_compute.
        if (!only || only == 1) timeit("1 HOT merge", NCALL, hot_bytes, [&] {
            double s = 0;
            for (uint64_t i = 0; i < NCALL; ++i) {
                uint32_t id = hot[i & 15];
                s += merge(qv, qn, D.indices_data + D.indptr[id], D.indptr[id + 1] - D.indptr[id]);
            }
            return s;
        });

        // 2. COLD STREAM: same bytes, no merge -> T_load with max MLP
        //    (addresses known upfront, like the dense gather).
        if (!only || only == 2) timeit("2 COLD stream (no ALU)", NCALL, cold_bytes, [&] {
            uint64_t s = 0;
            for (uint64_t i = 0; i < NCALL; ++i) {
                uint32_t id = ids[i];
                s += stream_row(D.indices_data + D.indptr[id], D.indptr[id + 1] - D.indptr[id]);
            }
            return (double)s;
        });

        // 3. COLD MERGE: the production kernel -> T_total.
        if (!only || only == 3) timeit("3 COLD merge (no pf)", NCALL, cold_bytes, [&] {
            double s = 0;
            for (uint64_t i = 0; i < NCALL; ++i) {
                uint32_t id = ids[i];
                s += merge(qv, qn, D.indices_data + D.indptr[id], D.indptr[id + 1] - D.indptr[id]);
            }
            return s;
        });

        // 4. COLD MERGE + batched prefetch, mimicking searchLayer's filtered loop.
        if (!only || only == 4) timeit("4 COLD merge (batch pf)", NCALL, cold_bytes, [&] {
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
                    s += merge(qv, qn, D.indices_data + D.indptr[id], D.indptr[id + 1] - D.indptr[id]);
                }
            }
            return s;
        });

        // 5. COLD GATHER->MERGE: explicit two-phase, the dense-style pipeline.
        //    Phase 1 memcpys the batch's rows into a small resident scratch
        //    (copy addresses known upfront -> full MLP, like the dense gather).
        //    Phase 2 merges from the scratch (like gemv on the dense buffer).
        if (!only || only == 5) timeit("5 COLD gather+merge", NCALL, cold_bytes, [&] {
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
                    s += merge(qv, qn, gbuf.data() + off[j], off[j + 1] - off[j]);
                }
            }
            return s;
        });
    }

    printf("\navg bytes/call: cold=%.0f hot=%.0f  (v5 scratch max=%zu KB, ~%.0f KB avg/batch)\n",
           cold_bytes / NCALL, hot_bytes / NCALL,
           gbuf.size() * sizeof(IndiceDataPair) / 1024,
           HOPS_BATCH * (cold_bytes / NCALL) / 1024);
    return 0;
}
