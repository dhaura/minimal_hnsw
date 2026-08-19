// E5 of the profiling ladder: the latency-vs-bandwidth verdict.
//
// For each thread count T, runs two kernels over random doc rows:
//   stream : touch the rows' bytes, no ALU     -> achievable bandwidth ceiling
//   dense  : the production distanceDense()    -> achieved throughput
//
// Each thread scatters the query into its OWN q_dense buffer, allocated inside
// an untimed parallel region so first-touch puts it on that thread's NUMA node.
// A single shared buffer would turn the gather stream into cross-socket traffic
// and make the scaling curve measure the interconnect instead of the kernel.
//
// Read-out:
//   per-core GB/s roughly FLAT as T grows        -> latency-bound  (fix: more MLP)
//   aggregate GB/s plateauing into stream ceiling-> bandwidth-bound (fix: fewer bytes)
//
// Usage:
//   bench_scale <base.csr> <queries.csr> [ncalls_per_thread=400000]
//               [threads=1,2,4,8,16,24,48] [alpha=1.0]
// (Grace is 2 x 24-core Xeon 6248R, so 24 = one full socket and 48 = both.)
//
// alpha applies mass-ratio pruning to the base matrix before timing and SHOULD
// MATCH THE INDEX'S alpha: the real search streams pruned rows (~215 B at
// alpha=0.8 vs ~507 B unpruned), and row length is exactly what sets both the
// bandwidth ceiling and the achieved rate this rung reports.
#include "csr_matrix.h"
#include "prune.h"
#include "perf_ctl.h"
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <omp.h>

__attribute__((always_inline))
static inline float dense(const float* qd, const IndiceDataPair* p, uint32_t pn) {
    float res = 0.0f;
    #pragma omp simd reduction(+:res)
    for (uint32_t i = 0; i < pn; ++i) {
        res += static_cast<float>(p[i].data) * qd[p[i].indice];
    }
    return 1.0f - res;
}

__attribute__((always_inline))
static inline uint64_t stream_row(const IndiceDataPair* p, uint32_t pn) {
    uint64_t acc = 0;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
    for (uint32_t i = 0; i < pn; ++i) acc += w[i];
    return acc;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <base.csr> <queries.csr> [ncalls_per_thread=400000] "
                        "[threads=1,2,4,8,16,24,48] [alpha=1.0]\n", argv[0]);
        return 1;
    }
    CSRMatrix* Dp = new CSRMatrix(argv[1], true);
    CSRMatrix Q(argv[2], true);
    uint64_t NC = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 400000ULL;
    float alpha = (argc > 5) ? strtof(argv[5], nullptr) : 1.0f;

    // Prune to match the index, before any timing; release the unpruned copy.
    if (alpha < 1.0f) {
        const int64_t nnz_before = Dp->nnz;
        CSRMatrix* pruned = sparse_hnsw::pruneMatrixWithAlpha(Dp, alpha);
        delete Dp;
        Dp = pruned;
        printf("pruned alpha=%.3f: nnz %ld -> %ld (keep_frac=%.4f), avg_row %.0f -> %.0f B\n",
               alpha, (long)nnz_before, (long)Dp->nnz, (double)Dp->nnz / nnz_before,
               (double)nnz_before / Dp->nrow * 4.0, (double)Dp->nnz / Dp->nrow * 4.0);
    } else {
        printf("alpha=1.0: no pruning (NOTE: the real search streams PRUNED rows)\n");
    }
    CSRMatrix& D = *Dp;

    std::vector<int> thread_list;
    {
        std::string spec = (argc > 4) ? argv[4] : "1,2,4,8,16,24,48";
        size_t pos = 0;
        while (pos < spec.size()) {
            size_t comma = spec.find(',', pos);
            if (comma == std::string::npos) comma = spec.size();
            thread_list.push_back(atoi(spec.substr(pos, comma - pos).c_str()));
            pos = comma + 1;
        }
    }

    const IndiceDataPair* qv = Q.indices_data + Q.indptr[0];
    const uint32_t qn = static_cast<uint32_t>(Q.indptr[1] - Q.indptr[0]);
    const int dim = static_cast<int>(D.ncol);
    printf("base nrow=%ld data=%.0f MB  query nnz=%u  dim=%d  q_dense=%.0f KB/thread  "
           "ncalls/thread=%llu\n\n",
           (long)D.nrow, D.nnz * 4.0 / 1e6, qn, dim, dim * 4.0 / 1024,
           (unsigned long long)NC);
    printf("%7s  %14s | %14s %14s %14s %14s\n",
           "threads", "stream GB/s", "dense ns/call", "dense Mcall/s", "dense GB/s", "GB/s/core");

    perf_ctl::init();

    for (int T : thread_list) {
        omp_set_num_threads(T);

        // Per-thread dense query, first-touched by its owner (untimed).
        std::vector<std::vector<float>> qbufs(T);
        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            qbufs[t].assign(static_cast<size_t>(dim), 0.0f);
            for (uint32_t i = 0; i < qn; ++i) {
                qbufs[t][qv[i].indice] = static_cast<float>(qv[i].data);
            }
        }

        // --- stream ceiling ---
        double stream_bytes = 0;
        volatile double sink = 0;
        auto s0 = std::chrono::steady_clock::now();
        #pragma omp parallel reduction(+:stream_bytes)
        {
            std::mt19937 rng(omp_get_thread_num() * 7919 + 1);
            std::uniform_int_distribution<uint32_t> pick(0, D.nrow - 1);
            uint64_t acc = 0; double b = 0;
            for (uint64_t i = 0; i < NC; ++i) {
                uint32_t id = pick(rng);
                int64_t st = D.indptr[id], en = D.indptr[id + 1];
                acc += stream_row(D.indices_data + st, static_cast<uint32_t>(en - st));
                b += (en - st) * 4.0;
            }
            stream_bytes += b;
            #pragma omp atomic
            sink += (double)acc;
        }
        auto s1 = std::chrono::steady_clock::now();
        double stream_sec = std::chrono::duration<double>(s1 - s0).count();

        // --- dense (production kernel) ---
        perf_ctl::enable();
        double dense_bytes = 0;
        auto m0 = std::chrono::steady_clock::now();
        #pragma omp parallel reduction(+:dense_bytes)
        {
            const float* qd = qbufs[omp_get_thread_num()].data();
            std::mt19937 rng(omp_get_thread_num() * 7919 + 1);
            std::uniform_int_distribution<uint32_t> pick(0, D.nrow - 1);
            double s = 0, b = 0;
            for (uint64_t i = 0; i < NC; ++i) {
                uint32_t id = pick(rng);
                int64_t st = D.indptr[id], en = D.indptr[id + 1];
                b += (en - st) * 4.0;
                s += dense(qd, D.indices_data + st, static_cast<uint32_t>(en - st));
            }
            dense_bytes += b;
            #pragma omp atomic
            sink += s;
        }
        auto m1 = std::chrono::steady_clock::now();
        perf_ctl::disable();
        double dense_sec = std::chrono::duration<double>(m1 - m0).count();

        uint64_t total = NC * (uint64_t)T;
        printf("%7d  %14.2f | %14.1f %14.2f %14.2f %14.3f\n",
               T,
               stream_bytes / stream_sec / 1e9,
               dense_sec * 1e9 / total,
               total / dense_sec / 1e6,
               dense_bytes / dense_sec / 1e9,
               dense_bytes / dense_sec / 1e9 / T);
        fflush(stdout);
    }
    return 0;
}
