// E5 of the profiling ladder: the latency-vs-bandwidth verdict.
//
// For each thread count T, runs two kernels over random doc rows:
//   stream : touch the rows' bytes, no merge -> achievable bandwidth ceiling
//   merge  : the production distance kernel  -> achieved throughput
//
// Usage:
//   bench_scale <base.csr> <queries.csr> [ncalls_per_thread=400000]
//               [threads=1,2,4,8,16,32,64,128]
#include "csr_matrix.h"
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
                        "[threads=1,2,4,8,16,32,64,128]\n", argv[0]);
        return 1;
    }
    CSRMatrix D(argv[1], true);
    CSRMatrix Q(argv[2], true);
    uint64_t NC = (argc > 3) ? strtoull(argv[3], nullptr, 10) : 400000ULL;

    std::vector<int> thread_list;
    {
        std::string spec = (argc > 4) ? argv[4] : "1,2,4,8,16,32,64,128";
        size_t pos = 0;
        while (pos < spec.size()) {
            size_t comma = spec.find(',', pos);
            if (comma == std::string::npos) comma = spec.size();
            thread_list.push_back(atoi(spec.substr(pos, comma - pos).c_str()));
            pos = comma + 1;
        }
    }

    const IndiceDataPair* qv = Q.indices_data + Q.indptr[0];
    const uint32_t qn = Q.indptr[1] - Q.indptr[0];
    printf("base nrow=%ld data=%.0f MB  query nnz=%u  ncalls/thread=%llu\n\n",
           (long)D.nrow, D.nnz * 4.0 / 1e6, qn, (unsigned long long)NC);
    printf("%7s  %14s | %14s %14s %14s %14s\n",
           "threads", "stream GB/s", "merge ns/call", "merge Mcall/s", "merge GB/s", "GB/s/core");

    perf_ctl::init();

    for (int T : thread_list) {
        omp_set_num_threads(T);

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
                uint32_t st = D.indptr[id], en = D.indptr[id + 1];
                acc += stream_row(D.indices_data + st, en - st);
                b += (en - st) * 4.0;
            }
            stream_bytes += b;
            #pragma omp atomic
            sink += (double)acc;
        }
        auto s1 = std::chrono::steady_clock::now();
        double stream_sec = std::chrono::duration<double>(s1 - s0).count();

        // --- merge (production kernel) ---
        perf_ctl::enable();
        double merge_bytes = 0;
        auto m0 = std::chrono::steady_clock::now();
        #pragma omp parallel reduction(+:merge_bytes)
        {
            std::mt19937 rng(omp_get_thread_num() * 7919 + 1);
            std::uniform_int_distribution<uint32_t> pick(0, D.nrow - 1);
            double s = 0, b = 0;
            for (uint64_t i = 0; i < NC; ++i) {
                uint32_t id = pick(rng);
                uint32_t st = D.indptr[id], en = D.indptr[id + 1];
                b += (en - st) * 4.0;
                s += merge(qv, qn, D.indices_data + st, en - st);
            }
            merge_bytes += b;
            #pragma omp atomic
            sink += s;
        }
        auto m1 = std::chrono::steady_clock::now();
        perf_ctl::disable();
        double merge_sec = std::chrono::duration<double>(m1 - m0).count();

        uint64_t total = NC * (uint64_t)T;
        printf("%7d  %14.2f | %14.1f %14.2f %14.2f %14.3f\n",
               T,
               stream_bytes / stream_sec / 1e9,
               merge_sec * 1e9 / total,
               total / merge_sec / 1e6,
               merge_bytes / merge_sec / 1e9,
               merge_bytes / merge_sec / 1e9 / T);
    }
    return 0;
}
