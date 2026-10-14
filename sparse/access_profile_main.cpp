// This driver records, for every distance call in the walk, which node's row
// was fetched, then reports the reuse structure:
//
//   accesses vs unique nodes   the raw reuse factor
//   coverage curve             what share of all accesses (and of all BYTES)
//                              the N hottest rows account for, at cache sizes
//                              matched to real hardware
//   upper-layer share          the entry point and layers >0 are touched by
//                              every query by construction -- how much of the
//                              traffic is that, i.e. what a trivially small
//                              always-resident set already buys
//
// Bytes matter more than access counts: rows vary in length, and a cache is
// sized in bytes. A node hit often but cheap to fetch is worth less than a
// long row hit slightly less often.

#include "sparse_hnsw.h"
#include "csr_matrix.h"
#include "bench_common.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <omp.h>

using namespace sparse_hnsw;

int main(int argc, char* argv[]) {
    if (argc < 11) {
        std::cerr << "Usage: " << argv[0]
                  << " <M> <efC> <ef> <alpha> <beta> <patience>"
                     " <base> <queries> <gt> <out_prefix>"
                     " [quantize=1] [seed_top_k=8] [seed_spec=8:4]"
                  << std::endl;
        return 1;
    }
    const int M = std::stoi(argv[1]);
    const int efC = std::stoi(argv[2]);
    const int ef = std::stoi(argv[3]);
    const double alpha = std::stod(argv[4]);
    const int beta = std::stoi(argv[5]);
    const int patience = std::stoi(argv[6]);
    const std::string base_fp = argv[7], query_fp = argv[8], gt_fp = argv[9];
    const std::string out_prefix = argv[10];
    const bool quantize = (argc > 11) ? (std::stoi(argv[11]) != 0) : true;
    const int seed_top_k = (argc > 12) ? std::stoi(argv[12]) : 8;
    const std::string seed_spec = (argc > 13) ? argv[13] : "8:4";

    std::cout << "SPARSE_HNSW node-access profile\n===============================\n"
              << "M=" << M << " efC=" << efC << " ef=" << ef
              << " alpha=" << alpha << " beta=" << beta
              << " pat=" << patience << " threads=" << omp_get_max_threads() << "\n";

    CSRMatrix* full = new CSRMatrix(base_fp, true);
    const int dim = static_cast<int>(full->ncol);
    const int n = static_cast<int>(full->nrow);
    std::cout << "Loaded " << n << " x " << dim << ", nnz=" << full->nnz << "\n";

    SPARSE_HNSW index(dim, full, M, efC, n, true, false, false,
                      static_cast<float>(alpha), beta);
    CSRMatrix* pruned = nullptr;
    if (alpha < 1.0) {
        pruned = pruneMatrixWithAlpha(full, static_cast<float>(alpha));
        index.setSearchMatrix(pruned);
    }
    std::cout << "Building index...\n";
    index.addPointsBatch(n);
    if (quantize) index.enableQuantizedTraversal();
    if (seed_top_k > 0) {
        index.buildSeedTable(static_cast<uint32_t>(seed_top_k));
        const size_t colon = seed_spec.find(':');
        index.setSeedParams(std::stoi(seed_spec.substr(0, colon)),
                            std::stoi(seed_spec.substr(colon + 1)));
    }
    index.setPatience(patience);

    CSRMatrix* queries = new CSRMatrix(query_fp, true);
    const int nq = static_cast<int>(queries->nrow);
    std::vector<uint32_t> I; uint32_t n_gt = 0, k = 0;
    bench::get_gt(gt_fp, I, n_gt, k);

    index.enableAccessProfile();
    std::cout << "Running " << nq << " queries with access profiling...\n";
    std::vector<uint32_t> pred;
    index.searchKNNBatch(queries, nq, k, ef, pred);
    const double recall = bench::calculate_recall(pred, I, k, nq);
    std::cout << "  recall@" << k << " = " << recall * 100 << "%\n";

    // ---- reuse structure -------------------------------------------------
    const std::atomic<uint32_t>* hits = index.nodeHits();
    struct Node { uint32_t id; uint32_t hits; uint64_t bytes; };
    std::vector<Node> hot;
    hot.reserve(1u << 20);
    uint64_t total_acc = 0, total_bytes = 0;
    uint64_t upper_acc = 0, upper_bytes = 0;
    for (int i = 0; i < n; ++i) {
        const uint32_t h = hits[i].load(std::memory_order_relaxed);
        if (!h) continue;
        const uint64_t rb = index.rowBytesOf(static_cast<uint32_t>(i));
        hot.push_back({static_cast<uint32_t>(i), h, rb});
        total_acc += h;
        total_bytes += static_cast<uint64_t>(h) * rb;
        if (index.elementLevel(static_cast<uint32_t>(i)) > 0) {
            upper_acc += h;
            upper_bytes += static_cast<uint64_t>(h) * rb;
        }
    }

    std::sort(hot.begin(), hot.end(), [](const Node& a, const Node& b) {
        if (a.hits != b.hits) return a.hits > b.hits;
        return a.bytes > b.bytes;
    });

    std::cout << "\n=== reuse ===\n"
              << "  total row fetches   : " << total_acc << "\n"
              << "  distinct rows        : " << hot.size()
              << "  (" << std::fixed << std::setprecision(2)
              << 100.0 * hot.size() / n << "% of corpus)\n"
              << "  reuse factor         : " << std::setprecision(1)
              << (hot.empty() ? 0.0 : double(total_acc) / hot.size()) << "x\n"
              << "  fetches per query    : " << std::setprecision(0)
              << double(total_acc) / nq << "\n"
              << "  bytes pulled (no cache): " << std::setprecision(2)
              << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GiB\n";

    std::cout << "\n=== always-resident set already implied by the graph ===\n"
              << "  layers >0 nodes      : " << std::setprecision(4)
              << 100.0 * upper_acc / total_acc << "% of fetches, "
              << 100.0 * upper_bytes / total_bytes << "% of bytes\n";
    const uint32_t epid = index.entryPoint();
    if (epid < static_cast<uint32_t>(n)) {
        std::cout << "  global entry point   : node " << epid << ", "
                  << hits[epid].load(std::memory_order_relaxed) << " fetches ("
                  << std::setprecision(4)
                  << 100.0 * hits[epid].load(std::memory_order_relaxed) / total_acc
                  << "% of all fetches)\n";
    }

    // ---- coverage curve: cache sized in BYTES ----------------------------
    std::cout << "\n=== coverage: a cache holding the N hottest rows ===\n"
              << "  " << std::setw(12) << "cache" << std::setw(12) << "rows"
              << std::setw(12) << "% fetches" << std::setw(12) << "% bytes"
              << "   note\n";
    // Zen3 (EPYC 7713/7763) cache hierarchy, verified from
    // /sys/devices/system/cpu/cpu0/cache/index*/{size,shared_cpu_list}.
    // NB lscpu reports L2/L3 as AGGREGATES across instances -- "L2 cache:
    // 64 MiB (128 instances)" is 512 KiB per core, not 64 MiB, and "L3:
    // 512 MiB (16 instances)" is 32 MiB per CCD shared by 8 cores.
    const struct { const char* name; uint64_t bytes; const char* note; } sizes[] = {
        {"32 KiB",   32ull << 10,  "L1d per core"},
        {"512 KiB", 512ull << 10,  "L2 per core"},
        {"1 MiB",     1ull << 20,  "2x L2 per core"},
        {"32 MiB",   32ull << 20,  "L3 per CCD, shared by 8 cores"},
        {"256 MiB", 256ull << 20,  "all L3 on one socket (8 CCDs)"},
        {"512 MiB", 512ull << 20,  "all L3, both sockets"},
        {"1 GiB",     1ull << 30,  "60% of the quantized index"},
        {"4 GiB",     4ull << 30,  "exceeds the index"},
    };
    std::ofstream curve(out_prefix + "_coverage.csv");
    curve << "cache_bytes,rows,pct_fetches,pct_bytes\n";
    for (const auto& sz : sizes) {
        uint64_t occupied = 0, acc = 0, byt = 0, rows = 0;
        for (const Node& nd : hot) {
            if (occupied + nd.bytes > sz.bytes) break;
            occupied += nd.bytes; ++rows;
            acc += nd.hits; byt += static_cast<uint64_t>(nd.hits) * nd.bytes;
        }
        std::cout << "  " << std::setw(12) << sz.name << std::setw(12) << rows
                  << std::setw(11) << std::setprecision(2) << 100.0 * acc / total_acc << "%"
                  << std::setw(11) << 100.0 * byt / total_bytes << "%"
                  << "   " << sz.note << "\n";
        curve << sz.bytes << "," << rows << "," << 100.0 * acc / total_acc
              << "," << 100.0 * byt / total_bytes << "\n";
    }

    // full histogram for offline analysis (node id, hits, row bytes, level)
    std::ofstream hist(out_prefix + "_hist.csv");
    hist << "node,hits,row_bytes,level\n";
    const size_t dump = std::min<size_t>(hot.size(), 2000000);
    for (size_t i = 0; i < dump; ++i) {
        hist << hot[i].id << "," << hot[i].hits << "," << hot[i].bytes << ","
             << index.elementLevel(hot[i].id) << "\n";
    }
    std::cout << "\nwrote " << out_prefix << "_coverage.csv and "
              << out_prefix << "_hist.csv (" << dump << " rows)\n";
    return 0;
}
