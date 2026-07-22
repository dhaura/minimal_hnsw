// Profiling driver for the SPARSE_HNSW search phase.
// Build target: sparse_profile (compiles sparse_hnsw.cpp with -DSPARSE_HNSW_PROFILE).
//
// Modes:
//   batch  (E0+E1): build index, run searchKNNBatch once with perf counting
//          gated to the search phase. Reports exact bytes and distance-call
//          counts -> achieved GB/s and ns per distance call. Run it under
//          profiling/perf_groups.sh for the E2 counter groups.
//   replay (E6)   : what share of search time does distance() actually own?
//          Records each query's distance sequence, then replays it: the
//          traversal, heaps and visited set are reproduced exactly while no CSR
//          row is touched. T_full - T_replay = distance()'s true cost. Run this
//          BEFORE trusting any "ns per distance call" number -- the batch mode's
//          ns_per_dist divides ALL search time by the distance count, which
//          silently assumes the answer.
//   repeat (E3b)  : run each query twice back-to-back on one thread with a
//          reused scratch. Pass 1 (cold) pays DRAM; pass 2 (warm) walks the
//          identical deterministic path with its footprint L3-resident.
//          (cold - warm) / cold = memory-attributable share of search time
//          on the REAL traversal (real hub locality, unlike the random-id
//          proxy in bench_distance).
//
// Usage:
//   sparse_profile <M> <ef_construction> <ef> <use_heuristic> <extend_candidates>
//                  <keep_pruned> <use_mkl> <mklThreshold>
//                  <base.csr> <queries.csr> <gt>
//                  [mode=batch|repeat] [num_queries=all] [gate=cold|warm|all]
//   (first 11 args identical to sparse_hnsw_demo; gate applies to repeat mode:
//
// Env:
//   PROF_BUILD_THREADS  threads for index construction (default: OMP max).
//   OMP_NUM_THREADS     threads for the batch search phase.
//   PERF_CTL_FIFO / PERF_ACK_FIFO  set by perf_groups.sh for gated counting.
#include "sparse_hnsw.h"
#include "csr_matrix.h"
#include "perf_ctl.h"
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <omp.h>

using namespace sparse_hnsw;

static void get_gt(const std::string gt_path, uint32_t *&I, uint32_t &n, uint32_t &d)
{
    std::ifstream infile(gt_path, std::ios::binary);
    if (infile.fail())
    {
        std::cerr << std::string("Failed to open file ") + gt_path;
        exit(1);
    }
    infile.read((char *)&n, sizeof(uint32_t));
    infile.read((char *)&d, sizeof(uint32_t));
    I = new uint32_t[n * d];
    infile.read((char *)I, n * d * sizeof(uint32_t));
    infile.close();
}

static float calculate_recall(const std::vector<uint32_t> &predicted_labels, uint32_t *I,
                              uint32_t k, uint32_t num_queries)
{
    long total_hits = 0;
    for (uint32_t i = 0; i < num_queries; ++i)
    {
        std::unordered_set<uint32_t> gt_neighbors(I + (size_t)i * k, I + (size_t)(i + 1) * k);
        for (uint32_t j = 0; j < k; ++j)
        {
            if (gt_neighbors.count(predicted_labels[(size_t)i * k + j]))
                total_hits++;
        }
    }
    return static_cast<float>(total_hits) / (static_cast<float>(num_queries) * k);
}

int main(int argc, char* argv[]) {
    if (argc < 12)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <M> <ef_construction> <ef> <use_heuristic> <extend_candidates>"
                     " <keep_pruned> <use_mkl> <mklThreshold>"
                     " <input_filepath> <query_filepath> <gt_filepath>"
                     " [mode=batch|repeat|replay] [num_queries=all]"
                     " [gate=cold|warm|all]\n";
        return 1;
    }

    int M = std::stoi(argv[1]);
    int ef_construction = std::stoi(argv[2]);
    int ef = std::stoi(argv[3]);
    bool use_heuristic = (std::stoi(argv[4]) != 0);
    bool extend_candidates = (std::stoi(argv[5]) != 0);
    bool keep_pruned = (std::stoi(argv[6]) != 0);
    bool use_mkl = (std::stoi(argv[7]) != 0);
    size_t mklThreshold = std::stoul(argv[8]);
    std::string input_filepath = argv[9];
    std::string query_filepath = argv[10];
    std::string gt_filepath = argv[11];
    std::string mode = (argc > 12) ? argv[12] : "batch";
    long nq_arg      = (argc > 13) ? std::stol(argv[13]) : 0;
    std::string gate = (argc > 14) ? argv[14] : "cold";

    perf_ctl::init();

    CSRMatrix *datamatrix = new CSRMatrix(input_filepath, true);
    CSRMatrix *querymatrix = new CSRMatrix(query_filepath, true);
    int dim = datamatrix->ncol;
    int num_points = datamatrix->nrow;
    int query_count = querymatrix->nrow;
    if (nq_arg > 0 && nq_arg < query_count) query_count = static_cast<int>(nq_arg);

    uint32_t *I = nullptr;
    uint32_t gt_n, k;
    get_gt(gt_filepath, I, gt_n, k);

    int search_threads = omp_get_max_threads();
    int build_threads = search_threads;
    if (const char* bt = getenv("PROF_BUILD_THREADS")) build_threads = std::max(1, atoi(bt));

    std::cout << "PROF mode=" << mode << " search_threads=" << search_threads
              << " build_threads=" << build_threads << " nq=" << query_count
              << " k=" << k << " ef=" << ef << "\n";

    // ---- Build ----
    SPARSE_HNSW index(dim, datamatrix, M, ef_construction, num_points,
                      use_heuristic, extend_candidates, keep_pruned, use_mkl, mklThreshold);
    omp_set_num_threads(build_threads);
    auto b0 = std::chrono::steady_clock::now();
    index.addPointsBatch(num_points);
    auto b1 = std::chrono::steady_clock::now();
    omp_set_num_threads(search_threads);
    std::cout << "PROF build_s=" << std::chrono::duration<double>(b1 - b0).count() << "\n";

    if (mode == "batch") {
        index.profReset();
        std::vector<uint32_t> pred_labels;

        perf_ctl::enable();
        auto t0 = std::chrono::steady_clock::now();
        index.searchKNNBatch(querymatrix, query_count, k, ef, pred_labels);
        auto t1 = std::chrono::steady_clock::now();
        perf_ctl::disable();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        uint64_t ndist = index.profNDist();
        uint64_t bytes = index.profBytes();
        float recall = calculate_recall(pred_labels, I, k, query_count) * 100.0f;

        std::cout << "PROF search_s=" << sec
                  << " qps=" << query_count / sec
                  << " recall=" << recall << "\n";
        std::cout << "PROF ndist=" << ndist
                  << " bytes=" << bytes
                  << " ndist_per_query=" << (double)ndist / query_count
                  << " bytes_per_query=" << (double)bytes / query_count << "\n";
        std::cout << "PROF ns_per_dist=" << sec * 1e9 / ndist
                  << " achieved_GBps=" << bytes / sec / 1e9
                  << " GBps_per_thread=" << bytes / sec / 1e9 / search_threads << "\n";
    }
    else if (mode == "repeat") {
        SearchScratch scratch;
        scratch.prepare(num_points);

        const bool gate_cold = (gate == "cold" || gate == "all");
        const bool gate_warm = (gate == "warm" || gate == "all");

        double cold_s = 0, warm_s = 0;
        uint64_t cold_ndist = 0, warm_ndist = 0, cold_bytes = 0;
        uint64_t max_query_bytes = 0;
        int path_mismatches = 0;
        volatile uint64_t sink = 0;   // keep the search results observable

        if (gate == "all") perf_ctl::enable();
        for (int i = 0; i < query_count; ++i) {
            uint64_t nd0 = scratch.prof_ndist, by0 = scratch.prof_bytes;

            if (gate_cold && gate != "all") perf_ctl::enable();
            auto t0 = std::chrono::steady_clock::now();
            auto r1 = index.searchKNNProf(i, querymatrix, k, ef, scratch);
            auto t1 = std::chrono::steady_clock::now();
            if (gate_cold && gate != "all") perf_ctl::disable();

            uint64_t nd1 = scratch.prof_ndist, by1 = scratch.prof_bytes;

            if (gate_warm && gate != "all") perf_ctl::enable();
            auto t2 = std::chrono::steady_clock::now();
            auto r2 = index.searchKNNProf(i, querymatrix, k, ef, scratch);
            auto t3 = std::chrono::steady_clock::now();
            if (gate_warm && gate != "all") perf_ctl::disable();

            uint64_t nd2 = scratch.prof_ndist;
            sink += r1.size() + r2.size();

            cold_s += std::chrono::duration<double>(t1 - t0).count();
            warm_s += std::chrono::duration<double>(t3 - t2).count();
            cold_ndist += nd1 - nd0;
            warm_ndist += nd2 - nd1;
            cold_bytes += by1 - by0;
            max_query_bytes = std::max(max_query_bytes, by1 - by0);
            // Deterministic search must expand the same nodes both passes.
            if (nd2 - nd1 != nd1 - nd0) path_mismatches++;
        }
        if (gate == "all") perf_ctl::disable();

        std::cout << "PROF cold_s=" << cold_s << " warm_s=" << warm_s
                  << " mem_share=" << (cold_s - warm_s) / cold_s << "\n";
        std::cout << "PROF cold_us_per_query=" << cold_s * 1e6 / query_count
                  << " warm_us_per_query=" << warm_s * 1e6 / query_count << "\n";
        std::cout << "PROF ndist=" << cold_ndist
                  << " cold_ns_per_dist=" << cold_s * 1e9 / cold_ndist
                  << " warm_ns_per_dist=" << warm_s * 1e9 / warm_ndist << "\n";
        std::cout << "PROF bytes_per_query_avg=" << (double)cold_bytes / query_count
                  << " bytes_per_query_max=" << max_query_bytes
                  << " path_mismatches=" << path_mismatches << "\n";
        // Warm-pass validity: the per-query footprint must fit in one CCX's
        // 32 MiB L3, else pass 2 is not actually warm.
        if (max_query_bytes > 24ull * 1024 * 1024)
            std::cout << "PROF WARNING: max query footprint exceeds ~24 MB; "
                         "warm pass polluted, mem_share is an underestimate\n";
    }
    else if (mode == "replay") {
        // E6
        //
        // searchLayer is deterministic given the distance VALUES.
        //   sweep A (untimed)  record every query's distance sequence
        //   sweep B (timed)    normal search           -> T_full
        //   sweep C (timed)    replay the recording    -> T_overhead
        // Sweep C reproduces the identical traversal, heap operations and
        // visited set, but never touches a CSR row and never prefetches one.
        //   distance share = (T_full - T_overhead) / T_full
        // Both timed sweeps run over the whole query set, so graph/heap cache
        // behaviour is comparable between them.
        SearchScratch scratch;
        scratch.prepare(num_points);

        // --- sweep A: record ---
        std::vector<float> flat;
        std::vector<size_t> offset(query_count + 1, 0);
        std::vector<float> rec;
        for (int i = 0; i < query_count; ++i) {
            rec.clear();
            scratch.record = &rec;
            volatile auto r = index.searchKNNProf(i, querymatrix, k, ef, scratch);
            (void)r;
            scratch.record = nullptr;
            flat.insert(flat.end(), rec.begin(), rec.end());
            offset[i + 1] = flat.size();
        }
        std::cout << "PROF recorded=" << flat.size() << "\n";

        volatile uint64_t sink = 0;

        // --- sweep B: full search ---
        uint64_t nd_a = scratch.prof_ndist;
        uint64_t by_a = scratch.prof_bytes;
        uint64_t gb_a = scratch.prof_graph_bytes;

        perf_ctl::enable();
        auto b0 = std::chrono::steady_clock::now();
        for (int i = 0; i < query_count; ++i) {
            auto r = index.searchKNNProf(i, querymatrix, k, ef, scratch);
            sink += r.size();
        }
        auto b1 = std::chrono::steady_clock::now();
        perf_ctl::disable();
        double full_s = std::chrono::duration<double>(b1 - b0).count();
        uint64_t ndist = scratch.prof_ndist - nd_a;
        uint64_t vec_bytes = scratch.prof_bytes - by_a;
        uint64_t graph_bytes = scratch.prof_graph_bytes - gb_a;

        // --- sweep C: replay ---
        uint64_t nd_before = scratch.prof_ndist;
        auto c0 = std::chrono::steady_clock::now();
        for (int i = 0; i < query_count; ++i) {
            scratch.replay = flat.data() + offset[i];
            scratch.replay_idx = 0;
            auto r = index.searchKNNProf(i, querymatrix, k, ef, scratch);
            sink += r.size();
        }
        auto c1 = std::chrono::steady_clock::now();
        scratch.replay = nullptr;
        double ovh_s = std::chrono::duration<double>(c1 - c0).count();
        uint64_t replay_ndist = scratch.prof_ndist - nd_before;

        double dist_s = full_s - ovh_s;
        double share = dist_s / full_s;

        std::cout << "PROF full_s=" << full_s << " overhead_s=" << ovh_s
                  << " distance_s=" << dist_s << "\n";
        std::cout << "PROF distance_share=" << share
                  << " overhead_share=" << (1.0 - share) << "\n";
        std::cout << "PROF ndist=" << ndist
                  << " ns_per_dist_true=" << dist_s * 1e9 / ndist
                  << " ns_per_dist_naive=" << full_s * 1e9 / ndist << "\n";
        // Traffic split: prof_bytes counts only doc rows; the graph is extra and
        // is NOT attributable to distance().
        std::cout << "PROF vec_bytes=" << vec_bytes
                  << " graph_bytes=" << graph_bytes
                  << " graph_frac_of_bytes="
                  << (double)graph_bytes / (vec_bytes + graph_bytes) << "\n";
        // Validity: the replay must have reproduced the traversal exactly.
        std::cout << "PROF replay_ndist=" << replay_ndist
                  << " ndist_match=" << (replay_ndist == ndist ? 1 : 0) << "\n";
        if (replay_ndist != ndist) {
            std::cout << "PROF WARNING: replay diverged from the recorded path; "
                         "distance_share is invalid\n";
        }
    }
    else {
        std::cerr << "unknown mode '" << mode << "'\n";
        return 1;
    }

    return 0;
}
