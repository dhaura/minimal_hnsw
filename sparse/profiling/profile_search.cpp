// Profiling driver for the SPARSE_HNSW search phase.
// Build target: sparse_profile (compiles sparse_hnsw.cpp with -DSPARSE_HNSW_PROFILE).
//
// Mirrors sparse_hnsw_demo's configuration exactly -- same 8 leading arguments,
// same mass-ratio pruning (alpha) and same beta-refine -- so what is profiled is
// what is actually raced.
//
// Modes:
//   batch  (E1): run searchKNNBatch once with perf counting gated to the search
//          phase. Reports exact bytes and distance-call counts -> achieved GB/s
//          and ns per distance call. Run it under profiling/perf_groups.sh for
//          the E2 counter groups (where perf exists).
//   replay (E6): what share of search time does distanceDense() actually own?
//          Records each query's distance sequence, then replays it: the
//          traversal, heaps and visited set are reproduced exactly while no CSR
//          row is touched. T_full - T_replay = the kernel's true cost. Run this
//          BEFORE trusting any "ns per distance call" number -- batch mode's
//          ns_per_dist divides ALL search time by the distance count, which
//          silently assumes the answer.
//   repeat (E3b): run each query twice back-to-back on one thread with a reused
//          scratch. Pass 1 (cold) pays DRAM; pass 2 (warm) walks the identical
//          deterministic path with its footprint cache-resident.
//          (cold - warm) / cold = memory-attributable share of REAL search time.
//
// Several modes can share one index build -- and they should, because building
// msmarco_full takes ~8 minutes at 48 threads. Pass them comma-separated, each
// optionally carrying its own thread count and query count:
//
//     <mode>[@threads][:num_queries]
//     batch@48,batch@1,repeat@1:2000,replay@1:2000
//
// NOTE on scope: repeat and replay drive searchKNN directly, which is the graph
// traversal only. The beta>1 refine pass lives in searchKNNBatch, so it is
// visible in batch mode alone (reported as refine_ndist / refine_bytes).
//
// Usage:
//   sparse_profile <M> <ef_construction> <ef> <use_heuristic> <extend_candidates>
//                  <keep_pruned> <alpha> <beta>
//                  <base.csr> <queries.csr> <gt>
//                  [modes=batch] [num_queries=all] [gate=cold|warm|all] [quantize=0]
//                  [seed_top_k=0] [seed_terms=0] [seed_per_term=1] [patience=0]
//
// Env:
//   PROF_BUILD_THREADS  threads for index construction (default: OMP max).
//   OMP_NUM_THREADS     default threads for a mode that does not say @threads.
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

// One "mode[@threads][:nq]" token.
struct ModeSpec {
    std::string name;
    int threads;
    int nq;
};

static std::vector<ModeSpec> parse_modes(const std::string& spec, int def_threads, int def_nq)
{
    std::vector<ModeSpec> out;
    size_t pos = 0;
    while (pos <= spec.size()) {
        size_t comma = spec.find(',', pos);
        if (comma == std::string::npos) comma = spec.size();
        std::string tok = spec.substr(pos, comma - pos);
        pos = comma + 1;
        if (tok.empty()) continue;

        ModeSpec m{tok, def_threads, def_nq};
        size_t colon = m.name.find(':');
        if (colon != std::string::npos) {
            m.nq = std::stoi(m.name.substr(colon + 1));
            m.name = m.name.substr(0, colon);
        }
        size_t at = m.name.find('@');
        if (at != std::string::npos) {
            m.threads = std::stoi(m.name.substr(at + 1));
            m.name = m.name.substr(0, at);
        }
        out.push_back(m);
    }
    return out;
}

// ---------------------------------------------------------------- E1 --------
static void run_batch(const SPARSE_HNSW& index, CSRMatrix* querymatrix, int query_count,
                      uint32_t k, int beta, int ef, uint32_t* I, int threads)
{
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
    uint64_t gbytes = index.profGraphBytes();
    uint64_t rdist = index.profRefineNDist();
    uint64_t rbytes = index.profRefineBytes();
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
              << " GBps_per_thread=" << bytes / sec / 1e9 / threads << "\n";
    std::cout << "PROF graph_bytes=" << gbytes
              << " graph_frac_of_bytes=" << (double)gbytes / (bytes + gbytes) << "\n";
    // The beta-refine pass: exact re-scoring of k*beta candidates against the
    // UNPRUNED rows, using the same dense gather kernel as the traversal
    // (distanceDenseRefine -- its own symbol in this build, so perf can split
    // it out). Its ns/call is not separable from the total here, but its call
    // and byte counts show how much of the work it is.
    std::cout << "PROF refine_ndist=" << rdist
              << " refine_bytes=" << rbytes
              << " refine_frac_of_ndist="
              << (double)rdist / (ndist + rdist)
              << " refine_frac_of_bytes="
              << (double)rbytes / (bytes + rbytes) << "\n";
    // Bytes including the graph and the refine pass -- the honest total the
    // memory system actually had to move for this search.
    std::cout << "PROF total_bytes=" << (bytes + gbytes + rbytes)
              << " total_GBps=" << (bytes + gbytes + rbytes) / sec / 1e9 << "\n";
}

// --------------------------------------------------------------- E3b --------
static void run_repeat(const SPARSE_HNSW& index, CSRMatrix* querymatrix, int query_count,
                       int num_points, uint32_t k_hat, int ef, const std::string& gate)
{
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
        auto r1 = index.searchKNNProf(i, querymatrix, k_hat, ef, scratch);
        auto t1 = std::chrono::steady_clock::now();
        if (gate_cold && gate != "all") perf_ctl::disable();

        uint64_t nd1 = scratch.prof_ndist, by1 = scratch.prof_bytes;

        if (gate_warm && gate != "all") perf_ctl::enable();
        auto t2 = std::chrono::steady_clock::now();
        auto r2 = index.searchKNNProf(i, querymatrix, k_hat, ef, scratch);
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
    // Warm-pass validity: the per-query footprint must fit in the 36 MiB L3 of
    // one Xeon 6248R socket, else pass 2 was not actually warm.
    if (max_query_bytes > 28ull * 1024 * 1024)
        std::cout << "PROF WARNING: max query footprint exceeds ~28 MB; "
                     "warm pass polluted, mem_share is an underestimate\n";
}

// ---------------------------------------------------------------- E6 --------
static void run_replay(const SPARSE_HNSW& index, CSRMatrix* querymatrix, int query_count,
                       int num_points, uint32_t k_hat, int ef)
{
    // searchLayer is deterministic given the distance VALUES.
    //   sweep A (untimed)  record every query's distance sequence
    //   sweep B (timed)    normal search           -> T_full
    //   sweep C (timed)    replay the recording    -> T_overhead
    // Sweep C reproduces the identical traversal, heap operations and visited
    // set, but never touches a CSR row and never prefetches one.
    //   distance share = (T_full - T_overhead) / T_full
    SearchScratch scratch;
    scratch.prepare(num_points);

    // --- sweep A: record ---
    std::vector<float> flat;
    std::vector<size_t> offset(query_count + 1, 0);
    std::vector<float> rec;
    for (int i = 0; i < query_count; ++i) {
        rec.clear();
        scratch.record = &rec;
        volatile auto r = index.searchKNNProf(i, querymatrix, k_hat, ef, scratch);
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
        auto r = index.searchKNNProf(i, querymatrix, k_hat, ef, scratch);
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
        auto r = index.searchKNNProf(i, querymatrix, k_hat, ef, scratch);
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
    // Traffic split: prof_bytes counts only doc rows; the graph is extra and is
    // NOT attributable to the distance kernel.
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
    // The replay sweep still scatters/unscatters q_dense per query. That is
    // real per-query setup, correctly charged to overhead, not to the kernel.
    std::cout << "PROF note=replay_includes_query_scatter_in_overhead\n";
}

int main(int argc, char* argv[]) {
    if (argc < 12)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <M> <ef_construction> <ef> <use_heuristic> <extend_candidates>"
                     " <keep_pruned> <alpha> <beta>"
                     " <input_filepath> <query_filepath> <gt_filepath>"
                     " [modes=batch] [num_queries=all] [gate=cold|warm|all] [quantize=0]"
                     " [seed_top_k=0] [seed_terms=0] [seed_per_term=1] [patience=0]\n"
                     "   modes: comma-separated <mode>[@threads][:nq], e.g.\n"
                     "          batch@48,batch@1,repeat@1:2000,replay@1:2000\n";
        return 1;
    }

    int M = std::stoi(argv[1]);
    int ef_construction = std::stoi(argv[2]);
    int ef = std::stoi(argv[3]);
    bool use_heuristic = (std::stoi(argv[4]) != 0);
    bool extend_candidates = (std::stoi(argv[5]) != 0);
    bool keep_pruned = (std::stoi(argv[6]) != 0);
    double alpha = std::stod(argv[7]);
    int beta = std::stoi(argv[8]);
    std::string input_filepath = argv[9];
    std::string query_filepath = argv[10];
    std::string gt_filepath = argv[11];
    std::string modes_arg = (argc > 12) ? argv[12] : "batch";
    long nq_arg          = (argc > 13) ? std::stol(argv[13]) : 0;
    std::string gate     = (argc > 14) ? argv[14] : "cold";
    bool quantize        = (argc > 15) && (std::stoi(argv[15]) != 0);
    const int seed_top_k   = (argc > 16) ? std::stoi(argv[16]) : 0;
    const int seed_terms   = (argc > 17) ? std::stoi(argv[17]) : 0;
    const int seed_per_term= (argc > 18) ? std::stoi(argv[18]) : 1;
    const int patience     = (argc > 19) ? std::stoi(argv[19]) : 0;

    if (alpha > 1.0 || alpha <= 0.0) {
        std::cerr << "Invalid alpha " << alpha << "; expected (0, 1].\n";
        return 1;
    }

    perf_ctl::init();

    CSRMatrix *datamatrix = new CSRMatrix(input_filepath, true);
    CSRMatrix *querymatrix = new CSRMatrix(query_filepath, true);
    int dim = datamatrix->ncol;
    int num_points = datamatrix->nrow;
    int all_queries = querymatrix->nrow;
    if (nq_arg > 0 && nq_arg < all_queries) all_queries = static_cast<int>(nq_arg);

    uint32_t *I = nullptr;
    uint32_t gt_n, k;
    get_gt(gt_filepath, I, gt_n, k);

    int default_threads = omp_get_max_threads();
    int build_threads = default_threads;
    if (const char* bt = getenv("PROF_BUILD_THREADS")) build_threads = std::max(1, atoi(bt));

    std::vector<ModeSpec> modes = parse_modes(modes_arg, default_threads, all_queries);

    std::cout << "PROF config M=" << M << " efC=" << ef_construction << " ef=" << ef
              << " alpha=" << alpha << " beta=" << beta
              << " heuristic=" << use_heuristic
              << " k=" << k << " nq=" << all_queries
              << " build_threads=" << build_threads
              << " dim=" << dim << " npoints=" << num_points << "\n";
    std::cout << "PROF q_dense_KB=" << dim * 4.0 / 1024
              << " avg_row_bytes=" << (double)datamatrix->nnz / num_points * 4.0 << "\n";

    // ---- Index (same recipe as sparse_hnsw_demo) ----
    SPARSE_HNSW index(dim, datamatrix, M, ef_construction, num_points,
                      use_heuristic, extend_candidates, keep_pruned,
                      static_cast<float>(alpha), beta);

    CSRMatrix *pruned_datamatrix = nullptr;
    if (alpha < 1.0) {
        auto p0 = std::chrono::steady_clock::now();
        pruned_datamatrix = index.pruneMatrix(datamatrix);
        index.setPrunedDataMatrix(pruned_datamatrix);
        auto p1 = std::chrono::steady_clock::now();
        std::cout << "PROF prune_s=" << std::chrono::duration<double>(p1 - p0).count()
                  << " nnz_before=" << datamatrix->nnz
                  << " nnz_after=" << pruned_datamatrix->nnz
                  << " keep_frac=" << (double)pruned_datamatrix->nnz / datamatrix->nnz
                  << " avg_row_bytes_pruned="
                  << (double)pruned_datamatrix->nnz / num_points * 4.0 << "\n";
    } else {
        std::cout << "PROF prune_s=0 keep_frac=1 (alpha=1, no pruning)\n";
    }

    omp_set_num_threads(build_threads);
    auto b0 = std::chrono::steady_clock::now();
    index.addPointsBatch(num_points);
    auto b1 = std::chrono::steady_clock::now();
    std::cout << "PROF build_s=" << std::chrono::duration<double>(b1 - b0).count() << "\n";

    if (quantize) {
        auto q0 = std::chrono::steady_clock::now();
        index.enableQuantizedTraversal();
        auto q1 = std::chrono::steady_clock::now();
        const double fp16_row = (alpha < 1.0 ? (double)pruned_datamatrix->nnz
                                             : (double)datamatrix->nnz) / num_points * 4.0;
        const double u8_row = (double)(index.quantizedBytes() - (size_t)(num_points + 1) * 8)
                              / num_points;
        std::cout << "PROF quantize_s=" << std::chrono::duration<double>(q1 - q0).count()
                  << " quant_MiB=" << index.quantizedBytes() / (1024.0 * 1024.0)
                  << " fp16_row_bytes=" << fp16_row
                  << " u8_row_bytes=" << u8_row
                  << " row_byte_ratio=" << fp16_row / u8_row << "\n";
    }
    if (seed_top_k > 0 && seed_terms > 0) {
        auto s0 = std::chrono::steady_clock::now();
        index.buildSeedTable(static_cast<uint32_t>(seed_top_k));
        index.setSeedParams(seed_terms, seed_per_term);
        auto s1 = std::chrono::steady_clock::now();
        std::cout << "PROF seed_s=" << std::chrono::duration<double>(s1 - s0).count()
                  << " seed_KiB=" << index.seedTableBytes() / 1024.0
                  << " seed_top_k=" << seed_top_k
                  << " seed_terms=" << seed_terms
                  << " seed_per_term=" << seed_per_term << "\n";
    }
    index.setPatience(patience);
    std::cout << "PROF quantized=" << (index.quantized() ? 1 : 0)
              << " seeding=" << (index.seedingEnabled() ? 1 : 0)
              << " patience=" << index.patience() << "\n";
    std::cout.flush();

    // The traversal searches for k*beta candidates before the refine pass trims
    // them back to k; single-threaded modes must do the same to be comparable.
    const uint32_t k_hat = (alpha < 1.0) ? k * static_cast<uint32_t>(beta) : k;

    // ---- Run each requested mode against this one index ----
    for (const ModeSpec& m : modes) {
        int nq = std::min(m.nq, all_queries);
        omp_set_num_threads(m.threads);
        std::cout << "\nPROF mode=" << m.name
                  << " search_threads=" << m.threads
                  << " nq=" << nq << " k=" << k << " k_hat=" << k_hat
                  << " ef=" << ef << " alpha=" << alpha << " beta=" << beta << "\n";

        if (m.name == "batch") {
            run_batch(index, querymatrix, nq, k, beta, ef, I, m.threads);
        } else if (m.name == "repeat") {
            run_repeat(index, querymatrix, nq, num_points, k_hat, ef, gate);
        } else if (m.name == "replay") {
            run_replay(index, querymatrix, nq, num_points, k_hat, ef);
        } else {
            std::cerr << "unknown mode '" << m.name << "'\n";
            return 1;
        }
        std::cout.flush();
    }

    return 0;
}
