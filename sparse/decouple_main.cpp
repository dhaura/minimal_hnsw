// Search-side alpha sweep: the graph is built once, on a matrix pruned at
// build_alpha, and the matrix the SEARCH scores against is then swapped per
// alpha over that one build.
//
// Usage:
//   sparse_decouple_sweep <M> <efC> <ef_list> <use_heuristic>
//                         <build_alpha in (0,1)> <search_alpha_list> <beta_list>
//                         <base> <queries> <gt> <results_csv>
//                         [model=SparseHNSW] [repeats=3] [warmup=1]
//                         [quantize=0] [seed_top_k=0] [seed_spec=off]
//                         [patience_list=0]

#include "sparse_hnsw.h"
#include "csr_matrix.h"
#include "bench_common.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>

using namespace sparse_hnsw;

namespace {

std::string trimZeros(double v) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2) << v;
    std::string s = os.str();
    while (s.size() > 1 && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
    return s;
}

double secondsSince(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - t0).count() / 1e6;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 12) {
        std::cerr << "Usage: " << argv[0]
                  << " <M> <ef_construction> <ef_list> <use_heuristic>"
                     " <build_alpha> <search_alpha_list> <beta_list>"
                     " <input_filepath> <query_filepath> <gt_filepath>"
                     " <results_csv_path> [model_name=SparseHNSW]"
                     " [repeats=3] [warmup=1] [quantize=0]"
                     " [seed_top_k=0] [seed_spec=off] [patience_list=0]"
                  << std::endl;
        return 1;
    }

    const int M = std::stoi(argv[1]);
    const int ef_construction = std::stoi(argv[2]);
    const std::vector<int> ef_list = bench::parseList<int>(argv[3], bench::toInt);
    const bool use_heuristic = (std::stoi(argv[4]) != 0);
    const double build_alpha = std::stod(argv[5]);
    const std::vector<float> search_alphas = bench::parseList<float>(argv[6], bench::toFloat);
    const std::vector<int> beta_list = bench::parseList<int>(argv[7], bench::toInt);
    const std::string input_filepath = argv[8];
    const std::string query_filepath = argv[9];
    const std::string gt_filepath = argv[10];
    const std::string results_csv_path = argv[11];
    const std::string model_name = (argc > 12) ? argv[12] : "SparseHNSW";
    const int repeats = (argc > 13) ? std::stoi(argv[13]) : 3;
    const int warmup = (argc > 14) ? std::stoi(argv[14]) : 1;
    const bool quantize = (argc > 15) && (std::stoi(argv[15]) != 0);
    const int seed_top_k = (argc > 16) ? std::stoi(argv[16]) : 0;
    const std::string seed_spec = (argc > 17) ? argv[17] : "off";
    const std::vector<int> patience_list =
        bench::parseList<int>((argc > 18) ? argv[18] : "0", bench::toInt);

    if (ef_list.empty() || beta_list.empty() || search_alphas.empty() ||
        patience_list.empty()) {
        std::cerr << "ef_list / search_alpha_list / beta_list / patience_list "
                     "must all be non-empty.\n";
        return 1;
    }
    if (build_alpha <= 0.0 || build_alpha >= 1.0) {
        std::cerr << "build_alpha " << build_alpha << " outside (0, 1) -- the "
                     "graph is always built on a pruned matrix (see the NOTE at "
                     "the top of this file).\n";
        return 1;
    }
    for (float a : search_alphas) {
        if (a <= 0.0f || a > 1.0f) {
            std::cerr << "search alpha " << a << " outside (0, 1].\n";
            return 1;
        }
        if (a >= 1.0f) {
            std::cout << "NOTE: search alpha=1 searches the unpruned matrix and "
                         "DISABLES the beta-refine; those rows are identical for "
                         "every beta.\n";
        }
    }

    std::vector<std::pair<int, int>> seed_cfgs;
    if (seed_spec == "off" || seed_top_k <= 0) {
        seed_cfgs.push_back({0, 0});
    } else {
        std::stringstream ss(seed_spec);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            const size_t colon = tok.find(':');
            if (colon == std::string::npos) {
                std::cerr << "seed_spec entry '" << tok << "' must be <terms>:<per_term>\n";
                return 1;
            }
            seed_cfgs.push_back({std::stoi(tok.substr(0, colon)),
                                 std::stoi(tok.substr(colon + 1))});
        }
    }

    const int num_omp_threads = omp_get_max_threads();
    std::cout << "SPARSE_HNSW decoupled build/search sweep\n"
                 "========================================\n"
              << "M=" << M << " efC=" << ef_construction
              << " build_alpha=" << build_alpha
              << " heuristic=" << use_heuristic
              << " threads=" << num_omp_threads << "\n";

    auto t_load = std::chrono::steady_clock::now();
    CSRMatrix* full = new CSRMatrix(input_filepath, true);
    const double load_time_sec = secondsSince(t_load);
    std::cout << "Loaded base vectors in " << load_time_sec << " s ("
              << full->nrow << " x " << full->ncol << ", nnz=" << full->nnz
              << ", " << static_cast<double>(full->nnz) / full->nrow
              << " nnz/row)\n";

    const int dim = static_cast<int>(full->ncol);
    const int num_points = static_cast<int>(full->nrow);

    auto t_index = std::chrono::steady_clock::now();
    SPARSE_HNSW index(dim, full, M, ef_construction, num_points,
                      use_heuristic, false, false,
                      static_cast<float>(build_alpha), beta_list.front());

    std::cout << "Pruning the BUILD matrix with alpha=" << build_alpha << " ...\n";
    CSRMatrix* build_matrix = pruneMatrixWithAlpha(full, static_cast<float>(build_alpha));
    index.setSearchMatrix(build_matrix);
    std::cout << "  nnz " << full->nnz << " -> " << build_matrix->nnz
              << " (keep=" << static_cast<double>(build_matrix->nnz) / full->nnz << ")\n";

    std::cout << "Adding points to the index...\n";
    index.addPointsBatch(num_points);
    const double indexing_time_sec = secondsSince(t_index);
    std::cout << "Built a searchable index over " << num_points << " points in "
              << indexing_time_sec << " s (build-side pruning included).\n";
    index.printInfo();

    if (seed_top_k > 0) {
        auto t_seed = std::chrono::steady_clock::now();
        index.buildSeedTable(static_cast<uint32_t>(seed_top_k));
        std::cout << "Seed table (top-" << seed_top_k << " per column) built in "
                  << secondsSince(t_seed) << " s ("
                  << index.seedTableBytes() / 1024.0 << " KiB)\n";
    }

    index.setSearchMatrix(full);
    delete build_matrix;

    CSRMatrix* querymatrix = new CSRMatrix(query_filepath, true);
    const int query_count = static_cast<int>(querymatrix->nrow);

    std::vector<uint32_t> I;
    uint32_t n_gt = 0, k = 0;
    bench::get_gt(gt_filepath, I, n_gt, k);
    if (static_cast<int>(n_gt) != query_count) {
        std::cerr << "Ground truth has " << n_gt << " rows but query file has "
                  << query_count << ".\n";
        return 1;
    }

    std::cout << "\nSweeping " << search_alphas.size() << " search alpha(s) x "
              << patience_list.size()
              << " patience x " << seed_cfgs.size() << " seed cfg(s) x "
              << beta_list.size() << " beta x " << ef_list.size() << " ef ("
              << query_count << " queries, k=" << k << ")\n"
              << "  " << repeats << " timed passes per point after " << warmup
              << " warm-up pass(es); the median is reported.\n";

    CSRMatrix* search_matrix = nullptr;

    for (const float alpha : search_alphas) {
        index.setSearchMatrix(full);
        delete search_matrix;
        search_matrix = nullptr;

        double prune_sec = 0.0;
        if (alpha < 1.0f) {
            auto t_prune = std::chrono::steady_clock::now();
            search_matrix = pruneMatrixWithAlpha(full, alpha);
            prune_sec = secondsSince(t_prune);
            index.setSearchMatrix(search_matrix);
        }
        index.setAlpha(alpha);

        const CSRMatrix* active = search_matrix ? search_matrix : full;
        std::cout << "\n--- search alpha=" << alpha << " | nnz " << full->nnz
                  << " -> " << active->nnz << " (keep="
                  << std::setprecision(4) << static_cast<double>(active->nnz) / full->nnz
                  << ", " << static_cast<double>(active->nnz) / active->nrow
                  << " nnz/row) | prune " << prune_sec << " s";

        if (quantize) {
            auto t_q = std::chrono::steady_clock::now();
            index.enableQuantizedTraversal();
            std::cout << " | quant " << secondsSince(t_q) << " s ("
                      << index.quantizedBytes() / (1024.0 * 1024.0) << " MiB)";
        }
        std::cout << " ---\n";

        for (const int patience : patience_list) {
            index.setPatience(patience);
            for (const auto& sc : seed_cfgs) {
                index.setSeedParams(sc.first, sc.second);
                for (const int beta : beta_list) {
                    index.setBeta(beta);
                    for (const int ef : ef_list) {
                        if (ef < static_cast<int>(k)) {
                            std::cerr << "  skipping ef=" << ef << " (< k=" << k << ")\n";
                            continue;
                        }

                        std::vector<uint32_t> pred_labels;
                        std::vector<double> times = bench::timedRuns(
                            [&] {
                                index.searchKNNBatch(querymatrix, query_count, k, ef, pred_labels);
                            },
                            repeats, warmup);

                        const double recall = bench::calculate_recall(pred_labels, I, k, query_count);
                        const double rr = bench::calculate_rr(pred_labels, I, k, query_count);

                        std::ostringstream params;
                        params << "M=" << M << " efC=" << ef_construction
                               << " ef=" << ef
                               << " buildAlpha=" << build_alpha
                               << " alpha=" << alpha
                               << " beta=" << beta;
                        if (quantize) params << " q=u8";
                        if (index.seedingEnabled()) {
                            params << " seedT=" << seed_top_k
                                   << " seedH=" << sc.first
                                   << " seedS=" << sc.second;
                        }
                        if (patience > 0) params << " pat=" << patience;

                        std::string row_model = model_name +
                            "_bA" + trimZeros(build_alpha) +
                            "_a" + trimZeros(alpha);
                        if (beta_list.size() > 1) row_model += "_b" + std::to_string(beta);
                        if (patience_list.size() > 1) row_model += "_p" + std::to_string(patience);
                        if (seed_cfgs.size() > 1) {
                            row_model += "_s" + std::to_string(sc.first) + "x" +
                                         std::to_string(sc.second);
                        }

                        bench::printPoint(params.str(), recall, bench::median(times),
                                          query_count, rr);
                        bench::appendRow(results_csv_path, row_model, params.str(),
                                         num_omp_threads, recall, rr,
                                         indexing_time_sec, load_time_sec, prune_sec,
                                         times, query_count, k);
                    }
                }
            }
        }
    }

    index.setSearchMatrix(full);
    delete search_matrix;
    delete querymatrix;
    delete full;

    std::cout << "\nResults appended to " << results_csv_path << "\n";
    return 0;
}
