// Builds one SPARSE_HNSW index and then sweeps ef over a list, recording a
// recall/throughput point per ef.

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

int main(int argc, char *argv[]) {
    if (argc < 13) {
        std::cerr << "Usage: " << argv[0]
                  << " <M> <ef_construction> <ef_list> <use_heuristic>"
                     " <extend_candidates> <keep_pruned> <alpha> <beta_list>"
                     " <input_filepath> <query_filepath> <gt_filepath>"
                     " <results_csv_path> [model_name=SparseHNSW]"
                     " [repeats=5] [warmup=1] [quantize=0]"
                     " [seed_top_k=0] [seed_spec=off] [patience_list=0]"
                  << std::endl;
        return 1;
    }

    int M = std::stoi(argv[1]);
    int ef_construction = std::stoi(argv[2]);
    std::vector<int> ef_list = bench::parseList<int>(argv[3], bench::toInt);
    bool use_heuristic = (std::stoi(argv[4]) != 0);
    bool extend_candidates = (std::stoi(argv[5]) != 0);
    bool keep_pruned = (std::stoi(argv[6]) != 0);
    double alpha = std::stod(argv[7]);
    std::vector<int> beta_list = bench::parseList<int>(argv[8], bench::toInt);
    std::string input_filepath = argv[9];
    std::string query_filepath = argv[10];
    std::string gt_filepath = argv[11];
    std::string results_csv_path = argv[12];
    std::string model_name = (argc > 13) ? argv[13] : "SparseHNSW";
    int repeats = (argc > 14) ? std::stoi(argv[14]) : 5;
    int warmup = (argc > 15) ? std::stoi(argv[15]) : 1;
    bool quantize = (argc > 16) && (std::stoi(argv[16]) != 0);
    const int seed_top_k = (argc > 17) ? std::stoi(argv[17]) : 0;
    const std::string seed_spec = (argc > 18) ? argv[18] : "off";
    const std::vector<int> patience_list =
        bench::parseList<int>((argc > 19) ? argv[19] : "0", bench::toInt);

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

    if (ef_list.empty()) {
        std::cerr << "ef_list is empty." << std::endl;
        return 1;
    }
    if (beta_list.empty()) {
        std::cerr << "beta_list is empty." << std::endl;
        return 1;
    }

    std::ostringstream beta_str;
    for (size_t i = 0; i < beta_list.size(); ++i) {
        beta_str << (i ? "," : "") << beta_list[i];
    }

    int num_omp_threads = omp_get_max_threads();
    std::cout << "SPARSE_HNSW ef sweep\n====================\n"
              << "M=" << M << " ef_construction=" << ef_construction
              << " alpha=" << alpha << " beta=" << beta_str.str()
              << " heuristic=" << use_heuristic
              << " threads=" << num_omp_threads << "\n";

    auto start_load_time = std::chrono::steady_clock::now();
    CSRMatrix *datamatrix = new CSRMatrix(input_filepath, true);
    auto end_load_time = std::chrono::steady_clock::now();
    double load_time_sec =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_load_time - start_load_time).count() / 1e6;
    std::cout << "Loaded base vectors in " << load_time_sec << " seconds.\n";

    int dim = datamatrix->ncol;
    int num_points = datamatrix->nrow;

    auto start_index_time = std::chrono::steady_clock::now();

    SPARSE_HNSW index(dim, datamatrix, M, ef_construction, num_points,
                      use_heuristic, extend_candidates, keep_pruned, alpha,
                      beta_list.front());

    CSRMatrix *pruned_datamatrix = nullptr;
    if (alpha < 1.0) {
        std::cout << "Pruning dataset with alpha = " << alpha << "...\n";
        auto start_prune = std::chrono::steady_clock::now();
        pruned_datamatrix = index.pruneMatrix(datamatrix);
        index.setPrunedDataMatrix(pruned_datamatrix);
        auto end_prune = std::chrono::steady_clock::now();
        std::cout << "Pruning completed in "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         end_prune - start_prune).count() / 1e6
                  << " seconds\n";
    } else if (alpha > 1.0) {
        std::cerr << "Invalid alpha " << alpha << "; expected (0, 1].\n";
        return 1;
    }

    std::cout << "Adding points to the index...\n";
    index.addPointsBatch(num_points);

    auto end_index_time = std::chrono::steady_clock::now();
    // Includes pruning: it is part of what you pay to get a searchable index,
    // which is what the "Indexing Time" column means for the other methods.
    double indexing_time_sec =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_index_time - start_index_time).count() / 1e6;
    std::cout << "Built a searchable index over " << num_points
              << " points in " << indexing_time_sec
              << " seconds (pruning included).\n";

    index.printInfo();

    if (quantize) {
        std::cout << "Building uint8-quantized traversal copy...\n";
        auto start_q = std::chrono::steady_clock::now();
        index.enableQuantizedTraversal();
        auto end_q = std::chrono::steady_clock::now();
        std::cout << "Quantization completed in "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         end_q - start_q).count() / 1e6
                  << " seconds (" << index.quantizedBytes() / (1024.0 * 1024.0)
                  << " MiB)\n";
    }

    if (seed_top_k > 0) {
        std::cout << "Building inverted seed table (top-" << seed_top_k << " per column)...\n";
        auto start_seed = std::chrono::steady_clock::now();
        index.buildSeedTable(static_cast<uint32_t>(seed_top_k));
        auto end_seed = std::chrono::steady_clock::now();
        std::cout << "Seed table built in "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         end_seed - start_seed).count() / 1e6
                  << " s (" << index.seedTableBytes() / 1024.0 << " KiB)\n";
    }

    CSRMatrix *querymatrix = new CSRMatrix(query_filepath, true);
    int query_count = querymatrix->nrow;

    std::vector<uint32_t> I;
    uint32_t n_gt, k;
    bench::get_gt(gt_filepath, I, n_gt, k);
    if (static_cast<int>(n_gt) != query_count) {
        std::cerr << "Ground truth has " << n_gt << " rows but query file has "
                  << query_count << ".\n";
        return 1;
    }

    std::cout << "\nSweeping ef over " << ef_list.size() << " values x beta over "
              << beta_list.size() << " values (" << query_count
              << " queries, k=" << k << ")\n";

    std::cout << "  " << repeats << " timed passes per point after " << warmup
              << " warm-up pass(es); the median is reported.\n";

    const bool multi_beta = beta_list.size() > 1;

    const bool multi_seed = seed_cfgs.size() > 1;

    const bool multi_pat = patience_list.size() > 1;

    for (int patience : patience_list) {
      index.setPatience(patience);
      for (const auto& sc : seed_cfgs) {
      index.setSeedParams(sc.first, sc.second);
      for (int beta : beta_list) {
        index.setBeta(beta);
        // Only the search side changes, so no rebuild -- see setBeta().
        std::string row_model = multi_beta
                ? model_name + "_b" + std::to_string(beta)
                : model_name;
        if (multi_seed) {
            row_model += "_s" + std::to_string(sc.first) + "x" + std::to_string(sc.second);
        }
        if (multi_pat) {
            row_model += "_p" + std::to_string(patience);
        }

        for (int ef : ef_list) {
            if (ef < static_cast<int>(k)) {
                std::cerr << "  skipping ef=" << ef << " (< k=" << k << ")\n";
                continue;
            }

            std::vector<uint32_t> pred_labels;
            std::vector<double> times = bench::timedRuns(
                [&] { index.searchKNNBatch(querymatrix, query_count, k, ef, pred_labels); },
                repeats, warmup);

            double recall = bench::calculate_recall(pred_labels, I, k, query_count);
            double rr = bench::calculate_rr(pred_labels, I, k, query_count);

            std::ostringstream params;
            params << "M=" << M << " efC=" << ef_construction << " ef=" << ef
                   << " alpha=" << alpha << " beta=" << beta;
            if (quantize) {
                params << " q=u8";
            }
            if (index.seedingEnabled()) {
                params << " seedT=" << seed_top_k
                       << " seedH=" << sc.first << " seedS=" << sc.second;
            }
            if (patience > 0) {
                params << " pat=" << patience;
            }

            bench::printPoint(params.str(), recall, bench::median(times), query_count, rr);
            bench::appendRow(results_csv_path, row_model, params.str(), num_omp_threads,
                             recall, rr, indexing_time_sec, load_time_sec, 0.0,
                             times, query_count, k);
        }
      }
      }
    }

    std::cout << "\nResults appended to " << results_csv_path << "\n";
    return 0;
}
