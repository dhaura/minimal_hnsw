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
                     " <extend_candidates> <keep_pruned> <alpha> <beta>"
                     " <input_filepath> <query_filepath> <gt_filepath>"
                     " <results_csv_path> [model_name=SparseHNSW]"
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
    int beta = std::stoi(argv[8]);
    std::string input_filepath = argv[9];
    std::string query_filepath = argv[10];
    std::string gt_filepath = argv[11];
    std::string results_csv_path = argv[12];
    std::string model_name = (argc > 13) ? argv[13] : "SparseHNSW";

    if (ef_list.empty()) {
        std::cerr << "ef_list is empty." << std::endl;
        return 1;
    }

    int num_omp_threads = omp_get_max_threads();
    std::cout << "SPARSE_HNSW ef sweep\n====================\n"
              << "M=" << M << " ef_construction=" << ef_construction
              << " alpha=" << alpha << " beta=" << beta
              << " heuristic=" << use_heuristic
              << " threads=" << num_omp_threads << "\n";

    CSRMatrix *datamatrix = new CSRMatrix(input_filepath, true);
    int dim = datamatrix->ncol;
    int num_points = datamatrix->nrow;

    auto start_index_time = std::chrono::steady_clock::now();

    SPARSE_HNSW index(dim, datamatrix, M, ef_construction, num_points,
                      use_heuristic, extend_candidates, keep_pruned, alpha, beta);

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
    std::cout << "Added " << num_points << " points to the index in "
              << indexing_time_sec << " seconds.\n";

    index.printInfo();

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

    std::cout << "\nSweeping ef over " << ef_list.size() << " values ("
              << query_count << " queries, k=" << k << ")\n";

    // Warm-up: first pass pays page-cache and NUMA first-touch costs that
    // would otherwise be charged to whichever ef happens to run first.
    {
        std::vector<uint32_t> warm;
        index.searchKNNBatch(querymatrix, query_count, k, ef_list[0], warm);
    }

    for (int ef : ef_list) {
        if (ef < static_cast<int>(k)) {
            std::cerr << "  skipping ef=" << ef << " (< k=" << k << ")\n";
            continue;
        }

        std::vector<uint32_t> pred_labels;
        auto start_query = std::chrono::steady_clock::now();
        index.searchKNNBatch(querymatrix, query_count, k, ef, pred_labels);
        auto end_query = std::chrono::steady_clock::now();

        double searching_time_sec =
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_query - start_query).count() / 1e6;

        double recall = bench::calculate_recall(pred_labels, I, k, query_count);
        double rr = bench::calculate_rr(pred_labels, I, k, query_count);

        std::ostringstream params;
        params << "M=" << M << " efC=" << ef_construction << " ef=" << ef
               << " alpha=" << alpha << " beta=" << beta
               << " threads=" << num_omp_threads;

        bench::printPoint(params.str(), recall, searching_time_sec, query_count, rr);
        bench::appendRow(results_csv_path, model_name, recall, indexing_time_sec,
                         searching_time_sec, query_count, rr, params.str());
    }

    std::cout << "\nResults appended to " << results_csv_path << "\n";
    return 0;
}
