#include "sparse_hnsw.h"
#include "csr_matrix.h"
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <omp.h>

using namespace sparse_hnsw;

void appendResultsRow(const std::string &csv_path, double alpha, int beta,
                       int64_t dataset_size, int threads, double pruning_time_sec,
                       double indexing_time_sec, double searching_time_sec, float recall) {
    bool write_header = true;
    {
        std::ifstream check(csv_path);
        write_header = !(check.good() && check.peek() != std::ifstream::traits_type::eof());
    }

    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) {
        std::cerr << "Failed to open results CSV '" << csv_path << "' for writing.\n";
        return;
    }

    if (write_header) {
        csv << "alpha,beta,dataset_size,threads,pruning_time_sec,indexing_time_sec,searching_time_sec,recall\n";
    }
    csv << alpha << "," << beta << "," << dataset_size << "," << threads << ","
        << std::fixed << std::setprecision(6)
        << pruning_time_sec << "," << indexing_time_sec << "," << searching_time_sec << ","
        << std::setprecision(4) << recall << "\n";
}

void get_gt(const std::string gt_path, uint32_t *&I, uint32_t &n, uint32_t &d)
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

float calculate_recall(const std::vector<uint32_t> &predicted_labels, uint32_t *&I, uint32_t k, uint32_t num_queries)
{

    int total_hits = 0;
    for (int i = 0; i < num_queries; ++i)
    {
        int query_label = i;
        std::vector<uint32_t> pred_ids = std::vector<uint32_t>(predicted_labels.begin() + i * k, predicted_labels.begin() + (i + 1) * k);
        std::unordered_set<uint32_t> gt_neighbors(I + query_label * k, I + (query_label + 1) * k);

        int hit_count = 0;
        for (int pid : pred_ids)
        {
            if (gt_neighbors.count(pid))
                hit_count++;
        }

        total_hits += hit_count;

        // if (i == 0)
        // {
        //     // print results and gt.
        //     std::cout << "Query label: " << query_label << "\n";
        //     std::cout << "Predicted IDs: ";
        //     for (int pid : pred_ids)
        //     {
        //         std::cout << pid << " ";
        //     }
        //     std::cout << "\nGround Truth IDs: ";
        //     for (int gt_pid : gt_neighbors)
        //     {
        //         std::cout << gt_pid << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }

    return static_cast<double>(total_hits) / (static_cast<double>(num_queries) * k);
}


int main(int argc, char* argv[]) {
    std::cout << "Minimal SPARSE_HNSW Demo\n";
    std::cout << "=================\n\n";

    if (argc < 13)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <ef_construction> <ef> <use_heuristic> <extend_candidates> <keep_pruned> <alpha> <beta> <input_filepath> <query_filepath> <gt_filepath> <results_csv_path>" << std::endl;
        return 1;
    }

    // Parse command line arguments into variables.
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
    std::string results_csv_path = argv[12];

    int num_omp_threads = omp_get_max_threads();
    std::cout << "Number of OpenMP threads: " << num_omp_threads << "\n";

    // Read a sparse dataset from file.
    CSRMatrix *datamatrix = new CSRMatrix(input_filepath, true);
    int dim = datamatrix->ncol;
    int num_points = datamatrix->nrow;

    // std::cout << "Applying Hilbert ordering before index construction...\n";
    // auto start_reorder_time = std::chrono::steady_clock::now();
    // std::vector<uint32_t> old_to_new;
    // std::vector<uint32_t> new_to_old;
    // HilbertOrdering::reorderDataset(points, old_to_new, new_to_old);
    // auto end_reorder_time = std::chrono::steady_clock::now();
    // auto reorder_time = std::chrono::duration_cast<std::chrono::microseconds>(end_reorder_time - start_reorder_time);
    // std::cout << "Hilbert pre-order completed in " << reorder_time.count() << " microseconds\n";
    
    auto start_index_time = std::chrono::steady_clock::now();
    
    // Create SPARSE_HNSW index with 2D vectors.
    SPARSE_HNSW index(dim, datamatrix, M, ef_construction, num_points, use_heuristic, extend_candidates, keep_pruned, alpha, beta);
    // index.setLabelRemapping(std::move(old_to_new), std::move(new_to_old));

    CSRMatrix *pruned_datamatrix = nullptr;
    std::chrono::microseconds prune_time{0};
    if (alpha < 1.0) {
        std::cout << "Pruning dataset using mass ratio pruning with alpha = " << alpha << "...\n";

        auto start_prune_time = std::chrono::steady_clock::now();
        pruned_datamatrix = index.pruneMatrix(datamatrix);
        index.setPrunedDataMatrix(pruned_datamatrix);
        auto end_prune_time = std::chrono::steady_clock::now();
        prune_time = std::chrono::duration_cast<std::chrono::microseconds>(end_prune_time - start_prune_time);

        std::cout << "Pruning completed in " << prune_time.count() << " microseconds\n";
    } else if (alpha > 1.0) {
        std::cout << "Invalid alpha value: " << alpha << ". Alpha should be in the range (0, 1]. No pruning applied.\n";
        return 1;
    }else {
        std::cout << "No pruning applied (alpha = 1.0)\n";
    }
    
    
    // Add points from the dataset to the index.
    std::cout << "Adding points to the index...\n";

    index.addPointsBatch(num_points);

    auto end_index_time = std::chrono::steady_clock::now();
    auto index_time = std::chrono::duration_cast<std::chrono::microseconds>(end_index_time - start_index_time);

    std::cout << "Added " << num_points << " points to the index in " << index_time.count() << " microseconds.\n";
    std::cout << "Average insertion time: " << index_time.count() / num_points << " microseconds\n";
    
    // Search for nearest neighbors.
    std::cout << "\nSearching for k-nearest neighbors...\n";
    
    CSRMatrix *querymatrix = new CSRMatrix(query_filepath, true);
    int dim_query = datamatrix->ncol;
    int query_count = querymatrix->nrow;

    uint32_t *I = nullptr;
    uint32_t n, k;
    get_gt(gt_filepath, I, n, k);
    
    auto start_query_time = std::chrono::steady_clock::now();

    std::vector<uint32_t> pred_lables;
    index.searchKNNBatch(querymatrix, query_count, k, ef, pred_lables);

    auto end_query_time = std::chrono::steady_clock::now();
    auto query_time = std::chrono::duration_cast<std::chrono::microseconds>(end_query_time - start_query_time);
    
    float recall = calculate_recall(pred_lables, I, k, query_count) * 100.0f;
    std::cout << "Recall@k: " << std::fixed << std::setprecision(2) << recall << "%\n";
    std::cout << "Total Query time: " << query_time.count() << " microseconds\n";
    std::cout << "Average Query time: " << query_time.count() / query_count << " microseconds\n";

    index.printInfo();

    appendResultsRow(results_csv_path, alpha, beta, num_points, num_omp_threads,
                      prune_time.count() / 1e6, index_time.count() / 1e6,
                      query_time.count() / 1e6, recall);
    std::cout << "Appended results row to " << results_csv_path << "\n";

    std::cout << "\nDemo completed successfully!\n";

    return 0;
}
