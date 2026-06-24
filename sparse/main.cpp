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

#if defined(__has_include)
#if __has_include(<mkl.h>)
#include <mkl.h>
#define SPARSE_HNSW_HAS_MKL 1
#endif
#endif

#ifndef SPARSE_HNSW_HAS_MKL
#define SPARSE_HNSW_HAS_MKL 0
#endif

using namespace sparse_hnsw;

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
        std::cerr << "Usage: " << argv[0] << " <M> <ef_construction> <ef> <use_heuristic> <extend_candidates> <keep_pruned> <use_mkl> <mklThreshold> <input_filepath> <query_filepath> <gt_filepath> <timing_csv_path>" << std::endl;
        return 1;
    }

    // if (argc < 13)
    // {
    //     std::cerr << "Usage: " << argv[0] << " <M> <ef_construction> <ef> <use_heuristic> <extend_candidates> <keep_pruned> <use_mkl> <mklThreshold> <input_filepath> <query_filepath> <gt_filepath> <output_path_folder>" << std::endl;
    //     return 1;
    // }

    // Parse command line arguments into variables.
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
    std::string timing_csv_path = argv[12];
    // std::string output_path_folder = argv[13];
    // std::string dist_counts_insertion_output_path = output_path_folder + "/layer0_distance_counts_insertion.txt";
    // std::string cand_elements_insertion_output_path = output_path_folder + "/layer0_cand_elements_counts_insertion.txt";
    // std::string max_hops_insertion_output_path = output_path_folder + "/layer0_max_hops_counts_insertion.txt";
    // std::string dist_counts_search_output_path = output_path_folder + "/layer0_distance_counts_search.txt";
    // std::string cand_elements_search_output_path = output_path_folder + "/layer0_cand_elements_counts_search.txt";
    // std::string max_hops_search_output_path = output_path_folder + "/layer0_max_hops_counts_search.txt";

    int num_omp_threads = omp_get_max_threads();
    int num_mkl_threads = mkl_get_max_threads();
    std::cout << "Number of OpenMP threads: " << num_omp_threads << "\n";
    std::cout << "Number of MKL threads: " << num_mkl_threads << "\n";

    // Read a dense dataset from file.
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
    SPARSE_HNSW index(dim, datamatrix, M, ef_construction, num_points, use_heuristic, extend_candidates, keep_pruned, use_mkl, mklThreshold);
    // index.setLabelRemapping(std::move(old_to_new), std::move(new_to_old));
    
    // Add points from the dataset to the index.
    std::cout << "Adding points to the index...\n";
    
    for (size_t i = 0; i < num_points; ++i) {
        index.addPoint(i, i);
    }

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

    std::vector<uint32_t> pred_lables(query_count * k);

    for (int i = 0; i < query_count; i++) {
        std::priority_queue<std::pair<float, uint32_t>> nns = index.searchKNN(i, querymatrix, k, ef);
        while (!nns.empty()) {
            auto nn = nns.top();
            nns.pop();
            pred_lables[i * k + (k - nns.size() - 1)] = nn.second;
        }
    }

    auto end_query_time = std::chrono::steady_clock::now();
    auto query_time = std::chrono::duration_cast<std::chrono::microseconds>(end_query_time - start_query_time);
    
    float recall = calculate_recall(pred_lables, I, k, query_count) * 100.0f;
    std::cout << "Recall@k: " << std::fixed << std::setprecision(2) << recall << "%\n";
    std::cout << "Total Query time: " << query_time.count() << " microseconds\n";
    std::cout << "Average Query time: " << query_time.count() / query_count << " microseconds\n";

    index.printInfo(timing_csv_path);

    // if (index.dumpLayer0Counts(dist_counts_insertion_output_path, "dist_calc_insertion")) {
    //     std::cout << "Wrote insertion layer-0 distance counts to: " << dist_counts_insertion_output_path << "\n";
    // } else {
    //     std::cerr << "Failed to write insertion layer-0 distance counts to: " << dist_counts_insertion_output_path << "\n";
    // }

    // if (index.dumpLayer0Counts(cand_elements_insertion_output_path, "cand_elements_insertion")) {
    //     std::cout << "Wrote insertion layer-0 candidate elements counts to: " << cand_elements_insertion_output_path << "\n";
    // } else {
    //     std::cerr << "Failed to write insertion layer-0 candidate elements counts to: " << cand_elements_insertion_output_path << "\n";
    // }

    // if (index.dumpLayer0Counts(max_hops_insertion_output_path, "max_hops_insertion")) {
    //     std::cout << "Wrote insertion layer-0 max hops counts to: " << max_hops_insertion_output_path << "\n";
    // } else {
    //     std::cerr << "Failed to write insertion layer-0 max hops counts to: " << max_hops_insertion_output_path << "\n";
    // }

    // if (index.dumpLayer0Counts(dist_counts_search_output_path, "dist_calc_search")) {
    //     std::cout << "Wrote search layer-0 distance counts to: " << dist_counts_search_output_path << "\n";
    // } else {
    //     std::cerr << "Failed to write search layer-0 distance counts to: " << dist_counts_search_output_path << "\n";
    // }

    // if (index.dumpLayer0Counts(cand_elements_search_output_path, "cand_elements_search")) {
    //     std::cout << "Wrote search layer-0 candidate elements counts to: " << cand_elements_search_output_path << "\n";
    // } else {
    //     std::cerr << "Failed to write search layer-0 candidate elements counts to: " << cand_elements_search_output_path << "\n";
    // }

    // if (index.dumpLayer0Counts(max_hops_search_output_path, "max_hops_search")) {
    //     std::cout << "Wrote search layer-0 max hops counts to: " << max_hops_search_output_path << "\n";
    // } else {
    //     std::cerr << "Failed to write search layer-0 max hops counts to: " << max_hops_search_output_path << "\n";
    // }
    
    std::cout << "\nDemo completed successfully!\n";
    
    return 0;
}
