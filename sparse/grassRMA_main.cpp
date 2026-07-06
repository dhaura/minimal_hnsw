#include "GrassRMA//hnswlib/hnswlib.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <chrono>

using namespace sparse_hnswlib;

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
    std::cout << "GrassRMA HNSW Demo\n";
    std::cout << "=================\n\n";

     if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <ef_construction> <ef> <input_filepath> <query_filepath> <gt_filepath>" << std::endl;
        return 1;
    }

    // Parse command line arguments into variables.
    int M = std::stoi(argv[1]);
    int ef_construction = std::stoi(argv[2]);
    int ef = std::stoi(argv[3]);
    std::string input_filepath = argv[4];
    std::string query_filepath = argv[5];
    std::string gt_filepath = argv[6];

    // Read a sparse dataset from file.
    CSRMatrix *datamatrix = new CSRMatrix(input_filepath, true);
    int dim = datamatrix->ncol;
    int num_points = datamatrix->nrow;
    
    auto start_index_time = std::chrono::steady_clock::now();

    // Create HNSW index with 2D vectors
    InnerProductSpace space(dim);
    HierarchicalNSW<float> *index =
        new HierarchicalNSW<float>(&space, datamatrix, num_points, M, ef_construction);
    index->ef_ = ef; // Set ef for search

    std::cout << "Adding points to the index...\n";

    for (size_t i = 0; i < num_points; ++i) {
        index->addPoint(i, i);
    }

    auto end_index_time = std::chrono::steady_clock::now();
    auto index_time = std::chrono::duration_cast<std::chrono::microseconds>(end_index_time - start_index_time);

    std::cout << "Added " << num_points << " points to the index in " << index_time.count() << " microseconds.\n";
    std::cout << "Average insertion time: " << index_time.count() / num_points << " microseconds\n";
    
    // Search for nearest neighbors
    std::cout << "\nSearching for k-nearest neighbors...\n";
    
    CSRMatrix *querymatrix = new CSRMatrix(query_filepath, true);
    int dim_query = datamatrix->ncol;
    int query_count = querymatrix->nrow;

    uint32_t *I = nullptr;
    uint32_t n, k;
    get_gt(gt_filepath, I, n, k);

    auto start_query_time = std::chrono::steady_clock::now();
    
    std::vector<uint32_t> pred_lables(query_count * k);

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < query_count; i++) {
        std::priority_queue<std::pair<float, labeltype>> nns = index->searchKnn(i, k, querymatrix);
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
    
    std::cout << "\nDemo completed successfully!\n";
    
    return 0;
}
