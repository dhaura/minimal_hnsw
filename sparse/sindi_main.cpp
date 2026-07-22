#include <vsag/vsag.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <iomanip>
#include <omp.h>
#include <chrono>

struct CSRData
{
    int64_t nrow{0};
    int64_t ncol{0};
    int64_t nnz{0};
    std::vector<int64_t> indptr;
    std::vector<uint32_t> indices;
    std::vector<float> data;

    explicit CSRData(const std::string &data_file_path)
    {
        std::ifstream infile(data_file_path, std::ios::binary);

        if (infile.fail())
        {
            std::cerr << std::string("Failed to open file ") + data_file_path;
            exit(1);
        }

        infile.read((char *)&nrow, sizeof(int64_t));
        infile.read((char *)&ncol, sizeof(int64_t));
        infile.read((char *)&nnz, sizeof(int64_t));

        indptr.resize(nrow + 1);
        infile.read((char *)indptr.data(), (nrow + 1) * sizeof(int64_t));

        indices.resize(nnz);
        infile.read((char *)indices.data(), nnz * sizeof(int32_t));

        data.resize(nnz);
        infile.read((char *)data.data(), nnz * sizeof(float));
        infile.close();
    }

    // vsag::SparseVector views into the CSR arrays (no copy).
    std::vector<vsag::SparseVector> make_sparse_vectors()
    {
        std::vector<vsag::SparseVector> rows(nrow);
        for (int64_t i = 0; i < nrow; ++i)
        {
            rows[i].len_ = static_cast<uint32_t>(indptr[i + 1] - indptr[i]);
            rows[i].ids_ = indices.data() + indptr[i];
            rows[i].vals_ = data.data() + indptr[i];
        }
        return rows;
    }
};

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
    }

    return static_cast<double>(total_hits) / (static_cast<double>(num_queries) * k);
}

int main(int argc, char *argv[])
{
    std::cout << "SINDI (vsag) Demo\n";
    std::cout << "=================\n\n";

    if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <doc_prune_ratio> <query_prune_ratio> <n_candidate>"
                     " <input_filepath> <query_filepath> <gt_filepath>"
                     " [term_prune_ratio=0] [window_size=50000] [use_reorder=1] [use_quantization=0|1|fp16]"
                  << std::endl;
        return 1;
    }

    // Parse command line arguments into variables.
    float doc_prune_ratio = std::stof(argv[1]);
    float query_prune_ratio = std::stof(argv[2]);
    int n_candidate = std::stoi(argv[3]);
    std::string input_filepath = argv[4];
    std::string query_filepath = argv[5];
    std::string gt_filepath = argv[6];
    float term_prune_ratio = (argc > 7) ? std::stof(argv[7]) : 0.0f;
    int window_size = (argc > 8) ? std::stoi(argv[8]) : 50000;
    bool use_reorder = (argc > 9) ? (std::stoi(argv[9]) != 0) : true;
    std::string use_quantization = (argc > 10) ? argv[10] : "0";
    if (use_quantization == "0")
        use_quantization = "false";
    else if (use_quantization == "1")
        use_quantization = "true";
    else if (use_quantization != "fp16")
    {
        std::cerr << "use_quantization must be 0, 1 (SQ8) or fp16" << std::endl;
        return 1;
    }

    vsag::init();
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kERR);

    // Read a sparse dataset from file.
    CSRData datamatrix(input_filepath);
    int64_t dim = datamatrix.ncol;
    int64_t num_points = datamatrix.nrow;

    std::vector<vsag::SparseVector> base_vectors = datamatrix.make_sparse_vectors();
    std::vector<int64_t> base_ids(num_points);
    for (int64_t i = 0; i < num_points; ++i)
        base_ids[i] = i;

    auto base = vsag::Dataset::Make();
    base->NumElements(num_points)
        ->SparseVectors(base_vectors.data())
        ->Ids(base_ids.data())
        ->Owner(false);

    std::ostringstream build_parameters;
    build_parameters << R"({
        "dtype": "sparse",
        "dim": )" << dim << R"(,
        "metric_type": "ip",
        "index_param": {
            "use_reorder": )" << (use_reorder ? "true" : "false") << R"(,
            "use_quantization": )" << use_quantization << R"(,
            "term_id_limit": )" << dim << R"(,
            "doc_prune_ratio": )" << doc_prune_ratio << R"(,
            "window_size": )" << window_size << R"(
        }
    })";

    auto create_result = vsag::Factory::CreateIndex("sindi", build_parameters.str());
    if (!create_result.has_value())
    {
        std::cerr << "Failed to create index: " << create_result.error().message << std::endl;
        return 1;
    }
    auto index = create_result.value();

    std::cout << "Adding points to the index...\n";

    auto start_index_time = std::chrono::steady_clock::now();

    // SINDI's Build() inserts serially under a global write lock.
    auto build_result = index->Build(base);
    if (!build_result.has_value())
    {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        return 1;
    }
    if (!build_result.value().empty())
        std::cout << "Warning: " << build_result.value().size() << " points failed to insert.\n";

    auto end_index_time = std::chrono::steady_clock::now();
    auto index_time = std::chrono::duration_cast<std::chrono::microseconds>(end_index_time - start_index_time);

    std::cout << "Added " << index->GetNumElements() << " points to the index in " << index_time.count() << " microseconds.\n";
    std::cout << "Average insertion time: " << index_time.count() / num_points << " microseconds\n";

    // Search for nearest neighbors.
    std::cout << "\nSearching for k-nearest neighbors...\n";

    CSRData querymatrix(query_filepath);
    int64_t query_count = querymatrix.nrow;
    std::vector<vsag::SparseVector> query_vectors = querymatrix.make_sparse_vectors();

    uint32_t *I = nullptr;
    uint32_t n, k;
    get_gt(gt_filepath, I, n, k);

    std::ostringstream search_parameters;
    search_parameters << R"({
        "sindi": {
            "query_prune_ratio": )" << query_prune_ratio << R"(,
            "term_prune_ratio": )" << term_prune_ratio << R"(,
            "n_candidate": )" << n_candidate << R"(
        }
    })";
    std::string search_parameters_str = search_parameters.str();

    auto start_query_time = std::chrono::steady_clock::now();

    // UINT32_MAX = "no result" (never matches a GT id).
    std::vector<uint32_t> pred_lables(query_count * k, UINT32_MAX);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t i = 0; i < query_count; i++)
    {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->SparseVectors(query_vectors.data() + i)->Owner(false);

        auto result = index->KnnSearch(query, k, search_parameters_str);
        if (!result.has_value())
            continue;

        const int64_t *ids = result.value()->GetIds();
        int64_t nres = result.value()->GetDim();
        for (int64_t j = 0; j < nres && j < (int64_t)k; ++j)
            pred_lables[i * k + j] = static_cast<uint32_t>(ids[j]);
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
