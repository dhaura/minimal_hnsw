// Sweeps SINDI (vsag) over doc_prune_ratio x query_prune_ratio x n_candidate,
// recording a recall/throughput point per combination.

#include "bench_common.h"

#include <vsag/vsag.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>

struct CSRData {
    int64_t nrow{0};
    int64_t ncol{0};
    int64_t nnz{0};
    std::vector<int64_t> indptr;
    std::vector<uint32_t> indices;
    std::vector<float> data;

    explicit CSRData(const std::string &data_file_path) {
        std::ifstream infile(data_file_path, std::ios::binary);
        if (infile.fail()) {
            std::cerr << "Failed to open file " << data_file_path;
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
    std::vector<vsag::SparseVector> make_sparse_vectors() {
        std::vector<vsag::SparseVector> rows(nrow);
        for (int64_t i = 0; i < nrow; ++i) {
            rows[i].len_ = static_cast<uint32_t>(indptr[i + 1] - indptr[i]);
            rows[i].ids_ = indices.data() + indptr[i];
            rows[i].vals_ = data.data() + indptr[i];
        }
        return rows;
    }
};

int main(int argc, char *argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <doc_prune_ratio_list> <query_prune_ratio_list> <n_candidate_list>"
                     " <input_filepath> <query_filepath> <gt_filepath>"
                     " <results_csv_path> [term_prune_ratio=0] [window_size=50000]"
                     " [use_reorder=1] [use_quantization=0|1|fp16] [model_name=SINDI]"
                     " [repeats=5] [warmup=1]"
                  << std::endl;
        return 1;
    }

    std::vector<float> dpr_list = bench::parseList<float>(argv[1], bench::toFloat);
    std::vector<float> qpr_list = bench::parseList<float>(argv[2], bench::toFloat);
    std::vector<int> ncand_list = bench::parseList<int>(argv[3], bench::toInt);
    std::string input_filepath = argv[4];
    std::string query_filepath = argv[5];
    std::string gt_filepath = argv[6];
    std::string results_csv_path = argv[7];
    float term_prune_ratio = (argc > 8) ? std::stof(argv[8]) : 0.0f;
    int window_size = (argc > 9) ? std::stoi(argv[9]) : 50000;
    bool use_reorder = (argc > 10) ? (std::stoi(argv[10]) != 0) : true;
    std::string use_quantization = (argc > 11) ? argv[11] : "0";
    std::string model_name = (argc > 12) ? argv[12] : "SINDI";
    int repeats = (argc > 13) ? std::stoi(argv[13]) : 5;
    int warmup = (argc > 14) ? std::stoi(argv[14]) : 1;

    if (use_quantization == "0")
        use_quantization = "false";
    else if (use_quantization == "1")
        use_quantization = "true";
    else if (use_quantization != "fp16") {
        std::cerr << "use_quantization must be 0, 1 (SQ8) or fp16" << std::endl;
        return 1;
    }
    if (dpr_list.empty() || qpr_list.empty() || ncand_list.empty()) {
        std::cerr << "doc_prune_ratio_list, query_prune_ratio_list and "
                     "n_candidate_list must all be non-empty." << std::endl;
        return 1;
    }

    vsag::init();
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kERR);

    int num_omp_threads = omp_get_max_threads();
    std::cout << "SINDI (vsag) sweep\n==================\n"
              << "doc_prune_ratios=" << argv[1]
              << " window_size=" << window_size
              << " use_reorder=" << use_reorder
              << " use_quantization=" << use_quantization
              << " threads=" << num_omp_threads << "\n";

    auto start_load_time = std::chrono::steady_clock::now();
    CSRData datamatrix(input_filepath);
    auto end_load_time = std::chrono::steady_clock::now();
    double load_time_sec =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_load_time - start_load_time).count() / 1e6;
    std::cout << "Loaded base vectors in " << load_time_sec << " seconds.\n";

    int64_t dim = datamatrix.ncol;
    int64_t num_points = datamatrix.nrow;

    std::vector<vsag::SparseVector> base_vectors = datamatrix.make_sparse_vectors();
    std::vector<int64_t> base_ids(num_points);
    for (int64_t i = 0; i < num_points; ++i) base_ids[i] = i;

    auto base = vsag::Dataset::Make();
    base->NumElements(num_points)
        ->SparseVectors(base_vectors.data())
        ->Ids(base_ids.data())
        ->Owner(false);

    CSRData querymatrix(query_filepath);
    int64_t query_count = querymatrix.nrow;
    std::vector<vsag::SparseVector> query_vectors = querymatrix.make_sparse_vectors();

    std::vector<uint32_t> I;
    uint32_t n_gt, k;
    bench::get_gt(gt_filepath, I, n_gt, k);
    if (static_cast<int64_t>(n_gt) != query_count) {
        std::cerr << "Ground truth has " << n_gt << " rows but query file has "
                  << query_count << ".\n";
        return 1;
    }

    // UINT32_MAX = "no result" (never matches a GT id).
    std::vector<uint32_t> pred_labels(query_count * k, UINT32_MAX);

    for (float doc_prune_ratio : dpr_list) {
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
        if (!create_result.has_value()) {
            std::cerr << "Failed to create index: " << create_result.error().message << std::endl;
            return 1;
        }
        auto index = create_result.value();

        std::cout << "\n=== doc_prune_ratio=" << doc_prune_ratio
                  << ": building index ===\n";
        auto start_index_time = std::chrono::steady_clock::now();
        auto build_result = index->Build(base);
        if (!build_result.has_value()) {
            std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
            return 1;
        }
        if (!build_result.value().empty())
            std::cout << "Warning: " << build_result.value().size() << " points failed to insert.\n";
        auto end_index_time = std::chrono::steady_clock::now();

        double indexing_time_sec =
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_index_time - start_index_time).count() / 1e6;
        std::cout << "Added " << index->GetNumElements() << " points to the index in "
                  << indexing_time_sec << " seconds.\n";

        auto run = [&](int n_candidate, float query_prune_ratio) {
            std::ostringstream search_parameters;
            search_parameters << R"({
                "sindi": {
                    "query_prune_ratio": )" << query_prune_ratio << R"(,
                    "term_prune_ratio": )" << term_prune_ratio << R"(,
                    "n_candidate": )" << n_candidate << R"(
                }
            })";
            std::string search_parameters_str = search_parameters.str();

            std::fill(pred_labels.begin(), pred_labels.end(), UINT32_MAX);

            #pragma omp parallel for schedule(dynamic, 4)
            for (int64_t i = 0; i < query_count; i++) {
                auto query = vsag::Dataset::Make();
                query->NumElements(1)->SparseVectors(query_vectors.data() + i)->Owner(false);

                auto result = index->KnnSearch(query, k, search_parameters_str);
                if (!result.has_value()) continue;

                const int64_t *ids = result.value()->GetIds();
                int64_t nres = result.value()->GetDim();
                for (int64_t j = 0; j < nres && j < (int64_t)k; ++j)
                    pred_labels[i * k + j] = static_cast<uint32_t>(ids[j]);
            }
        };

        std::cout << "Sweeping " << qpr_list.size() * ncand_list.size()
                  << " (query_prune_ratio, n_candidate) points (" << query_count
                  << " queries, k=" << k << ")\n"
                  << "  " << repeats << " timed passes per point after " << warmup
                  << " warm-up pass(es); the median is reported.\n";

        for (float qpr : qpr_list) {
            for (int n_candidate : ncand_list) {
                std::vector<double> times = bench::timedRuns(
                    [&] { run(n_candidate, qpr); }, repeats, warmup);

                double recall = bench::calculate_recall(pred_labels, I, k, query_count);
                double rr = bench::calculate_rr(pred_labels, I, k, query_count);

                std::ostringstream params;
                params << "doc_prune=" << doc_prune_ratio
                       << " query_prune=" << qpr
                       << " n_cand=" << n_candidate;

                bench::printPoint(params.str(), recall, bench::median(times), query_count, rr);
                bench::appendRow(results_csv_path, model_name, params.str(), num_omp_threads,
                                 recall, rr, indexing_time_sec, load_time_sec, 0.0,
                                 times, query_count, k);
            }
        }
    }

    std::cout << "\nResults appended to " << results_csv_path << "\n";
    return 0;
}
