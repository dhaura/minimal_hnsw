// Diagnostic ef sweep: why does recall plateau at the high-recall end?
//
// The production number (recall@k after the beta-refine) conflates two very
// different failure modes, and they have opposite fixes:
//
//   traversal loss   a ground-truth neighbour was NEVER VISITED by the graph
//                    walk, so no amount of re-ranking can recover it.
//                    Fix = a better graph / better navigation.
//
//   ranking loss     it WAS visited, but the pruned-distance ordering pushed
//                    it out of the k*beta shortlist before the exact refine
//                    could see it.
//                    Fix = a cheaper/wider refine, or a less biased estimator.
//
// Per ef this driver reports, over the same one index build:
//
//   containment@ef   |gt_top_k INTERSECT layer-0 candidate set| / k
//                      -> the reachability ceiling. 1 - this = traversal loss.
//   containment@khat |gt_top_k INTERSECT top-(k*beta) by PRUNED distance| / k
//                      -> what survives the shortlist.
//   recall@k         production recall after the exact refine.
//
//   traversal_loss = 1 - containment@ef
//   ranking_loss   = containment@ef - containment@khat
//
// plus ndist / bytes per query (SPARSE_HNSW_PROFILE), so the cost side of the
// recall/cost curve is measured in the same run rather than inferred.
//
// Usage mirrors sparse_hnsw_sweep, with an extra <diag_csv_path>.

#include "sparse_hnsw.h"
#include "csr_matrix.h"
#include "bench_common.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>

using namespace sparse_hnsw;

namespace {

// containment@ef IS the oracle recall: if all k ground-truth ids are inside the
// candidate set, they are by definition the top-k of that set under the exact
// metric, so a perfect re-ranker would return exactly them. No separate oracle
// rescoring pass is needed.
void appendDiagRow(const std::string& path, const std::string& params,
                   int ef, int k, int k_hat, int threads,
                   double containment_ef, double containment_khat,
                   double recall, double search_sec, uint32_t nq,
                   double ndist_per_query, double bytes_per_query,
                   double cand_set_size) {
    bool write_header;
    {
        std::ifstream check(path);
        write_header = !(check.good() && check.peek() != std::ifstream::traits_type::eof());
    }
    std::ofstream csv(path, std::ios::app);
    if (!csv) {
        std::cerr << "Failed to open diag CSV '" << path << "'\n";
        return;
    }
    if (write_header) {
        csv << "Params,ef,k,k_hat,Threads,ContainmentAtEf,ContainmentAtKhat,Recall,"
               "TraversalLoss,RankingLoss,SearchSec,QPS,NDistPerQuery,BytesPerQuery,"
               "CandSetSize,NDistPerRecallPoint\n";
    }
    const double traversal_loss = 1.0 - containment_ef;
    const double ranking_loss = containment_ef - containment_khat;
    csv << "\"" << params << "\"," << ef << "," << k << "," << k_hat << "," << threads << ","
        << std::fixed << std::setprecision(6)
        << containment_ef << "," << containment_khat << "," << recall << ","
        << traversal_loss << "," << ranking_loss << ","
        << search_sec << "," << (nq / search_sec) << ","
        << std::setprecision(2) << ndist_per_query << "," << bytes_per_query << ","
        << cand_set_size << ","
        << std::setprecision(1) << (recall > 0 ? ndist_per_query / recall : 0.0) << "\n";
    csv.flush();
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 14) {
        std::cerr << "Usage: " << argv[0]
                  << " <M> <ef_construction> <ef_list> <use_heuristic>"
                     " <extend_candidates> <keep_pruned> <alpha> <beta>"
                     " <input_filepath> <query_filepath> <gt_filepath>"
                     " <results_csv_path> <diag_csv_path>"
                     " [model_name=SparseHNSW] [repeats=3] [warmup=1]"
                     " [diag_queries=all] [quantize=0]"
                  << std::endl;
        return 1;
    }

    const int M = std::stoi(argv[1]);
    const int ef_construction = std::stoi(argv[2]);
    const std::vector<int> ef_list = bench::parseList<int>(argv[3], bench::toInt);
    const bool use_heuristic = (std::stoi(argv[4]) != 0);
    const bool extend_candidates = (std::stoi(argv[5]) != 0);
    const bool keep_pruned = (std::stoi(argv[6]) != 0);
    const double alpha = std::stod(argv[7]);
    const int beta = std::stoi(argv[8]);
    const std::string input_filepath = argv[9];
    const std::string query_filepath = argv[10];
    const std::string gt_filepath = argv[11];
    const std::string results_csv_path = argv[12];
    const std::string diag_csv_path = argv[13];
    const std::string model_name = (argc > 14) ? argv[14] : "SparseHNSW";
    const int repeats = (argc > 15) ? std::stoi(argv[15]) : 3;
    const int warmup = (argc > 16) ? std::stoi(argv[16]) : 1;
    const long diag_nq_arg = (argc > 17) ? std::stol(argv[17]) : 0;
    const bool quantize = (argc > 18) && (std::stoi(argv[18]) != 0);

    if (ef_list.empty()) {
        std::cerr << "ef_list is empty.\n";
        return 1;
    }

    const int num_omp_threads = omp_get_max_threads();
    std::cout << "SPARSE_HNSW diagnostic ef sweep\n===============================\n"
              << "M=" << M << " efC=" << ef_construction
              << " alpha=" << alpha << " beta=" << beta
              << " heuristic=" << use_heuristic
              << " threads=" << num_omp_threads << "\n";

    auto t0 = std::chrono::steady_clock::now();
    CSRMatrix* datamatrix = new CSRMatrix(input_filepath, true);
    auto t1 = std::chrono::steady_clock::now();
    const double load_time_sec =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
    std::cout << "Loaded base vectors in " << load_time_sec << " s ("
              << datamatrix->nrow << " x " << datamatrix->ncol
              << ", nnz=" << datamatrix->nnz << ")\n";

    const int dim = datamatrix->ncol;
    const int num_points = datamatrix->nrow;

    auto start_index = std::chrono::steady_clock::now();
    SPARSE_HNSW index(dim, datamatrix, M, ef_construction, num_points,
                      use_heuristic, extend_candidates, keep_pruned,
                      static_cast<float>(alpha), beta);

    CSRMatrix* pruned = nullptr;
    if (alpha < 1.0) {
        std::cout << "Pruning with alpha=" << alpha << " ...\n";
        pruned = index.pruneMatrix(datamatrix);
        index.setPrunedDataMatrix(pruned);
        std::cout << "  nnz " << datamatrix->nnz << " -> " << pruned->nnz
                  << " (keep_frac="
                  << static_cast<double>(pruned->nnz) / datamatrix->nnz << ")\n";
    } else if (alpha > 1.0) {
        std::cerr << "Invalid alpha " << alpha << "; expected (0, 1].\n";
        return 1;
    }

    std::cout << "Building index...\n";
    index.addPointsBatch(num_points);
    if (quantize) {
        index.enableQuantizedTraversal();
        std::cout << "Quantized traversal enabled ("
                  << index.quantizedBytes() / (1024.0 * 1024.0) << " MiB)\n";
    }
    auto end_index = std::chrono::steady_clock::now();
    const double indexing_time_sec =
        std::chrono::duration_cast<std::chrono::microseconds>(end_index - start_index).count() / 1e6;
    std::cout << "Built in " << indexing_time_sec << " s (pruning included)\n";

    CSRMatrix* querymatrix = new CSRMatrix(query_filepath, true);
    const int query_count = querymatrix->nrow;

    std::vector<uint32_t> I;
    uint32_t n_gt = 0, k = 0;
    bench::get_gt(gt_filepath, I, n_gt, k);
    if (static_cast<int>(n_gt) != query_count) {
        std::cerr << "Ground truth rows " << n_gt << " != queries " << query_count << "\n";
        return 1;
    }

    const int k_hat = (alpha < 1.0) ? static_cast<int>(k) * beta : static_cast<int>(k);
    const int diag_nq = (diag_nq_arg > 0 && diag_nq_arg < query_count)
                            ? static_cast<int>(diag_nq_arg) : query_count;

    std::cout << "\nSweeping ef over " << ef_list.size() << " values ("
              << query_count << " queries, k=" << k << ", k_hat=" << k_hat
              << "; diagnostics on " << diag_nq << " queries)\n";

    for (const int ef : ef_list) {
        if (ef < static_cast<int>(k)) {
            std::cerr << "  skipping ef=" << ef << " (< k=" << k << ")\n";
            continue;
        }

        // ---- Timed production path (also collects the profile counters) ----
        std::vector<uint32_t> pred_labels;
#ifdef SPARSE_HNSW_PROFILE
        index.profReset();
#endif
        std::vector<double> times = bench::timedRuns(
            [&] { index.searchKNNBatch(querymatrix, query_count, k, ef, pred_labels); },
            repeats, warmup);
        const double recall = bench::calculate_recall(pred_labels, I, k, query_count);
        const double rr = bench::calculate_rr(pred_labels, I, k, query_count);
        const double search_sec = bench::median(times);

        double ndist_per_query = 0.0, bytes_per_query = 0.0;
#ifdef SPARSE_HNSW_PROFILE
        // timedRuns ran (warmup + repeats) passes over every query.
        const double passes = static_cast<double>(warmup + std::max(1, repeats));
        const double total_q = passes * query_count;
        ndist_per_query =
            (index.profNDist() + index.profRefineNDist()) / total_q;
        bytes_per_query =
            (index.profBytes() + index.profRefineBytes() + index.profGraphBytes()) / total_q;
#endif

        // ---- Untimed diagnostic pass ----
        // One layer-0 search per query returning the FULL candidate set, so
        // containment@ef and containment@k_hat come from the same walk.
        //
        // searchKNN widens layer 0 to max(ef, k) and searchKNNBatch passes
        // k = k_hat, so the production walk is really ef_eff wide. Reproduce
        // that here or the two halves of the comparison disagree at small ef
        // (this is also why ef=10 and ef=20 score identically in the sweeps).
        const int ef_eff = std::max(ef, k_hat);
        uint64_t hits_ef = 0, hits_khat = 0, cand_total = 0;

        #pragma omp parallel reduction(+ : hits_ef, hits_khat, cand_total)
        {
            SearchScratch scratch;
            scratch.prepare(num_points);
            std::vector<std::pair<float, uint32_t>> cands;

            #pragma omp for schedule(dynamic, 4)
            for (int q = 0; q < diag_nq; ++q) {
                // k = ef_eff keeps the whole layer-0 result set.
#ifdef SPARSE_HNSW_PROFILE
                std::priority_queue<std::pair<float, uint32_t>> pq =
                    index.searchKNNProf(static_cast<uint32_t>(q), querymatrix, ef_eff, ef_eff, scratch);
#else
                std::priority_queue<std::pair<float, uint32_t>> pq =
                    index.searchKNN(static_cast<uint32_t>(q), querymatrix, ef_eff, ef_eff);
#endif
                cands.clear();
                while (!pq.empty()) {
                    cands.push_back(pq.top());
                    pq.pop();
                }
                // cands is now ordered farthest -> nearest by PRUNED distance.
                cand_total += cands.size();

                const std::unordered_set<uint32_t> gt(
                    I.begin() + static_cast<size_t>(q) * k,
                    I.begin() + static_cast<size_t>(q + 1) * k);

                for (const auto& c : cands) {
                    if (gt.count(c.second)) ++hits_ef;
                }

                // The last k_hat entries are the k_hat nearest by pruned distance:
                // exactly the shortlist searchKNNBatch hands to the refine pass.
                const size_t take = std::min<size_t>(k_hat, cands.size());
                for (size_t i = cands.size() - take; i < cands.size(); ++i) {
                    if (gt.count(cands[i].second)) ++hits_khat;
                }
            }
        }

        // NOTE: an earlier revision also ran an "oracle-seeded" pass here --
        // layer 0 started AT the true nearest neighbour via a
        // searchLayer0FromSeeds() hook on SPARSE_HNSW -- to split traversal
        // loss into entry-point vs connectivity. The hook was removed from the
        // library; the measured result (gain collapses from +11.6 pp at ef=10
        // to +0.11 pp at ef=3200) is recorded in HIGH_RECALL_ANALYSIS.md.
        const double denom = static_cast<double>(diag_nq) * k;
        const double containment_ef = hits_ef / denom;
        const double containment_khat = hits_khat / denom;
        const double cand_set_size = static_cast<double>(cand_total) / diag_nq;

        std::ostringstream params;
        params << "M=" << M << " efC=" << ef_construction << " ef=" << ef
               << " alpha=" << alpha << " beta=" << beta;

        std::cout << std::fixed
                  << "  ef=" << std::setw(5) << ef
                  << " | recall@" << k << " " << std::setprecision(4) << recall * 100 << "%"
                  << " | contain@ef " << containment_ef * 100 << "%"
                  << " | trav_loss " << (1 - containment_ef) * 100 << "%"
                  << " | rank_loss " << (containment_ef - containment_khat) * 100 << "%"
                  << " | ndist/q " << std::setprecision(1) << ndist_per_query
                  << " | " << std::setprecision(2) << query_count / search_sec << " QPS\n";

        bench::appendRow(results_csv_path, model_name, params.str(), num_omp_threads,
                         recall, rr, indexing_time_sec, load_time_sec, 0.0,
                         times, query_count, k);
        appendDiagRow(diag_csv_path, params.str(), ef, k, k_hat, num_omp_threads,
                      containment_ef, containment_khat, recall, search_sec,
                      query_count, ndist_per_query, bytes_per_query, cand_set_size);
    }

    std::cout << "\nResults  -> " << results_csv_path
              << "\nDiagnostics -> " << diag_csv_path << "\n";
    return 0;
}
