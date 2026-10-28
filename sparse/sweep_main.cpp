// Builds one SPARSE_HNSW index and then sweeps ef over a list, recording a
// recall/throughput point per ef.

#include "sparse_hnsw.h"
#include "csr_matrix.h"
#include "bench_common.h"

#include <algorithm>
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
                     " [seed_top_k=0] [seed_spec=off] [stop_list=0]"
                     " [slot_size=-320] [bloom_thresholds=off] [bloom_hashes=1] [gate_layouts=sep]"
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
    // Stop rules (comma list): "P" = patience P (0 = none); "gG" = score gap gamma G.
    struct StopCfg { std::string spec; int P = 0; double gamma = -1.0; };
    std::vector<StopCfg> stop_list;
    for (const std::string& spec : bench::parseList<std::string>((argc > 19) ? argv[19] : "0",
                                                                [](const std::string& v) { return v; })) {
        StopCfg c;
        c.spec = spec;
        if (spec[0] == 'g') { c.gamma = std::stod(spec.substr(1)); } else { c.P = std::stoi(spec); }
        stop_list.push_back(c);
    }
    // Default -320: slots only, the library default layout. 0: packed rows only.
    const int slot_arg = (argc > 20) ? std::stoi(argv[20])
                                     : -static_cast<int>(SPARSE_HNSW::kDefaultSlotSize);
    const bool slots_only = slot_arg < 0;
    const int slot_size = slots_only ? 0 : slot_arg;
    // Bloom distance gate: comma list of thresholds (% of query terms that must
    // look present), each swept over the whole grid; "off" = no gate.
    // A negative entry runs that pass with the gate off (same-job baseline).
    const std::string bloom_arg = (argc > 21) ? argv[21] : "off";
    const std::vector<double> bloom_thresholds = bloom_arg == "off"
        ? std::vector<double>{-1.0} : bench::parseList<double>(bloom_arg, [](const std::string& v) { return std::stod(v); });
    const int bloom_hashes = (argc > 22) ? std::stoi(argv[22]) : 1;
    // Gate layouts (comma list): sep = Bloom filter rows in their own array, run at
    // every threshold; list16 = top-16-term sketch beside each layer-0 neighbor id
    // instead, run once with the Bloom check off (rows labelled _list16).
    const std::vector<std::string> gate_layouts = bench::parseList<std::string>(
        (argc > 23) ? argv[23] : "sep", [](const std::string& v) { return v; });
    for (const std::string& layout : gate_layouts) {
        if (layout != "sep" && layout != "list16") {
            std::cerr << "unknown gate layout '" << layout << "' (want sep or list16)\n";
            return 1;
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
    // IndexSec = everything needed before the first query: pruning, graph build,
    // and (added below) the quantized traversal rows and the seed table. It is
    // what the "Indexing Time" column means for the other methods. Loading the
    // base matrix is excluded (LoadSec).
    double indexing_time_sec =
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_index_time - start_index_time).count() / 1e6;
    std::cout << "Built a searchable index over " << num_points
              << " points in " << indexing_time_sec
              << " seconds (pruning + graph build).\n";

    index.printInfo();

    if (quantize && slots_only) {
        auto start_q = std::chrono::steady_clock::now();
        index.enableSlottedTraversal(static_cast<uint32_t>(-slot_arg));
        std::cout << "Slots-only quantized traversal: slot_size=" << -slot_arg << " B, overflow rows="
                  << index.slotOverflowRows() << ", " << index.slotBytes() / (1024.0 * 1024.0) << " MiB in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start_q).count() << " s\n";
        indexing_time_sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_q).count();
    } else if (quantize) {
        std::cout << "Building uint8-quantized traversal copy...\n";
        auto start_q = std::chrono::steady_clock::now();
        index.enableQuantizedTraversal(0);  // packed rows (slot_arg >= 0)
        auto end_q = std::chrono::steady_clock::now();
        indexing_time_sec += std::chrono::duration<double>(end_q - start_q).count();
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
        indexing_time_sec += std::chrono::duration<double>(end_seed - start_seed).count();
        std::cout << "Seed table built in "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         end_seed - start_seed).count() / 1e6
                  << " s (" << index.seedTableBytes() / 1024.0 << " KiB)\n";
    }

    if (bloom_arg != "off") {
        auto start_bloom = std::chrono::steady_clock::now();
        index.buildBloomFilter(static_cast<uint8_t>(bloom_hashes));
        const double bloom_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_bloom).count();
        indexing_time_sec += bloom_sec;
        std::cout << "Bloom filter built (h=" << bloom_hashes << ", " << index.bloomBytes() / (1024.0 * 1024.0)
                  << " MiB) in " << bloom_sec << " s\n";
    }
    double list_sec = 0.0;  // added to the indexing time of the list16 rows only
    if (std::find(gate_layouts.begin(), gate_layouts.end(), "list16") != gate_layouts.end()) {
        const auto t0 = std::chrono::steady_clock::now();
        index.buildListSketch();
        list_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "List sketch built: " << index.listSketchBytes() / (1024.0 * 1024.0) << " MiB in "
                  << list_sec << " s\n";
    }

    std::cout << "IndexSec (pruning + graph + quantized rows + seed table + Bloom filter, loading excluded): "
              << indexing_time_sec << " s\n";

    if (slot_size > 0) {
        if (!quantize) {
            std::cerr << "slot_size needs quantize=1.\n";
            return 1;
        }
        index.buildRowSlots(static_cast<uint32_t>(slot_size));
        std::cout << "Row slots: slot_size=" << slot_size << " B, overflow rows="
                  << index.slotOverflowRows() << ", " << index.slotBytes() / (1024.0 * 1024.0)
                  << " MiB\n";
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

    const bool multi_pat = stop_list.size() > 1;

    for (const std::string& layout : gate_layouts) {
      const bool list = layout == "list16";
      index.setListSketchGate(list);
      const double row_index_sec = indexing_time_sec + (list ? list_sec : 0.0);
    for (const double t : bloom_thresholds) {
      if (list && t != bloom_thresholds.front()) {
          continue;  // list16 ignores T: one pass
      }
      const double bloom_t = list ? -1.0 : t;
      index.setBloomThreshold(bloom_t);
    for (const StopCfg& stop : stop_list) {
      const int patience = stop.P;
      index.setPatience(patience);
      index.setStopGap(stop.gamma);
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
            row_model += "_p" + stop.spec;
        }
        if (bloom_t >= 0.0) {
            std::ostringstream bt;
            bt << bloom_t;
            row_model += "_T" + bt.str();
        }
        if (list) {
            row_model += "_" + layout;
        }

        for (int ef : ef_list) {
            if (ef < static_cast<int>(k)) {
                std::cerr << "  skipping ef=" << ef << " (< k=" << k << ")\n";
                continue;
            }

            std::vector<uint32_t> pred_labels;
            std::vector<double> times, slot_times;
            if (slot_size > 0) {
                // Alternate layouts each round so drift hits both equally.
                std::vector<uint32_t> slot_labels;
                for (int w = 0; w < warmup; ++w) {
                    index.setSlottedRows(false);
                    index.searchKNNBatch(querymatrix, query_count, k, ef, pred_labels);
                    index.setSlottedRows(true);
                    index.searchKNNBatch(querymatrix, query_count, k, ef, slot_labels);
                }
                for (int r = 0; r < 2 * std::max(1, repeats); ++r) {
                    const bool slotted = (r + r / 2) % 2 == 1;
                    index.setSlottedRows(slotted);
                    auto t0 = std::chrono::steady_clock::now();
                    index.searchKNNBatch(querymatrix, query_count, k, ef, slotted ? slot_labels : pred_labels);
                    (slotted ? slot_times : times).push_back(std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count());
                }
                index.setSlottedRows(false);
                if (slot_labels != pred_labels) {
                    std::cerr << "Slotted predictions differ from the original layout.\n";
                    return 1;
                }
            } else {
                times = bench::timedRuns(
                    [&] { index.searchKNNBatch(querymatrix, query_count, k, ef, pred_labels); },
                    repeats, warmup);
            }

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
            if (stop.gamma >= 0.0) {
                params << " gap=" << stop.gamma;
            }
            if (slots_only) {
                params << " slots=" << -slot_arg;
            }
            if (list) {
                params << " layout=" << layout;
            }
            if (bloom_t >= 0.0) {
                params << " bloom_h=" << bloom_hashes << " T=" << bloom_t;
            }

            bench::printPoint(params.str(), recall, bench::median(times), query_count, rr);
            bench::appendRow(results_csv_path, row_model, params.str(), num_omp_threads,
                             recall, rr, row_index_sec, load_time_sec, 0.0,
                             times, query_count, k);
            if (slot_size > 0) {
                const std::string slot_params = params.str() + " slots=" + std::to_string(slot_size);
                bench::printPoint(slot_params, recall, bench::median(slot_times), query_count, rr);
                bench::appendRow(results_csv_path, row_model + "_slots", slot_params, num_omp_threads,
                                 recall, rr, row_index_sec, load_time_sec, 0.0,
                                 slot_times, query_count, k);
            }
        }
      }
      }
    }
    }
    }

    std::cout << "\nResults appended to " << results_csv_path << "\n";
    return 0;
}
