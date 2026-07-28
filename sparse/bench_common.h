#ifndef SPARSE_BENCH_COMMON_H
#define SPARSE_BENCH_COMMON_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace bench {

// BigANN-style ground truth: uint32 n, uint32 d, then n*d uint32 ids.
inline void get_gt(const std::string &gt_path, std::vector<uint32_t> &I,
                   uint32_t &n, uint32_t &d) {
    std::ifstream infile(gt_path, std::ios::binary);
    if (infile.fail()) {
        std::cerr << "Failed to open file " << gt_path << "\n";
        exit(1);
    }
    infile.read((char *)&n, sizeof(uint32_t));
    infile.read((char *)&d, sizeof(uint32_t));
    I.resize(static_cast<size_t>(n) * d);
    infile.read((char *)I.data(), static_cast<size_t>(n) * d * sizeof(uint32_t));
    infile.close();
}

// Recall@k as |pred ∩ gt| / (num_queries * k), in [0, 1].
inline double calculate_recall(const std::vector<uint32_t> &predicted_labels,
                               const std::vector<uint32_t> &I, uint32_t k,
                               uint32_t num_queries) {
    uint64_t total_hits = 0;
    for (uint32_t i = 0; i < num_queries; ++i) {
        std::unordered_set<uint32_t> gt_neighbors(
            I.begin() + static_cast<size_t>(i) * k,
            I.begin() + static_cast<size_t>(i + 1) * k);
        for (uint32_t j = 0; j < k; ++j) {
            if (gt_neighbors.count(predicted_labels[static_cast<size_t>(i) * k + j]))
                ++total_hits;
        }
    }
    return static_cast<double>(total_hits) /
           (static_cast<double>(num_queries) * k);
}

// Mean reciprocal rank of the first predicted id that is in the exact-NN
// ground truth.
inline double calculate_rr(const std::vector<uint32_t> &predicted_labels,
                           const std::vector<uint32_t> &I, uint32_t k,
                           uint32_t num_queries) {
    double rr = 0.0;
    for (uint32_t i = 0; i < num_queries; ++i) {
        std::unordered_set<uint32_t> gt_neighbors(
            I.begin() + static_cast<size_t>(i) * k,
            I.begin() + static_cast<size_t>(i + 1) * k);
        for (uint32_t rank = 1; rank <= k; ++rank) {
            if (gt_neighbors.count(
                    predicted_labels[static_cast<size_t>(i) * k + rank - 1])) {
                rr += 1.0 / rank;
                break;
            }
        }
    }
    return rr / num_queries;
}

// Peak resident set size of this process, in GB. ru_maxrss is KiB on Linux.
inline double peakRssGb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
}

inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline std::string joinTimes(const std::vector<double> &v) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ';';
        os << v[i];
    }
    return os.str();
}

inline void appendRow(const std::string &csv_path, const std::string &model,
                      const std::string &params, int threads,
                      double recall, double rr,
                      double index_sec, double load_sec, double convert_sec,
                      const std::vector<double> &search_times,
                      uint32_t num_queries, uint32_t k,
                      uint64_t index_bytes = 0,
                      bool results_ordered = true) {
    bool write_header;
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
        csv << "Model,Params,Threads,Recall,RR@10,IndexSec,LoadSec,ConvertSec,"
               "SearchSecMedian,SearchSecRuns,AmortizedUsPerQuery,QPS,"
               "LatencyUsSingleThread,PeakRSSGB,IndexBytes,NumQueries,K,"
               "ResultsOrdered,Host,JobID\n";
    }

    const double med = median(search_times);
    const double amortized_us = med / num_queries * 1e6;
    const double qps = num_queries / med;

    char host[256] = {0};
    if (gethostname(host, sizeof(host) - 1) != 0) std::snprintf(host, sizeof(host), "?");
    const char *job = std::getenv("SLURM_JOB_ID");

    csv << model << ","
        << "\"" << params << "\","
        << threads << ","
        << std::fixed << std::setprecision(10) << recall << ","
        << std::setprecision(4) << rr << ","
        << std::setprecision(4) << index_sec << ","
        << std::setprecision(4) << load_sec << ","
        << std::setprecision(4) << convert_sec << ","
        << std::setprecision(6) << med << ","
        << "\"" << joinTimes(search_times) << "\","
        << std::setprecision(6) << amortized_us << ","
        << std::setprecision(2) << qps << ","
        << ","                                   // LatencyUsSingleThread: unset
        << std::setprecision(2) << peakRssGb() << ","
        << index_bytes << ","
        << num_queries << ","
        << k << ","
        << (results_ordered ? "True" : "False") << ","
        << host << ","
        << (job ? job : "") << "\n";
    csv.flush();
}

inline void printPoint(const std::string &params, double recall,
                       double searching_time_sec, uint32_t num_queries,
                       double rr) {
    std::cout << std::fixed
              << "  " << params
              << " | recall@k " << std::setprecision(4) << recall * 100.0 << "%"
              << " | " << std::setprecision(4) << searching_time_sec << " s"
              << " | " << std::setprecision(2) << searching_time_sec / num_queries * 1e6 << " us/query"
              << " | " << std::setprecision(2) << num_queries / searching_time_sec << " QPS"
              << " | RR " << std::setprecision(4) << rr
              << " | peak " << std::setprecision(1) << peakRssGb() << " GB\n";
}

template <typename F>
inline std::vector<double> timedRuns(F &&body, int repeats, int warmup) {
    for (int i = 0; i < warmup; ++i) body();
    std::vector<double> times;
    times.reserve(std::max(1, repeats));
    for (int i = 0; i < std::max(1, repeats); ++i) {
        auto t0 = std::chrono::steady_clock::now();
        body();
        auto t1 = std::chrono::steady_clock::now();
        times.push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6);
    }
    return times;
}

// Parses "10,20,50" into a vector.
template <typename T>
inline std::vector<T> parseList(const std::string &s, T (*conv)(const std::string &)) {
    std::vector<T> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        if (comma > start) out.push_back(conv(s.substr(start, comma - start)));
        start = comma + 1;
    }
    return out;
}

inline int toInt(const std::string &s) { return std::stoi(s); }
inline float toFloat(const std::string &s) { return std::stof(s); }

}  // namespace bench

#endif  // SPARSE_BENCH_COMMON_H
