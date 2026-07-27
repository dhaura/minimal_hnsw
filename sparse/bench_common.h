#ifndef SPARSE_BENCH_COMMON_H
#define SPARSE_BENCH_COMMON_H

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
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
// ground truth. Same definition sindi_ex.py / the other spknn-playground
// runners use -- it is scored against exact NN, not against MS MARCO qrels.
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

// One sweep point. searching_time_sec is wall time for the whole query batch,
// so QPS/latency are throughput figures at whatever thread count the run used.
inline void appendRow(const std::string &csv_path, const std::string &model,
                      double recall, double indexing_time_sec,
                      double searching_time_sec, uint32_t num_queries,
                      double rr, const std::string &params) {
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
        csv << "Model,Recall,Indexing Time,Single Query Time (microseconds),"
               "Searching Time (Seconds),QPS,RR@10,params\n";
    }

    const double latency_us = searching_time_sec / num_queries * 1e6;
    const double qps = num_queries / searching_time_sec;

    csv << model << ","
        << std::fixed << std::setprecision(10) << recall << ","
        << std::setprecision(2) << indexing_time_sec << ","
        << std::setprecision(6) << latency_us << ","
        << std::setprecision(16) << searching_time_sec << ","
        << std::setprecision(2) << qps << ","
        << std::setprecision(4) << rr << ","
        << "\"" << params << "\"\n";
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
              << " | RR " << std::setprecision(4) << rr << "\n";
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
