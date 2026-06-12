#include "hnsw.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstring>
#include <immintrin.h>
#include <unordered_set>
#include <omp.h>

#if defined(__has_include)
#if __has_include(<mkl.h>)
#include <mkl.h>
#define HNSW_HAS_MKL 1
#endif
#endif

#ifndef HNSW_HAS_MKL
#define HNSW_HAS_MKL 0
#endif

using namespace hnsw;

namespace {
struct LoopTimingStats {
    std::vector<uint64_t> samples_ns;

    void add(uint64_t ns) {
        samples_ns.push_back(ns);
    }

    double average_us() const {
        if (samples_ns.empty()) {
            return 0.0;
        }
        uint64_t total_ns = 0;
        for (uint64_t ns : samples_ns) {
            total_ns += ns;
        }
        return static_cast<double>(total_ns) / static_cast<double>(samples_ns.size()) / 1000.0;
    }

    size_t count() const {
        return samples_ns.size();
    }

    double sample_us(size_t index) const {
        if (index >= samples_ns.size()) {
            return 0.0;
        }
        return static_cast<double>(samples_ns[index]) / 1000.0;
    }
};

LoopTimingStats g_filter_loop_stats;
LoopTimingStats g_mkl_pack_loop_stats;
LoopTimingStats g_mkl_gemv_loop_stats;
LoopTimingStats g_distance_loop_stats;
LoopTimingStats g_cand_update_loop_stats;

LoopTimingStats g_search_layer0_loop_stats;
LoopTimingStats g_search_layerN_loop_stats;

std::vector<int> g_nfilter_stats;
} // namespace

HNSW::HNSW(int dim, int M, int ef_construction, int max_elements, 
    bool use_heuristic, bool extend_candidates, bool keep_pruned, bool use_mkl, size_t mklThreshold)
    : dim_(dim), M_(M), ef_construction_(ef_construction), max_elements_(max_elements), 
      use_heuristic_(use_heuristic), extend_candidates_(extend_candidates), keep_pruned_(keep_pruned), use_mkl_(use_mkl),
      mklThreshold_(mklThreshold), max_level_(0), entry_point_(-1), current_phase_(Phase::Insertion), 
      rng_(42), level_generator_(0.0, 1.0) {
    data_.reserve(max_elements * dim);
    norms_.reserve(max_elements);
    size_neighbor_list_level0_ = static_cast<uint32_t>(2 * M_ + 1);  // count + maxM0 neighbors
    size_neighbor_list_per_element_ = static_cast<uint32_t>(M_ + 1); // count + maxM neighbors
    level0_neighbor_lists_.assign(static_cast<size_t>(max_elements_) * size_neighbor_list_level0_, 0);
    neighbor_lists_.clear();
    neighbor_list_offsets_.assign(max_elements_, std::numeric_limits<uint32_t>::max());
    element_levels_.assign(max_elements_, 0);

    const size_t num_visited_words = (static_cast<size_t>(max_elements_) + 63) / 64;
    visited_bits_.assign(num_visited_words, 0);
    visited_list_.reserve(static_cast<size_t>(max_elements_));

    // Since first inserted element is not searched in layer 0:
    // num_dist_calc_layer0_insertion_.push_back(0);
    // num_cand_elements_layer0_insertion_.push_back(0);
    // max_hops_layer0_insertion_.push_back(0);
}

uint32_t* HNSW::get_neighbor_list0(uint32_t node_id) {
    return level0_neighbor_lists_.data() + static_cast<size_t>(node_id) * size_neighbor_list_level0_;
}

const uint32_t* HNSW::get_neighbor_list0(uint32_t node_id) const {
    return level0_neighbor_lists_.data() + static_cast<size_t>(node_id) * size_neighbor_list_level0_;
}

uint32_t* HNSW::get_neighbor_list(uint32_t node_id, int level) {
    if (level <= 0 || level > element_levels_[node_id]) {
        return nullptr;
    }
    uint32_t base = neighbor_list_offsets_[node_id];
    if (base == std::numeric_limits<uint32_t>::max()) {
        return nullptr;
    }
    return neighbor_lists_.data() + static_cast<size_t>(base) + static_cast<size_t>(level - 1) * size_neighbor_list_per_element_;
}

const uint32_t* HNSW::get_neighbor_list(uint32_t node_id, int level) const {
    if (level <= 0 || level > element_levels_[node_id]) {
        return nullptr;
    }
    uint32_t base = neighbor_list_offsets_[node_id];
    if (base == std::numeric_limits<uint32_t>::max()) {
        return nullptr;
    }
    return neighbor_lists_.data() + static_cast<size_t>(base) + static_cast<size_t>(level - 1) * size_neighbor_list_per_element_;
}

uint32_t* HNSW::get_neighbor_list_at_level(uint32_t node_id, int level) {
    return level == 0 ? get_neighbor_list0(node_id) : get_neighbor_list(node_id, level);
}

const uint32_t* HNSW::get_neighbor_list_at_level(uint32_t node_id, int level) const {
    return level == 0 ? get_neighbor_list0(node_id) : get_neighbor_list(node_id, level);
}

uint32_t HNSW::getListCount(const uint32_t* ptr) const {
    return ptr ? ptr[0] : 0;
}

void HNSW::setListCount(uint32_t* ptr, uint32_t size) {
    if (ptr) {
        ptr[0] = size;
    }
}

std::vector<uint32_t> HNSW::getNeighborsAtLevel(uint32_t node_id, int level) const {
    const uint32_t* ll = get_neighbor_list_at_level(node_id, level);
    if (!ll) {
        return {};
    }
    uint32_t count = getListCount(ll);
    std::vector<uint32_t> neighbors;
    neighbors.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        neighbors.push_back(ll[1 + i]);
    }
    return neighbors;
}

void HNSW::setNeighborsAtLevel(uint32_t node_id, int level, const std::vector<uint32_t>& neighbors, int max_degree) {
    uint32_t* ll = get_neighbor_list_at_level(node_id, level);
    if (!ll) {
        return;
    }
    uint32_t block_size = (level == 0) ? size_neighbor_list_level0_ : size_neighbor_list_per_element_;
    uint32_t capped = std::min<uint32_t>(static_cast<uint32_t>(neighbors.size()), static_cast<uint32_t>(max_degree));
    setListCount(ll, capped);
    for (uint32_t i = 0; i < capped; ++i) {
        ll[1 + i] = neighbors[i];
    }
    for (uint32_t i = capped + 1; i < block_size; ++i) {
        ll[i] = 0;
    }
}

void HNSW::prepareVisited() {
    const size_t num_visited_words = (static_cast<size_t>(max_elements_) + 63) / 64;
    if (visited_bits_.size() < num_visited_words) {
        visited_bits_.resize(num_visited_words, 0);
    }
    visited_list_.clear();
}

bool HNSW::isVisited(uint32_t id) const {
    const size_t word = static_cast<size_t>(id) >> 6;
    const uint64_t mask = 1ULL << (id & 63U);
    return (visited_bits_[word] & mask) != 0;
}

void HNSW::markVisited(uint32_t id) {
    const size_t word = static_cast<size_t>(id) >> 6;
    const uint64_t mask = 1ULL << (id & 63U);
    if ((visited_bits_[word] & mask) == 0) {
        visited_bits_[word] |= mask;
        visited_list_.push_back(id);
    }
}

void HNSW::clearVisited() {
    for (uint32_t id : visited_list_) {
        const size_t word = static_cast<size_t>(id) >> 6;
        const uint64_t mask = 1ULL << (id & 63U);
        visited_bits_[word] &= ~mask;
    }
}

// L2 Euclidean distance
float HNSW::distance(float * a, float * b) const {
    float dist = 0.0f;
    size_t size = dim_;
    size_t i = 0;
    
    // Process as many floats as the available SIMD width allows.
    #ifdef __AVX512F__
    __m512 sum_v = _mm512_setzero_ps();
    constexpr size_t simd_width = 16;
    const size_t simd_end = size - (size % simd_width);
    for (; i < simd_end; i += simd_width) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 diff = _mm512_sub_ps(va, vb);
        sum_v = _mm512_fmadd_ps(diff, diff, sum_v);
    }
    alignas(64) float tmp[16];
    _mm512_storeu_ps(tmp, sum_v);
    for (int j = 0; j < 16; ++j) dist += tmp[j];
    #elif defined(__AVX__)
    __m256 sum_v = _mm256_setzero_ps();
    constexpr size_t simd_width = 8;
    const size_t simd_end = size - (size % simd_width);
    for (; i < simd_end; i += simd_width) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_v = _mm256_fmadd_ps(diff, diff, sum_v);  // fused multiply-add
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, sum_v);
    for (int j = 0; j < 8; ++j) dist += tmp[j];
    #endif
    
    // Remaining elements
    #pragma omp simd reduction(+:dist)
    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;

    // float dist = 0.0f;
    // size_t size = a.size();
    
    // #pragma omp simd reduction(+:dist)
    // for (; i < size; ++i) {
    //     float diff = a[i] - b[i];
    //     dist += diff * diff;
    // }
    // return dist;
}

int HNSW::getRandomLevel() {
    double r = level_generator_(rng_);
    // Ensure r is not too close to 0 to avoid log(0).
    r = std::max(r, std::numeric_limits<double>::min());
    return static_cast<int>(-log(r) * (1.0 / log(M_)));
}

/*
    Using MKL and OMP.
    Batched Cabdidates + Neighbors.
*/
std::priority_queue<std::pair<float, uint32_t>> HNSW::searchLayer(std::vector<float> query, std::vector<uint32_t> entry_points, int ef, int layer) {
    // Use a bitset to track visited nodes so that the eliminate redundant distance calculations and candidate expansions.
    prepareVisited();

    float* query_data = query.data();

    // Calculate q^2 for query vector.
    float query_norm = 0.0f;
    #pragma omp reduction(+:query_norm)
    for (int i = 0; i < dim_; ++i) {
        query_norm += query_data[i] * query_data[i];
    }
    
    MinPQ candidates; // Tracks all candidates to explore, sorted by distance.
    std::priority_queue<std::pair<float, uint32_t>> top_candidates; // Tracks the current top ef candidates found.
    
    // Initialize candidates and top_candidates with entry points, mark them as visited.
    for (uint32_t entry_point : entry_points) {
        float d = distance(query_data, data_.data() + entry_point * dim_);
        candidates.push({d, entry_point});
        top_candidates.push({d, entry_point});
        markVisited(entry_point);
    }

    int thread_count = omp_get_max_threads();
    int batch_size = thread_count * 10;
    std::vector<uint32_t> candidate_batch; // Tracks the current batch of candidates being processed in parallel.
    candidate_batch.reserve(batch_size);

    int max_degree = (layer == 0) ? (2 * M_) : M_; // Maximum degree of neighbors to explore at the current layer.
    int max_neighbors = max_degree * batch_size; // Maximum neighbors to consider in the current batch.
    std::vector<uint32_t> filtered_neighbors; // Tracks the unique neighbors of the current batch of candidates that haven't been visited yet.
    filtered_neighbors.reserve(max_neighbors);

    std::vector<float> neighbor_data; // Buffer to hold neighbor data vectors for distance computation.
    neighbor_data.reserve(max_neighbors * static_cast<size_t>(dim_));

    std::vector<float> dots; // Buffer to hold dot product results of neighbors with the query for distance computation.
    dots.reserve(max_neighbors);

    // If the candidate list is not empty, then explore neighbors of the top candidates.
    while (!candidates.empty()) {
        // If the farthest candidate in the current batch is farther than the closest candidate in the top candidates, we can stop exploring further candidates.
        if (top_candidates.size() >= static_cast<size_t>(ef) &&
            candidates.top().first > top_candidates.top().first) {
            break;
        }

        // Clear the candidate batch and filtered neighbors for the next iteration.
        candidate_batch.clear();
        filtered_neighbors.clear();
        neighbor_data.clear();
        dots.clear();


        auto filter_start = std::chrono::steady_clock::now();
        // Extract a batch of candidates to process in parallel.
        while (!candidates.empty() && candidate_batch.size() < static_cast<size_t>(batch_size)) {
            auto current = candidates.top();
            if (top_candidates.size() >= static_cast<size_t>(ef) &&
                current.first > top_candidates.top().first) {
                break;
            }
            candidate_batch.push_back(current.second);
            candidates.pop();
        }
        
        // For each candidate in the batch, gather their neighbors and filter out those that have already been visited.
        for (uint32_t current_node : candidate_batch) {
            for (uint32_t neighbor_id : getNeighborsAtLevel(current_node, layer)) {
                if (!isVisited(neighbor_id)) {
                    filtered_neighbors.push_back(neighbor_id);
                    markVisited(neighbor_id);
                }
            }
        }

        // Sort the filtered neighbors to improve cache locality in the upcoming distance calculations.
        std::sort(filtered_neighbors.begin(), filtered_neighbors.end());
        auto filter_end = std::chrono::steady_clock::now();

        if (layer == 0 && current_phase_ == Phase::Search) {
            g_nfilter_stats.push_back(static_cast<int>(filtered_neighbors.size()));
            g_filter_loop_stats.add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(filter_end - filter_start).count()));
        }

        std::vector<float> neighbor_dists(filtered_neighbors.size()); // Pre-allocate a vector to store distances to the filtered neighbors.

        // Compute distances from the query to each of the filtered neighbors in parallel. If MKL is enabled and the number of neighbors exceeds the threshold, 
        // use a batched matrix-vector multiplication to compute distances efficiently.
        #if HNSW_HAS_MKL
        if (use_mkl_ && filtered_neighbors.size() >= mklThreshold_) {
            const MKL_INT m = static_cast<MKL_INT>(filtered_neighbors.size());
            const MKL_INT d = static_cast<MKL_INT>(dim_);

            // Copy neighbor vectors into a contiguous buffer for MKL processing.
            auto pack_start = std::chrono::steady_clock::now();
            #pragma omp parallel for
            for (MKL_INT i = 0; i < m; ++i) {
                const float* src = data_.data() + static_cast<size_t>(filtered_neighbors[static_cast<size_t>(i)]) * dim_;
                float* dst = neighbor_data.data() + static_cast<size_t>(i) * dim_;
                memcpy(dst, src, static_cast<size_t>(d) * sizeof(float));
            }
            auto pack_end = std::chrono::steady_clock::now();
            if (layer == 0 && current_phase_ == Phase::Search) {
                g_mkl_pack_loop_stats.add(static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(pack_end - pack_start).count()));
            }
            
            auto gemv_start = std::chrono::steady_clock::now();
            // dots = X * q, where X is m x d neighbor data matrix.
            cblas_sgemv(CblasRowMajor, CblasNoTrans, m, d, 1.0f,
                        neighbor_data.data(), d, query_data, 1, 0.0f,
                        dots.data(), 1);
            auto gemv_end = std::chrono::steady_clock::now();
            if (layer == 0 && current_phase_ == Phase::Search) {
                g_mkl_gemv_loop_stats.add(static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(gemv_end - gemv_start).count()));
            }

            auto distance_calc_start = std::chrono::steady_clock::now();
            #pragma omp parallel for
            for (size_t i = 0; i < filtered_neighbors.size(); ++i) {
                // For distance, dist(X, q) = (x - q)^2 = ||X||^2 + ||q||^2 - 2 * X.q
                neighbor_dists[i] = norms_[filtered_neighbors[static_cast<size_t>(i)]] + query_norm - 2.0f * dots[i];
            }
            auto distance_calc_end = std::chrono::steady_clock::now();
            if (layer == 0 && current_phase_ == Phase::Search) {
                g_distance_loop_stats.add(static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(distance_calc_end - distance_calc_start).count()));
            }
        } else {
        #endif
            // Compute distances from the query to each of the filtered neighbors in parallel using only OMP without MKL.
            auto distance_calc_start = std::chrono::steady_clock::now();
            #pragma omp parallel for
            for (size_t i = 0; i < filtered_neighbors.size(); ++i) {
                uint32_t neighbor_id = filtered_neighbors[i];
                float* neighbor_data = data_.data() + neighbor_id * dim_;
                float dist = distance(query_data, neighbor_data);
                neighbor_dists[i] = dist;
            }
            auto distance_calc_end = std::chrono::steady_clock::now();
            if (layer == 0 && current_phase_ == Phase::Search) {
                g_distance_loop_stats.add(static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(distance_calc_end - distance_calc_start).count()));
            }
        #if HNSW_HAS_MKL
        }
        #endif

        auto cand_update_start = std::chrono::steady_clock::now();

        // Check calculated distances of the neighbors against the current top candidates. 
        // If a neighbor is closer than the farthest candidate in the top ef candidates, 
        // add it to the candidates to explore in the next iteration and also add it to the top candidates.
        for (size_t i = 0; i < filtered_neighbors.size(); ++i) {
            uint32_t neighbor_id = filtered_neighbors[i];
            float dist = neighbor_dists[i];
            if (top_candidates.size() < static_cast<size_t>(ef) || dist < top_candidates.top().first) {
                candidates.push({dist, neighbor_id});
                top_candidates.push({dist, neighbor_id});

                if (top_candidates.size() > static_cast<size_t>(ef)) {
                    top_candidates.pop();
                }
            }
        }
        auto cand_update_end = std::chrono::steady_clock::now();
        if (layer == 0 && current_phase_ == Phase::Search) {
            g_cand_update_loop_stats.add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(cand_update_end - cand_update_start).count()));
        }
    }
    
    clearVisited();

    return top_candidates;
}

/*
    Using MKL and OMP.
    Batched Neighbors.
*/
// std::priority_queue<std::pair<float, uint32_t>> HNSW::searchLayer(std::vector<float> query, std::vector<uint32_t> entry_points, int ef, int layer) {
//     std::vector<bool> visited(max_elements_, false);
//     std::vector<int32_t> hop_counts;
//     if (layer == 0) {
//         hop_counts.assign(max_elements_, -1);
//     }

//     float* query_data = query.data();
//     float query_norm = 0.0f;
//     #pragma omp simd reduction(+:query_norm)
//     for (int i = 0; i < dim_; ++i) {
//         query_norm += query_data[i] * query_data[i];
//     }
    
//     MinPQ candidates;
//     std::priority_queue<std::pair<float, uint32_t>> top_candidates;
    
//     for (uint32_t entry_point : entry_points) {
//         float d = distance(query_data, data_.data() + entry_point * dim_);
//         candidates.push({d, entry_point});
//         top_candidates.push({d, entry_point});
//         visited[entry_point] = true;
//     }

//     int thread_count = omp_get_max_threads();

//     std::vector<uint32_t> filtered_neighbors;
//     filtered_neighbors.reserve(((layer == 0) ? (2 * M_) : M_));
    
//     while (!candidates.empty()) {
//         if (top_candidates.size() >= static_cast<size_t>(ef) &&
//             candidates.top().first > top_candidates.top().first) {
//             break;
//         }

//         filtered_neighbors.clear();

//         auto current = candidates.top();
//         candidates.pop();
        
//         auto loop1_start = std::chrono::steady_clock::now();

//         for (uint32_t neighbor_id : getNeighborsAtLevel(current.second, layer)) {
//                 if (!visited[neighbor_id]) {
//                     filtered_neighbors.push_back(neighbor_id);
//                     visited[neighbor_id] = true;
//                 }
//             }

//         auto loop1_end = std::chrono::steady_clock::now();
//         g_filter_loop_stats.add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(loop1_end - loop1_start).count()));

//         std::vector<float> neighbor_dists(filtered_neighbors.size());

//         auto loop2_start = std::chrono::steady_clock::now();
//         #if HNSW_HAS_MKL
//         if (use_mkl_ && filtered_neighbors.size() >= mklThreshold) {
//             const MKL_INT m = static_cast<MKL_INT>(filtered_neighbors.size());
//             const MKL_INT d = static_cast<MKL_INT>(dim_);

//             std::vector<float> packed_neighbors(static_cast<size_t>(m) * static_cast<size_t>(d));
//             std::vector<float> neighbor_norms(static_cast<size_t>(m), 0.0f);
//             std::vector<float> dots(static_cast<size_t>(m), 0.0f);

//             for (MKL_INT i = 0; i < m; ++i) {
//                 const float* src = data_.data() + static_cast<size_t>(filtered_neighbors[static_cast<size_t>(i)]) * dim_;
//                 float* dst = packed_neighbors.data() + static_cast<size_t>(i) * dim_;
//                 float norm = 0.0f;
//                 #pragma omp simd reduction(+:norm)
//                 for (int j = 0; j < dim_; ++j) {
//                     const float v = src[j];
//                     dst[j] = v;
//                     norm += v * v;
//                 }
//                 neighbor_norms[static_cast<size_t>(i)] = norm;
//             }

//             // dots = A * q, where A is m x d packed neighbor matrix.
//             cblas_sgemv(CblasRowMajor, CblasNoTrans, m, d, 1.0f,
//                         packed_neighbors.data(), d, query_data, 1, 0.0f,
//                         dots.data(), 1);

//             #pragma omp parallel for
//             for (size_t i = 0; i < filtered_neighbors.size(); ++i) {
//                 neighbor_dists[i] = neighbor_norms[i] + query_norm - 2.0f * dots[i];
//             }
//         } else {
//         #endif
//             #pragma omp parallel for
//             for (size_t i = 0; i < filtered_neighbors.size(); ++i) {
//                 uint32_t neighbor_id = filtered_neighbors[i];
//                 float* neighbor_data = data_.data() + neighbor_id * dim_;
//                 float dist = distance(query_data, neighbor_data);
//                 neighbor_dists[i] = dist;
//             }
//         #if HNSW_HAS_MKL
//         }
//         #endif
//         auto loop2_end = std::chrono::steady_clock::now();
//         g_distance_loop_stats.add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(loop2_end - loop2_start).count()));

//         auto loop3_start = std::chrono::steady_clock::now();
//         for (size_t i = 0; i < filtered_neighbors.size(); ++i) {
//             uint32_t neighbor_id = filtered_neighbors[i];
//             float dist = neighbor_dists[i];
//             if (top_candidates.size() < static_cast<size_t>(ef) || dist < top_candidates.top().first) {
//                 candidates.push({dist, neighbor_id});
//                 top_candidates.push({dist, neighbor_id});

//                 if (top_candidates.size() > static_cast<size_t>(ef)) {
//                     top_candidates.pop();
//                 }
//             }
//         }
//         auto loop3_end = std::chrono::steady_clock::now();
//         g_push_loop_stats.add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(loop3_end - loop3_start).count()));
//     }
    
//     return top_candidates;
// }

/*
    Sequential version without MKL.
*/
// std::priority_queue<std::pair<float, uint32_t>> HNSW::searchLayer(std::vector<float> query, std::vector<uint32_t> entry_points, int ef, int layer) {
//     std::vector<bool> visited(max_elements_, false);
//     std::vector<int32_t> hop_counts;
//     if (layer == 0) {
//         hop_counts.assign(max_elements_, -1);
//     }
    
//     MinPQ candidates;
//     std::priority_queue<std::pair<float, uint32_t>> top_candidates;

//     // int32_t dist_calc_count = 0;
//     // int32_t cand_elements_count = 0;
//     // int32_t max_hops = 0;
//     // auto push_metrics = [&](uint32_t dist_count, uint32_t cand_count, uint32_t hop_count) {
//     //     if (current_phase_ == Phase::Insertion) {
//     //         num_dist_calc_layer0_insertion_.push_back(dist_count);
//     //         num_cand_elements_layer0_insertion_.push_back(cand_count);
//     //         max_hops_layer0_insertion_.push_back(hop_count);
//     //     } else {
//     //         num_dist_calc_layer0_search_.push_back(dist_count);
//     //         num_cand_elements_layer0_search_.push_back(cand_count);
//     //         max_hops_layer0_search_.push_back(hop_count);
//     //     }
//     // };
    
//     for (uint32_t entry_point : entry_points) {
//         float d = distance(query.data(), data_.data() + entry_point * dim_);
//         // if (layer == 0) {
//         //     dist_calc_count++;
//         //     hop_counts[entry_point] = 0;
//         // }
//         candidates.push({d, entry_point});
//         // if (layer == 0) {
//         //     cand_elements_count++;
//         // }
//         top_candidates.push({d, entry_point});
//         visited[entry_point] = true;
//     }
    
//     const auto loop_start = std::chrono::steady_clock::now();
//     while (!candidates.empty()) {
//         auto current = candidates.top();
//         candidates.pop();
        
//         // Compare with the farthest in nearest neighbors.
//         if (top_candidates.size() >= static_cast<size_t>(ef) && current.first > top_candidates.top().first) {
//             break;
//         }
        
//         uint32_t current_node = current.second;
//         int32_t current_hops = (layer == 0) ? hop_counts[current_node] : 0;
        
//         const uint32_t* ll = get_neighbor_list_at_level(current_node, layer);
//         if (!ll) {
//             continue;
//         }
//         uint32_t count = getListCount(ll);
//         for (uint32_t i = 0; i < count; ++i) {
//             uint32_t neighbor_id = ll[1 + i];
//             if (!visited[neighbor_id]) {
//                 visited[neighbor_id] = true;
//                 float dist = distance(query.data(), data_.data() + neighbor_id * dim_);

//                 // if (layer == 0) {
//                 //     dist_calc_count++;
//                 //     int32_t neighbor_hops = current_hops + 1;
//                 //     hop_counts[neighbor_id] = neighbor_hops;
//                 //     max_hops = std::max(max_hops, neighbor_hops);
//                 // }

//                 if (top_candidates.size() < static_cast<size_t>(ef) || dist < top_candidates.top().first) {
//                     candidates.push({dist, neighbor_id});
//                     // if (layer == 0) {
//                     //     cand_elements_count++;
//                     // }
//                     top_candidates.push({dist, neighbor_id});

//                     if (top_candidates.size() > static_cast<size_t>(ef)) {
//                         top_candidates.pop();
//                     }
//                 }
//             }
//         }
//     }

//     const auto loop_end = std::chrono::steady_clock::now();
//     const uint64_t loop_ns = static_cast<uint64_t>(
//         std::chrono::duration_cast<std::chrono::nanoseconds>(loop_end - loop_start).count());
//     if (layer == 0) {
//         g_search_layer0_loop_stats.add(loop_ns);
//     } else {
//         g_search_layerN_loop_stats.add(loop_ns);
//     }

//     // if (layer == 0) {
//     //     push_metrics(dist_calc_count, cand_elements_count, max_hops);
//     // }
    
//     return top_candidates;
// }

std::vector<uint32_t> HNSW::selectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M) {

    if (candidates.size() <= static_cast<size_t>(M)) {
        std::vector<uint32_t> result;
        while (!candidates.empty()) {
            result.push_back(candidates.top().second);
            candidates.pop();
        }
        return result;
    }

    std::vector<uint32_t> selected_candidates;
    while (!candidates.empty()) {
        if (static_cast<int>(candidates.size()) > M) {
            candidates.pop();
            continue;
        }
        uint32_t candidate = candidates.top().second;
        candidates.pop();
        selected_candidates.push_back(candidate);
    }
    return selected_candidates;
}

std::vector<uint32_t> HNSW::selectNeighborsHeuristic(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int M, int level) {
    
   if (candidates.size() <= static_cast<size_t>(M)) {
        std::priority_queue<std::pair<float, uint32_t>> temp = candidates;
        std::vector<uint32_t> result;
        while (!temp.empty()) {
            result.push_back(temp.top().second);
            temp.pop();
        }
        return result;
    }

    MinPQ working_set;
    while (!candidates.empty()) {
        working_set.push(candidates.top());
        candidates.pop();
    }
    std::vector<uint32_t> results_set;
    
    if (extend_candidates_) {
        MinPQ temp = working_set;
        while (!temp.empty()) {
            uint32_t candidate = temp.top().second;
            temp.pop();
            for (uint32_t neighbor : getNeighborsAtLevel(candidate, level)) {
                if (neighbor != node_id) { // Ideally should check in working_set if the neighbor is already there.
                    float dist = distance(data_.data() + node_id * dim_, data_.data() + neighbor * dim_);
                    working_set.push({dist, neighbor});
                }
            }
        }
    }

    MinPQ discarded_set;
    while (!working_set.empty() && static_cast<int>(results_set.size()) < M) {
        auto current = working_set.top();
        working_set.pop();
        
        bool good = true;
        for (int result : results_set) {
            float dist = distance(data_.data() + current.second * dim_, data_.data() + result * dim_);
            if (dist < current.first) {
                good = false;
                break;
            }
        }
        
        if (good) {
            results_set.push_back(current.second);
        } else if (keep_pruned_) {
            discarded_set.push(current);
        }
    }

    if (keep_pruned_) {
        while (!discarded_set.empty() && static_cast<int>(results_set.size()) < M) {
            results_set.push_back(discarded_set.top().second);
            discarded_set.pop();
        }
    }

    return results_set;
}

std::vector<uint32_t> HNSW::connectNeighbors(uint32_t node_id, std::priority_queue<std::pair<float, uint32_t>> candidates, int level, int M) {    

    std::vector<uint32_t> selected_neighbors;
    if (use_heuristic_) {
        selected_neighbors = selectNeighborsHeuristic(node_id, candidates, M, level);
    } else {
        selected_neighbors = selectNeighbors(node_id, candidates, M);
    }
    
    setNeighborsAtLevel(node_id, level, selected_neighbors, M);

    int neighbor_max_degree = (level == 0) ? (2 * M_) : M_;

    for (uint32_t neighbor : selected_neighbors) {
        std::vector<uint32_t> neighbor_list = getNeighborsAtLevel(neighbor, level);

        // Add bidirectional connection if needed.
        if (std::find(neighbor_list.begin(), neighbor_list.end(), node_id) == neighbor_list.end()) {
            neighbor_list.push_back(node_id);
            if (static_cast<int>(neighbor_list.size()) > neighbor_max_degree) {
                std::priority_queue<std::pair<float, uint32_t>> econn_candidates;
                for (uint32_t econn_neighbor : neighbor_list) {
                    float dist = distance(data_.data() + neighbor * dim_, data_.data() + econn_neighbor * dim_);
                    econn_candidates.push({dist, econn_neighbor});
                }
                std::vector<uint32_t> reduced_neighbors = use_heuristic_
                    ? selectNeighborsHeuristic(neighbor, econn_candidates, neighbor_max_degree, level)
                    : selectNeighbors(neighbor, econn_candidates, neighbor_max_degree);
                setNeighborsAtLevel(neighbor, level, reduced_neighbors, neighbor_max_degree);
            } else {
                setNeighborsAtLevel(neighbor, level, neighbor_list, neighbor_max_degree);
            }
        }
    }
    return getNeighborsAtLevel(node_id, level);
}

void HNSW::addPoint(std::vector<float> point, uint32_t label) {
    current_phase_ = Phase::Insertion;
    int level = getRandomLevel();

    element_levels_[label] = level;
    if (level > 0) {
        neighbor_list_offsets_[label] = static_cast<uint32_t>(neighbor_lists_.size());
        neighbor_lists_.resize(neighbor_lists_.size() + static_cast<size_t>(level) * size_neighbor_list_per_element_, 0);
    } else {
        neighbor_list_offsets_[label] = std::numeric_limits<uint32_t>::max();
    }
    
    data_.insert(data_.end(), point.begin(), point.end());

    if (entry_point_ == -1) {
        entry_point_ = label;
        max_level_ = level;
        return;
    }
    
    std::vector<uint32_t> entry_points = {entry_point_};
    
    // Search from top layer to target layer
    for (int lc = max_level_; lc > level; --lc) {
        std::priority_queue<std::pair<float, uint32_t>> nearest = searchLayer(point, entry_points, 1, lc);
        if (!nearest.empty()) {
            entry_points.clear();
            while (!nearest.empty()) {
                entry_points.push_back(nearest.top().second);
                nearest.pop();
            }
        }
    }
    
    // Insert at all layers from level to 0
    for (int lc = level; lc >= 0; --lc) {
        int M_max = (lc == 0) ? M_ * 2 : M_;
        std::priority_queue<std::pair<float, uint32_t>> candidates = searchLayer(point, entry_points, ef_construction_, lc);
        if (!candidates.empty()) {
            entry_points.clear();
            std::priority_queue<std::pair<float, uint32_t>> temp = candidates;
            while (!temp.empty()) {
                entry_points.push_back(temp.top().second);
                temp.pop();
            }
        }

        auto neighbors = connectNeighbors(label, candidates, lc, M_);
        for (uint32_t neighbor : neighbors) {
            std::vector<uint32_t> econn = getNeighborsAtLevel(neighbor, lc);
            int neighborhood_size = static_cast<int>(econn.size());
            if (neighborhood_size > M_max) {
                std::priority_queue<std::pair<float, uint32_t>> econn_candidates;
                for (uint32_t econn_neighbor : econn) {
                    float dist = distance(data_.data() + neighbor * dim_, data_.data() + econn_neighbor * dim_);
                    econn_candidates.push({dist, econn_neighbor});
                }
                connectNeighbors(neighbor, econn_candidates, lc, M_max);
            }
        }
    }
    
    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = label;
    }

    float norm = 0.0f;
    #pragma omp reduction(+:norm)
    for (float v : point) {
        norm += v * v;
    }
    norms_.push_back(norm);
}

std::priority_queue<std::pair<float, uint32_t>> HNSW::searchKNN(std::vector<float> query, int k, int ef) {
    if (entry_point_ == -1) {
        return {};
    }

    current_phase_ = Phase::Search;

    std::vector<uint32_t> entry_points = {entry_point_};

    // Search from top layer to layer 0
    for (int lc = max_level_; lc > 0; --lc) {
        std::priority_queue<std::pair<float, uint32_t>> nearest = searchLayer(query, entry_points, 1, lc);
        if (!nearest.empty()) {
            entry_points.clear();
            while (!nearest.empty()) {
                entry_points.push_back(nearest.top().second);
                nearest.pop();
            }
        }
    }
    
    // Search at layer 0
    return searchLayer(query, entry_points, std::max(ef, k), 0);
}

void HNSW::setLabelRemapping(std::vector<uint32_t> old_to_new, std::vector<uint32_t> new_to_old) {
    old_to_new_labels_ = std::move(old_to_new);
    new_to_old_labels_ = std::move(new_to_old);
}


void HNSW::relabelGroundTruth(std::vector<std::vector<uint32_t>>& groundtruth) const {
    if (old_to_new_labels_.empty()) {
        return;
    }

    for (size_t i = 0; i < groundtruth.size(); ++i) {
        std::vector<uint32_t>& labels = groundtruth[i];
        for (size_t j = 0; j < labels.size(); ++j) {
            uint32_t label = labels[j];
            if (label < old_to_new_labels_.size()) {
                labels[j] = old_to_new_labels_[label];
            }
        }
    }
}

void HNSW::printInfo(const std::string& timing_csv_path) const {
    std::cout << "\nHNSW Index Info:\n";
    std::cout << "Dimension: " << dim_ << "\n";
    std::cout << "M (max connections per layer): " << M_ << "\n";
    std::cout << "ef_construction: " << ef_construction_ << "\n";
    std::cout << "Max elements: " << max_elements_ << "\n";
    std::cout << "Current number of nodes: " << data_.size() / dim_ << "\n";
    std::cout << "Max level: " << max_level_ << "\n";
    std::cout << "Entry point ID: " << entry_point_ << "\n";

    std::vector<int> layer_counts;
    for (int level_id = 0; level_id <= max_level_; ++level_id) {
        int count = 0;
        size_t num_nodes = data_.size() / dim_;
        for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
            if (!getNeighborsAtLevel(static_cast<uint32_t>(node_id), level_id).empty()) {
                count++;
            }
        }
        layer_counts.push_back(count);
    }
    for (size_t i = 0; i < layer_counts.size(); ++i) {
        std::cout << "Layer " << i << " has " << layer_counts[i] << " nodes\n";
    }

    // std::cout << "\nAverage runtime per searchLayer call:\n";
    // std::cout << "Layer 0: " << g_search_layer0_loop_stats.average_us() << " us over " << g_search_layer0_loop_stats.calls << " calls\n";   
    // std::cout << "Other Layers: " << g_search_layerN_loop_stats.average_us() << " us over " << g_search_layerN_loop_stats.calls << " calls\n";

    std::cout << "\nMemcpy timing:\n";
    std::cout << "MKL pack memcpy: " << g_mkl_pack_loop_stats.average_us() << " us over " << g_mkl_pack_loop_stats.count() << " calls\n";

    std::cout << "\nAverage runtime per selected loop:\n";
    std::cout << "Neighbor Filter: " << g_filter_loop_stats.average_us()
              << " us over " << g_filter_loop_stats.count() << " calls\n";
    std::cout << "MKL pack memcpy: " << g_mkl_pack_loop_stats.average_us() 
              << " us over " << g_mkl_pack_loop_stats.count() << " calls\n";
    std::cout << "MKL gemv: " << g_mkl_gemv_loop_stats.average_us()
              << " us over " << g_mkl_gemv_loop_stats.count() << " calls\n";
    std::cout << "Distance Compute: " << g_distance_loop_stats.average_us()
              << " us over " << g_distance_loop_stats.count() << " calls\n";
    std::cout << "Candidate Update: " << g_cand_update_loop_stats.average_us()
              << " us over " << g_cand_update_loop_stats.count() << " calls\n";

    std::ofstream timing_csv(timing_csv_path);
    if (timing_csv) {
        timing_csv << "nfilter,t1,t2,t3,t4,t5,label_mad\n";
        const size_t max_samples = std::max({
            g_nfilter_stats.size(),
            g_filter_loop_stats.count(),
            g_mkl_pack_loop_stats.count(),
            g_mkl_gemv_loop_stats.count(),
            g_distance_loop_stats.count(),
            g_cand_update_loop_stats.count(),
        });

        for (size_t i = 0; i < max_samples; ++i) {

            if (i < g_nfilter_stats.size()) {
                timing_csv << g_nfilter_stats[i];
            } else {
                timing_csv << "0";
            }
            timing_csv << ",";
            if (i < g_filter_loop_stats.count()) {
                timing_csv << g_filter_loop_stats.sample_us(i);
            } else {
                timing_csv << "0";
            }
            timing_csv << ",";
            if (i < g_mkl_pack_loop_stats.count()) {
                timing_csv << g_mkl_pack_loop_stats.sample_us(i);
            } else {
                timing_csv << "0";
            }
            timing_csv << ",";
            if (i < g_mkl_gemv_loop_stats.count()) {
                timing_csv << g_mkl_gemv_loop_stats.sample_us(i);
            } else {
                timing_csv << "0";
            }
            timing_csv << ",";
            if (i < g_distance_loop_stats.count()) {
                timing_csv << g_distance_loop_stats.sample_us(i);
            } else {
                timing_csv << "0";
            }
            timing_csv << ",";
            if (i < g_cand_update_loop_stats.count()) {
                timing_csv << g_cand_update_loop_stats.sample_us(i);
            } else {
                timing_csv << "0";
            }
            timing_csv << "\n";
        }
    }
}

// bool HNSW::dumpLayer0Counts(const std::string& output_path, const std::string param) const {
//     std::ofstream out(output_path);
//     if (!out) {
//         return false;
//     }

//     if (param == "dist_calc_insertion") {
//         for (uint32_t count : num_dist_calc_layer0_insertion_) {
//             out << count << "\n";
//         }
//     } else if (param == "cand_elements_insertion") {
//         for (uint32_t count : num_cand_elements_layer0_insertion_) {
//             out << count << "\n";
//         }
//     } else if (param == "max_hops_insertion") {
//         for (uint32_t count : max_hops_layer0_insertion_) {
//             out << count << "\n";
//         }
//     } else if (param == "dist_calc_search") {
//         for (uint32_t count : num_dist_calc_layer0_search_) {
//             out << count << "\n";
//         }
//     } else if (param == "cand_elements_search") {
//         for (uint32_t count : num_cand_elements_layer0_search_) {
//             out << count << "\n";
//         }
//     } else if (param == "max_hops_search") {
//         for (uint32_t count : max_hops_layer0_search_) {
//             out << count << "\n";
//         }
//     } else {
//         return false; // Invalid parameter
//     }
//     return true;
// }
