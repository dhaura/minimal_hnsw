#include "hnsw.h"
#include <iostream>
#include <fstream>
#include <immintrin.h>
#include <unordered_set>
#include <omp.h>

using namespace hnsw;

HNSW::HNSW(int dim, int M, int ef_construction, int max_elements, 
    bool use_heuristic, bool extend_candidates, bool keep_pruned)
    : dim_(dim), M_(M), ef_construction_(ef_construction), max_elements_(max_elements), 
      use_heuristic_(use_heuristic), extend_candidates_(extend_candidates), keep_pruned_(keep_pruned),
        max_level_(0), entry_point_(-1), pca_projection_ready_(false), current_phase_(Phase::Insertion), rng_(42), level_generator_(0.0, 1.0) {
    data_.reserve(max_elements * dim);
    size_neighbor_list_level0_ = static_cast<uint32_t>(2 * M_ + 1);  // count + maxM0 neighbors
    size_neighbor_list_per_element_ = static_cast<uint32_t>(M_ + 1); // count + maxM neighbors
    level0_neighbor_lists_.assign(static_cast<size_t>(max_elements_) * size_neighbor_list_level0_, 0);
    neighbor_lists_.clear();
    neighbor_list_offsets_.assign(max_elements_, std::numeric_limits<uint32_t>::max());
    element_levels_.assign(max_elements_, 0);

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

void HNSW::fitPcaProjection() {
    pca_projection_ready_ = false;
    pca_mean_.assign(static_cast<size_t>(dim_), 0.0f);
    pca_component_1_.assign(static_cast<size_t>(dim_), 0.0f);
    pca_component_2_.assign(static_cast<size_t>(dim_), 0.0f);

    const size_t num_nodes = (dim_ > 0) ? (data_.size() / static_cast<size_t>(dim_)) : 0;
    if (num_nodes == 0 || dim_ <= 0) {
        return;
    }

    if (dim_ == 1) {
        pca_component_1_[0] = 1.0f;
        pca_projection_ready_ = true;
        return;
    }

    for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
        const float* point = data_.data() + node_id * static_cast<size_t>(dim_);
        for (int dim_id = 0; dim_id < dim_; ++dim_id) {
            pca_mean_[static_cast<size_t>(dim_id)] += point[dim_id];
        }
    }
    const float inv_num_nodes = 1.0f / static_cast<float>(num_nodes);
    for (int dim_id = 0; dim_id < dim_; ++dim_id) {
        pca_mean_[static_cast<size_t>(dim_id)] *= inv_num_nodes;
    }

    std::vector<double> covariance(static_cast<size_t>(dim_) * static_cast<size_t>(dim_), 0.0);
    for (size_t node_id = 0; node_id < num_nodes; ++node_id) {
        const float* point = data_.data() + node_id * static_cast<size_t>(dim_);
        for (int row = 0; row < dim_; ++row) {
            const double centered_row = static_cast<double>(point[row]) - static_cast<double>(pca_mean_[static_cast<size_t>(row)]);
            for (int col = row; col < dim_; ++col) {
                const double centered_col = static_cast<double>(point[col]) - static_cast<double>(pca_mean_[static_cast<size_t>(col)]);
                covariance[static_cast<size_t>(row) * static_cast<size_t>(dim_) + static_cast<size_t>(col)] += centered_row * centered_col;
            }
        }
    }

    const double inv_denom = (num_nodes > 1) ? (1.0 / static_cast<double>(num_nodes - 1)) : 1.0;
    for (int row = 0; row < dim_; ++row) {
        for (int col = row; col < dim_; ++col) {
            const double value = covariance[static_cast<size_t>(row) * static_cast<size_t>(dim_) + static_cast<size_t>(col)] * inv_denom;
            covariance[static_cast<size_t>(row) * static_cast<size_t>(dim_) + static_cast<size_t>(col)] = value;
            covariance[static_cast<size_t>(col) * static_cast<size_t>(dim_) + static_cast<size_t>(row)] = value;
        }
    }

    auto multiplyCovariance = [&](const std::vector<float>& vector) {
        std::vector<double> result(static_cast<size_t>(dim_), 0.0);
        for (int row = 0; row < dim_; ++row) {
            double sum = 0.0;
            for (int col = 0; col < dim_; ++col) {
                sum += covariance[static_cast<size_t>(row) * static_cast<size_t>(dim_) + static_cast<size_t>(col)] * static_cast<double>(vector[static_cast<size_t>(col)]);
            }
            result[static_cast<size_t>(row)] = sum;
        }
        return result;
    };

    auto normalizeVector = [](std::vector<float>& vector) {
        double norm_sq = 0.0;
        for (float value : vector) {
            norm_sq += static_cast<double>(value) * static_cast<double>(value);
        }
        if (norm_sq <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        const float inv_norm = static_cast<float>(1.0 / std::sqrt(norm_sq));
        for (float& value : vector) {
            value *= inv_norm;
        }
        return true;
    };

    auto computeComponent = [&](const std::vector<std::vector<float>>& basis) {
        std::vector<float> vector(static_cast<size_t>(dim_), 0.0f);
        vector[0] = 1.0f;
        if (!basis.empty()) {
            for (int dim_id = 0; dim_id < dim_; ++dim_id) {
                vector[static_cast<size_t>(dim_id)] = static_cast<float>((dim_id == 0) ? 0.5f : 0.0f);
            }
        }

        if (!normalizeVector(vector)) {
            vector.assign(static_cast<size_t>(dim_), 0.0f);
            vector[0] = 1.0f;
        }

        for (int iteration = 0; iteration < 64; ++iteration) {
            std::vector<double> next_double = multiplyCovariance(vector);
            for (const std::vector<float>& prior : basis) {
                double dot = 0.0;
                for (int dim_id = 0; dim_id < dim_; ++dim_id) {
                    dot += next_double[static_cast<size_t>(dim_id)] * static_cast<double>(prior[static_cast<size_t>(dim_id)]);
                }
                for (int dim_id = 0; dim_id < dim_; ++dim_id) {
                    next_double[static_cast<size_t>(dim_id)] -= dot * static_cast<double>(prior[static_cast<size_t>(dim_id)]);
                }
            }

            std::vector<float> next(static_cast<size_t>(dim_), 0.0f);
            for (int dim_id = 0; dim_id < dim_; ++dim_id) {
                next[static_cast<size_t>(dim_id)] = static_cast<float>(next_double[static_cast<size_t>(dim_id)]);
            }

            if (!normalizeVector(next)) {
                break;
            }

            double diff_sq = 0.0;
            for (int dim_id = 0; dim_id < dim_; ++dim_id) {
                const double delta = static_cast<double>(next[static_cast<size_t>(dim_id)]) - static_cast<double>(vector[static_cast<size_t>(dim_id)]);
                diff_sq += delta * delta;
            }
            vector = std::move(next);
            if (diff_sq <= 1e-10) {
                break;
            }
        }

        return vector;
    };

    pca_component_1_ = computeComponent({});
    if (!normalizeVector(pca_component_1_)) {
        pca_component_1_.assign(static_cast<size_t>(dim_), 0.0f);
        pca_component_1_[0] = 1.0f;
    }

    std::vector<std::vector<float>> basis;
    basis.push_back(pca_component_1_);
    pca_component_2_ = computeComponent(basis);
    if (!normalizeVector(pca_component_2_)) {
        pca_component_2_.assign(static_cast<size_t>(dim_), 0.0f);
        pca_component_2_[1] = 1.0f;
    }

    double dot = 0.0;
    for (int dim_id = 0; dim_id < dim_; ++dim_id) {
        dot += static_cast<double>(pca_component_1_[static_cast<size_t>(dim_id)]) * static_cast<double>(pca_component_2_[static_cast<size_t>(dim_id)]);
    }
    for (int dim_id = 0; dim_id < dim_; ++dim_id) {
        pca_component_2_[static_cast<size_t>(dim_id)] -= static_cast<float>(dot * static_cast<double>(pca_component_1_[static_cast<size_t>(dim_id)]));
    }
    if (!normalizeVector(pca_component_2_)) {
        pca_component_2_.assign(static_cast<size_t>(dim_), 0.0f);
        pca_component_2_[1] = 1.0f;
    }

    pca_projection_ready_ = true;
}

// L2 Euclidean distance
float HNSW::distance(float * a, float * b) const {
    float dist = 0.0f;
    size_t size = dim_;
    size_t i = 0;
    
    // Process 8 floats at a time with AVX
    #ifdef __AVX__
    __m256 sum_v = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_v = _mm256_fmadd_ps(diff, diff, sum_v);  // fused multiply-add
    }
    // Horizontal sum
    float tmp[8];
    _mm256_storeu_ps(tmp, sum_v);
    for (int j = 0; j < 8; ++j) dist += tmp[j];
    #endif
    
    // Remaining elements
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

std::priority_queue<std::pair<float, uint32_t>> HNSW::searchLayer(std::vector<float> query, std::vector<uint32_t> entry_points, int ef, int layer) {
    std::vector<bool> visited(max_elements_, false);
    std::vector<int32_t> hop_counts;
    if (layer == 0) {
        hop_counts.assign(max_elements_, -1);
    }
    
    MinPQ candidates;
    std::priority_queue<std::pair<float, uint32_t>> top_candidates;

    // int32_t dist_calc_count = 0;
    // int32_t cand_elements_count = 0;
    // int32_t max_hops = 0;
    // auto push_metrics = [&](uint32_t dist_count, uint32_t cand_count, uint32_t hop_count) {
    //     if (current_phase_ == Phase::Insertion) {
    //         num_dist_calc_layer0_insertion_.push_back(dist_count);
    //         num_cand_elements_layer0_insertion_.push_back(cand_count);
    //         max_hops_layer0_insertion_.push_back(hop_count);
    //     } else {
    //         num_dist_calc_layer0_search_.push_back(dist_count);
    //         num_cand_elements_layer0_search_.push_back(cand_count);
    //         max_hops_layer0_search_.push_back(hop_count);
    //     }
    // };
    
    for (uint32_t entry_point : entry_points) {
        float d = distance(query.data(), data_.data() + entry_point * dim_);
        // if (layer == 0) {
        //     dist_calc_count++;
        //     hop_counts[entry_point] = 0;
        // }
        candidates.push({d, entry_point});
        // if (layer == 0) {
        //     cand_elements_count++;
        // }
        top_candidates.push({d, entry_point});
        visited[entry_point] = true;
    }
    
    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();
        
        // Compare with the farthest in nearest neighbors.
        if (top_candidates.size() >= static_cast<size_t>(ef) && current.first > top_candidates.top().first) {
            break;
        }
        
        uint32_t current_node = current.second;
        int32_t current_hops = (layer == 0) ? hop_counts[current_node] : 0;
        for (uint32_t neighbor_id : getNeighborsAtLevel(current_node, layer)) {
            if (!visited[neighbor_id]) {
                visited[neighbor_id] = true;
                float dist = distance(query.data(), data_.data() + neighbor_id * dim_);

                // if (layer == 0) {
                //     dist_calc_count++;
                //     int32_t neighbor_hops = current_hops + 1;
                //     hop_counts[neighbor_id] = neighbor_hops;
                //     max_hops = std::max(max_hops, neighbor_hops);
                // }

                if (top_candidates.size() < static_cast<size_t>(ef) || dist < top_candidates.top().first) {
                    candidates.push({dist, neighbor_id});
                    // if (layer == 0) {
                    //     cand_elements_count++;
                    // }
                    top_candidates.push({dist, neighbor_id});

                    if (top_candidates.size() > static_cast<size_t>(ef)) {
                        top_candidates.pop();
                    }
                }
            }
        }
    }

    // if (layer == 0) {
    //     push_metrics(dist_calc_count, cand_elements_count, max_hops);
    // }
    
    return top_candidates;
}

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
    pca_projection_ready_ = false;
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

// Helper function to convert coordinates to Hilbert index
// For 2D case with normalized coordinates [0, 1]
uint64_t HNSW::xy2d(uint32_t n, uint32_t x, uint32_t y) {
    uint64_t rx, ry, s, d = 0;
    for (s = n / 2; s > 0; s /= 2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        // Rotate
        if (ry == 0) {
            if (rx == 1) {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}

// Convert multi-dimensional point to 2D using PCA projection
std::pair<float, float> HNSW::projectTo2D(const std::vector<float>& point) const {
    if (!pca_projection_ready_ || static_cast<int>(pca_mean_.size()) != dim_ ||
        static_cast<int>(pca_component_1_.size()) != dim_ || static_cast<int>(pca_component_2_.size()) != dim_) {
        const float x = (point.size() > 0) ? point[0] : 0.0f;
        const float y = (point.size() > 1) ? point[1] : 0.0f;
        return std::make_pair(x, y);
    }

    double x = 0.0;
    double y = 0.0;
    for (int dim_id = 0; dim_id < dim_; ++dim_id) {
        const double centered = static_cast<double>(point[static_cast<size_t>(dim_id)]) - static_cast<double>(pca_mean_[static_cast<size_t>(dim_id)]);
        x += centered * static_cast<double>(pca_component_1_[static_cast<size_t>(dim_id)]);
        y += centered * static_cast<double>(pca_component_2_[static_cast<size_t>(dim_id)]);
    }
    return std::make_pair(static_cast<float>(x), static_cast<float>(y));
}

// Compute Hilbert index for a point
uint64_t HNSW::computeHilbertIndex(const std::vector<float>& point, 
                                  float min_x, float max_x, 
                                  float min_y, float max_y) {
    std::pair<float, float> proj = projectTo2D(point);
    float x = proj.first;
    float y = proj.second;
    
    // Normalize to [0, 1]
    float range_x = max_x - min_x;
    float range_y = max_y - min_y;
    if (range_x < 1e-6f) range_x = 1.0f;
    if (range_y < 1e-6f) range_y = 1.0f;
    
    float norm_x = (x - min_x) / range_x;
    float norm_y = (y - min_y) / range_y;
    
    // Clamp to [0, 1]
    norm_x = std::max(0.0f, std::min(1.0f, norm_x));
    norm_y = std::max(0.0f, std::min(1.0f, norm_y));
    
    // Scale to grid coordinates (use 16-bit grid for good resolution)
    uint32_t grid_size = 65536;  // 2^16
    uint32_t grid_x = static_cast<uint32_t>(norm_x * (grid_size - 1));
    uint32_t grid_y = static_cast<uint32_t>(norm_y * (grid_size - 1));
    
    return xy2d(grid_size, grid_x, grid_y);
}

void HNSW::finalizeIndex() {
    // Apply Hilbert curve ordering to improve cache locality
    size_t num_nodes = data_.size() / dim_;
    if (num_nodes <= 1) {
        return;  // Nothing to optimize
    }

    fitPcaProjection();
    
    // Compute bounds for normalization
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    
    for (size_t i = 0; i < num_nodes; ++i) {
        std::vector<float> point(data_.begin() + i * dim_,
                                 data_.begin() + (i + 1) * dim_);
        std::pair<float, float> proj = projectTo2D(point);
        float x = proj.first;
        float y = proj.second;
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }
    
    // Compute Hilbert indices for all points
    std::vector<std::pair<uint64_t, uint32_t>> hilbert_indices;
    hilbert_indices.reserve(num_nodes);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        std::vector<float> point(data_.begin() + i * dim_,
                                 data_.begin() + (i + 1) * dim_);
        uint64_t h_idx = computeHilbertIndex(point, min_x, max_x, min_y, max_y);
        hilbert_indices.push_back(std::make_pair(h_idx, static_cast<uint32_t>(i)));
    }
    
    // Sort by Hilbert index
    std::sort(hilbert_indices.begin(), hilbert_indices.end());
    
    // Create mapping from old IDs to new IDs
    std::vector<uint32_t> old_to_new(num_nodes);
    std::vector<uint32_t> new_to_old(num_nodes);
    for (uint32_t new_id = 0; new_id < num_nodes; ++new_id) {
        uint32_t old_id = hilbert_indices[new_id].second;
        old_to_new[old_id] = new_id;
        new_to_old[new_id] = old_id;
    }
    old_to_new_labels_ = old_to_new;
    new_to_old_labels_ = new_to_old;
    
    // Reorder data vector
    std::vector<float> new_data;
    new_data.reserve(data_.size());
    for (uint32_t new_id = 0; new_id < num_nodes; ++new_id) {
        uint32_t old_id = new_to_old[new_id];
        new_data.insert(new_data.end(),
                       data_.begin() + old_id * dim_,
                       data_.begin() + (old_id + 1) * dim_);
    }
    data_ = new_data;
    
    // Update element_levels mapping
    std::vector<int> new_element_levels(num_nodes);
    for (uint32_t old_id = 0; old_id < num_nodes; ++old_id) {
        uint32_t new_id = old_to_new[old_id];
        new_element_levels[new_id] = element_levels_[old_id];
    }
    element_levels_ = new_element_levels;
    
    // Update neighbor list offsets
    std::vector<uint32_t> new_neighbor_list_offsets(num_nodes, std::numeric_limits<uint32_t>::max());
    for (uint32_t old_id = 0; old_id < num_nodes; ++old_id) {
        uint32_t new_id = old_to_new[old_id];
        new_neighbor_list_offsets[new_id] = neighbor_list_offsets_[old_id];
    }
    neighbor_list_offsets_ = new_neighbor_list_offsets;
    
    // Update level 0 neighbor lists with new IDs
    std::vector<uint32_t> new_level0_neighbor_lists(level0_neighbor_lists_.size(), 0);
    for (uint32_t old_id = 0; old_id < num_nodes; ++old_id) {
        uint32_t new_id = old_to_new[old_id];
        
        const uint32_t* old_list = level0_neighbor_lists_.data() + old_id * size_neighbor_list_level0_;
        uint32_t* new_list = new_level0_neighbor_lists.data() + new_id * size_neighbor_list_level0_;
        
        uint32_t neighbor_count = old_list[0];
        new_list[0] = neighbor_count;
        
        for (uint32_t i = 0; i < neighbor_count; ++i) {
            uint32_t old_neighbor = old_list[1 + i];
            uint32_t new_neighbor = old_to_new[old_neighbor];
            new_list[1 + i] = new_neighbor;
        }
    }
    level0_neighbor_lists_ = new_level0_neighbor_lists;
    
    // Update upper layer neighbor lists
    for (uint32_t old_id = 0; old_id < num_nodes; ++old_id) {
        uint32_t offset = neighbor_list_offsets_[old_to_new[old_id]];
        if (offset != std::numeric_limits<uint32_t>::max()) {
            int level = element_levels_[old_to_new[old_id]];
            for (int lv = 1; lv <= level; ++lv) {
                uint32_t* neighbor_list = neighbor_lists_.data() + offset + (lv - 1) * size_neighbor_list_per_element_;
                uint32_t neighbor_count = neighbor_list[0];
                
                for (uint32_t i = 0; i < neighbor_count; ++i) {
                    uint32_t old_neighbor = neighbor_list[1 + i];
                    uint32_t new_neighbor = old_to_new[old_neighbor];
                    neighbor_list[1 + i] = new_neighbor;
                }
            }
        }
    }
    
    // Update entry point
    entry_point_ = old_to_new[entry_point_];
    
    std::cout << "Hilbert curve optimization applied: " << num_nodes << " points reordered\n";
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

void HNSW::printInfo() const {
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
