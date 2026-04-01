#include "hnsw.h"
#include <iostream>
#include <immintrin.h>
#include <unordered_set>
#include <omp.h>

using namespace hnsw;

HNSW::HNSW(int dim, int M, int ef_construction, int max_elements, 
    bool use_heuristic, bool extend_candidates, bool keep_pruned)
    : dim_(dim), M_(M), ef_construction_(ef_construction), max_elements_(max_elements), 
      use_heuristic_(use_heuristic), extend_candidates_(extend_candidates), keep_pruned_(keep_pruned),
      max_level_(0), entry_point_(-1), rng_(42), level_generator_(0.0, 1.0) {
    data_.reserve(max_elements * dim);
    size_neighbor_list_level0_ = static_cast<uint32_t>(2 * M_ + 1);  // count + maxM0 neighbors
    size_neighbor_list_per_element_ = static_cast<uint32_t>(M_ + 1); // count + maxM neighbors
    level0_neighbor_lists_.assign(static_cast<size_t>(max_elements_) * size_neighbor_list_level0_, 0);
    neighbor_lists_.clear();
    neighbor_list_offsets_.assign(max_elements_, std::numeric_limits<uint32_t>::max());
    element_levels_.assign(max_elements_, 0);
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
    
    MinPQ candidates;
    std::priority_queue<std::pair<float, uint32_t>> top_candidates;
    
    for (uint32_t entry_point : entry_points) {
        float d = distance(query.data(), data_.data() + entry_point * dim_);
        candidates.push({d, entry_point});
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
        for (uint32_t neighbor_id : getNeighborsAtLevel(current_node, layer)) {
            if (!visited[neighbor_id]) {
                visited[neighbor_id] = true;
                float dist = distance(query.data(), data_.data() + neighbor_id * dim_);

                if (top_candidates.size() < static_cast<size_t>(ef) || dist < top_candidates.top().first) {
                    candidates.push({dist, neighbor_id});
                    top_candidates.push({dist, neighbor_id});

                    if (top_candidates.size() > static_cast<size_t>(ef)) {
                        top_candidates.pop();
                    }
                }
            }
        }
    }
    
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
