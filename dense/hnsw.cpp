#include "hnsw.h"

HNSW::HNSW(int dim, int M, int ef_construction, int max_elements)
    : dim_(dim), M_(M), ef_construction_(ef_construction), max_elements_(max_elements),
      max_level_(0), entry_point_(-1), rng_(42), level_generator_(0.0, 1.0) {
    nodes_.reserve(max_elements);
}

// L2 distance
float HNSW::distance(const std::vector<float>& a, const std::vector<float>& b) const {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

int HNSW::getRandomLevel() {
    double r = level_generator_(rng_);
    // Ensure r is not too close to 0 to avoid log(0)
    r = std::max(r, std::numeric_limits<double>::min());
    return static_cast<int>(-log(r) * (1.0 / log(1.0 * M_)));
}

std::vector<int> HNSW::searchLayer(const std::vector<float>& query, std::vector<int> entry_points, int ef, int layer) {
    std::unordered_set<int> visited;
    
    auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first < b.first;
    };
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp)> candidates(cmp);
    std::priority_queue<std::pair<float, int>> nearest;
    
    for (int entry_point : entry_points) {
        float d = distance(query, nodes_[entry_point].data);
        candidates.push({d, entry_point});
        nearest.push({d, entry_point});
        visited.insert(entry_point);
    }
    
    while (!candidates.empty()) {
        auto current = candidates.top();
        candidates.pop();
        
        // Compare with the farthest in nearest neighbors.
        if (nearest.size() >= static_cast<size_t>(ef) && current.first > nearest.top().first) {
            break;
        }
        
        int current_node = current.second;
        // Ideally this check shouldn't  fail.
        if (layer < static_cast<int>(nodes_[current_node].neighbors.size())) {
            for (int neighbor_id : nodes_[current_node].neighbors[layer]) {
                if (visited.find(neighbor_id) == visited.end()) {
                    visited.insert(neighbor_id);
                    float dist = distance(query, nodes_[neighbor_id].data);
                    
                    if (nearest.size() < static_cast<size_t>(ef) || dist < nearest.top().first) {
                        candidates.push({dist, neighbor_id});
                        nearest.push({dist, neighbor_id});
                        
                        if (nearest.size() > static_cast<size_t>(ef)) {
                            nearest.pop();
                        }
                    }
                }
            }
        }
    }
    
    std::vector<int> result;
    while (!nearest.empty()) {
        result.push_back(nearest.top().second);
        nearest.pop();
    }

    // Reverse to have closest first in a sorted manner
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<int> HNSW::connectNeighbors(int node_id, const std::vector<int>& candidates, int level, int M) {    
    
    if (static_cast<int>(nodes_[node_id].neighbors.size()) <= level) {
        nodes_[node_id].neighbors.resize(level + 1);
    }

    // Clear existing neighbors at this level
    nodes_[node_id].neighbors[level].clear();
    
    // Candidates list is sorted according to distance
    for (int candidate : candidates) {
        if (static_cast<int>(nodes_[node_id].neighbors[level].size()) >= M) {
            break;
        }
        nodes_[node_id].neighbors[level].push_back(candidate);
        
        // Add bidirectional connection
        if (static_cast<int>(nodes_[candidate].neighbors.size()) <= level) {
            nodes_[candidate].neighbors.resize(level + 1);
        }
        if (std::find(nodes_[candidate].neighbors[level].begin(), nodes_[candidate].neighbors[level].end(), node_id) == nodes_[candidate].neighbors[level].end()) {
            nodes_[candidate].neighbors[level].push_back(node_id);
        }
    }
    return nodes_[node_id].neighbors[level];
}

std::vector<int> HNSW::sortCandidates(int node_id, const std::vector<int>& candidates) {
    std::vector<std::pair<float, int>> dists;
    for (int candidate : candidates) {
        float dist = distance(nodes_[node_id].data, nodes_[candidate].data);
        dists.push_back({dist, candidate});
    }
    std::sort(dists.begin(), dists.end());
    
    std::vector<int> sorted_candidates;
    for (const auto& pair : dists) {
        sorted_candidates.push_back(pair.second);
    }
    return sorted_candidates;
}

void HNSW::addPoint(const std::vector<float>& point, int label) {
    Node new_node;
    new_node.id = label;
    new_node.data = point;
    
    int node_id = nodes_.size();
    int level = getRandomLevel();
    
    new_node.neighbors.resize(level + 1);
    nodes_.push_back(new_node);
    
    if (entry_point_ == -1) {
        entry_point_ = node_id;
        max_level_ = level;
        return;
    }
    
    std::vector<int> enter_points = {entry_point_};
    
    // Search from top layer to target layer
    for (int lc = max_level_; lc > level; --lc) {
        auto nearest = searchLayer(point, enter_points, 1, lc);
        if (!nearest.empty()) {
            enter_points = nearest;
        }
    }
    
    // Insert at all layers from level to 0
    for (int lc = level; lc >= 0; --lc) {
        int M_max = (lc == 0) ? M_ * 2 : M_;
        auto candidates = searchLayer(point, enter_points, ef_construction_, lc);
        auto neighbors = connectNeighbors(node_id, candidates, lc, M_);
        for (int neighbor : neighbors) {
            int neighborhood_size = static_cast<int>(nodes_[neighbor].neighbors[lc].size());
            if (neighborhood_size > M_max) {
                std::vector<int> neighbor_candidates;
                neighbor_candidates.push_back(node_id);
                for (int n : nodes_[neighbor].neighbors[lc]) {
                    if (n != node_id) {
                        neighbor_candidates.push_back(n);
                    }
                }
                auto sorted_neighbor_candidates = sortCandidates(neighbor, neighbor_candidates);
                auto pruned_neighbors = connectNeighbors(neighbor, sorted_neighbor_candidates, lc, M_max);
            }
        }
        if (!candidates.empty()) {
            enter_points = candidates;
        }
    }
    
    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = node_id;
    }
}

std::vector<std::pair<int, float>> HNSW::searchKNN(const std::vector<float>& query, int k, int ef) {
    if (entry_point_ == -1) {
        return {};
    }
    
    std::vector<int> enter_points = {entry_point_};
    
    // Search from top layer to layer 0
    for (int lc = max_level_; lc > 0; --lc) {
        auto nearest = searchLayer(query, enter_points, 1, lc);
        if (!nearest.empty()) {
            enter_points = nearest;
        }
    }
    
    // Search at layer 0
    auto candidates = searchLayer(query, enter_points, std::max(ef, k), 0);
    
    std::vector<std::pair<int, float>> results;
    for (size_t i = 0; i < candidates.size() && i < static_cast<size_t>(k); ++i) {
        int node_id = candidates[i];
        float dist = distance(query, nodes_[node_id].data);
        results.push_back({nodes_[node_id].id, dist});
    }
    
    return results;
}
