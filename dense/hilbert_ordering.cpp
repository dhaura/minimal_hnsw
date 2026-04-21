#include "hilbert_ordering.h"

#include <limits>
#include <stdexcept>
#include <algorithm>

#include <faiss/VectorTransform.h>

namespace hnsw {
namespace {

uint64_t xy2d(uint32_t n, uint32_t x, uint32_t y) {
    uint64_t rx, ry, s, d = 0;
    for (s = n / 2; s > 0; s /= 2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
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

uint64_t computeHilbertIndex(float x,
                             float y,
                             float min_x, float max_x,
                             float min_y, float max_y) {
    float range_x = max_x - min_x;
    float range_y = max_y - min_y;
    if (range_x < 1e-6f) range_x = 1.0f;
    if (range_y < 1e-6f) range_y = 1.0f;

    float norm_x = (x - min_x) / range_x;
    float norm_y = (y - min_y) / range_y;

    norm_x = std::max(0.0f, std::min(1.0f, norm_x));
    norm_y = std::max(0.0f, std::min(1.0f, norm_y));

    const uint32_t grid_size = 65536;
    uint32_t grid_x = static_cast<uint32_t>(norm_x * (grid_size - 1));
    uint32_t grid_y = static_cast<uint32_t>(norm_y * (grid_size - 1));

    return xy2d(grid_size, grid_x, grid_y);
}

} // namespace

void HilbertOrdering::reorderDataset(std::vector<std::vector<float>>& points,
                                     std::vector<uint32_t>& old_to_new,
                                     std::vector<uint32_t>& new_to_old) {
    const size_t num_nodes = points.size();
    old_to_new.clear();
    new_to_old.clear();
    if (num_nodes <= 1) {
        old_to_new.resize(num_nodes);
        new_to_old.resize(num_nodes);
        for (size_t i = 0; i < num_nodes; ++i) {
            old_to_new[i] = static_cast<uint32_t>(i);
            new_to_old[i] = static_cast<uint32_t>(i);
        }
        return;
    }

    const size_t dim = points[0].size();
    if (dim == 0) {
        throw std::runtime_error("Cannot apply Hilbert ordering to zero-dimensional points");
    }
    for (size_t i = 1; i < num_nodes; ++i) {
        if (points[i].size() != dim) {
            throw std::runtime_error("Input points have inconsistent dimensions");
        }
    }

    std::vector<float> flat_data(num_nodes * dim, 0.0f);
    for (size_t i = 0; i < num_nodes; ++i) {
        std::copy(points[i].begin(), points[i].end(), flat_data.begin() + i * dim);
    }

    const int d_in = static_cast<int>(dim);
    const int d_out = (d_in > 1) ? 2 : 1;
    faiss::PCAMatrix pca(d_in, d_out);
    pca.train(num_nodes, flat_data.data());
    if (!pca.is_trained) {
        throw std::runtime_error("Faiss PCA training failed for Hilbert projection");
    }

    std::vector<float> pca_output(num_nodes * static_cast<size_t>(d_out), 0.0f);
    pca.apply_noalloc(num_nodes, flat_data.data(), pca_output.data());

    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();

    for (size_t i = 0; i < num_nodes; ++i) {
        const float x = pca_output[i * static_cast<size_t>(d_out)];
        const float y = (d_out > 1) ? pca_output[i * static_cast<size_t>(d_out) + 1] : 0.0f;
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
    }

    std::vector<std::pair<uint64_t, uint32_t>> hilbert_indices;
    hilbert_indices.reserve(num_nodes);

    for (size_t i = 0; i < num_nodes; ++i) {
        const float x = pca_output[i * static_cast<size_t>(d_out)];
        const float y = (d_out > 1) ? pca_output[i * static_cast<size_t>(d_out) + 1] : 0.0f;
        const uint64_t h_idx = computeHilbertIndex(x, y, min_x, max_x, min_y, max_y);
        hilbert_indices.push_back(std::make_pair(h_idx, static_cast<uint32_t>(i)));
    }

    std::sort(hilbert_indices.begin(), hilbert_indices.end());

    old_to_new.resize(num_nodes);
    new_to_old.resize(num_nodes);
    for (uint32_t new_id = 0; new_id < num_nodes; ++new_id) {
        const uint32_t old_id = hilbert_indices[new_id].second;
        old_to_new[old_id] = new_id;
        new_to_old[new_id] = old_id;
    }

    std::vector<std::vector<float>> reordered_points(num_nodes);
    for (uint32_t new_id = 0; new_id < num_nodes; ++new_id) {
        reordered_points[new_id] = std::move(points[new_to_old[new_id]]);
    }
    points = std::move(reordered_points);
}

} // namespace hnsw
