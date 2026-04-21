#ifndef HILBERT_ORDERING_H
#define HILBERT_ORDERING_H

#include <cstdint>
#include <vector>

namespace hnsw {

class HilbertOrdering {
public:
    static void reorderDataset(std::vector<std::vector<float>>& points,
                               std::vector<uint32_t>& old_to_new,
                               std::vector<uint32_t>& new_to_old);
};

} // namespace hnsw

#endif // HILBERT_ORDERING_H
