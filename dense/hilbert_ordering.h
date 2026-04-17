#ifndef HILBERT_ORDERING_H
#define HILBERT_ORDERING_H

#include "hnsw.h"

namespace hnsw {

class HilbertOrdering {
public:
    static void reorder(HNSW& index);
};

} // namespace hnsw

#endif // HILBERT_ORDERING_H
