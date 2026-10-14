#pragma once

#include "csr_matrix.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace sparse_hnsw {

template <std::size_t Capacity = 512>
class alignas(64) QueryFlatMap {
    static_assert(Capacity >= 2, "Capacity must be at least two");
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");

public:
    static constexpr std::size_t kMask = Capacity - 1;

    QueryFlatMap() noexcept {
        clear();
    }

    void clear() noexcept {
        // Values do not need clearing: they are read only after a key match.
        std::fill(keys_.begin(), keys_.end(), emptyKey());
    }

    // Returns false if the recommended <= 0.5 load factor would be exceeded
    // or the reserved empty key occurs. Duplicate keys are last-write-wins.
    bool build(const IndiceDataPair* query, uint32_t nnz) noexcept {
        clear();
        if (nnz > Capacity / 2) {
            return false;
        }

        for (uint32_t i = 0; i < nnz; ++i) {
            const uint16_t key = query[i].indice;
            if (key == emptyKey()) {
                return false;
            }
            if (!insertOrAssign(key, static_cast<float>(query[i].data))) {
                return false;
            }
        }
        return true;
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    inline float lookup(uint16_t key) const noexcept {
        std::size_t slot = hash(key);
        for (std::size_t probes = 0; probes < Capacity; ++probes) {
            const uint16_t observed = keys_[slot];
            if (observed == emptyKey()) {
                return 0.0f;
            }
            if (observed == key) {
                return values_[slot];
            }
            slot = (slot + 1) & kMask;
        }
        return 0.0f;
    }

private:
    static uint16_t emptyKey() noexcept {
        return std::numeric_limits<uint16_t>::max();
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    static inline std::size_t hash(uint16_t key) noexcept {
        uint32_t x = static_cast<uint32_t>(key) * 0x9E3779B1u;
        x ^= x >> 16;
        return static_cast<std::size_t>(x) & kMask;
    }

    bool insertOrAssign(uint16_t key, float value) noexcept {
        std::size_t slot = hash(key);
        for (std::size_t probes = 0; probes < Capacity; ++probes) {
            const uint16_t observed = keys_[slot];
            if (observed == emptyKey()) {
                keys_[slot] = key;
                values_[slot] = value;
                return true;
            }
            if (observed == key) {
                values_[slot] = value;
                return true;
            }
            slot = (slot + 1) & kMask;
        }
        return false;
    }

    alignas(64) std::array<uint16_t, Capacity> keys_;
    alignas(64) std::array<float, Capacity> values_;
};

using QueryHashMap = QueryFlatMap<512>;

} // namespace sparse_hnsw
