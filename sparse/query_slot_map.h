#pragma once

#include "csr_matrix.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sparse_hnsw {

// Exact direct-address query map for sparse queries with at most 255 unique
// dimensions. Each vocabulary dimension stores an 8-bit index into values_.
// Slot 255 is the absent-key sentinel and permanently contains 0.0f.
//
// For MS MARCO (30,109 dimensions), the lookup working set is:
//   30,109 bytes dimension_to_slot_ + 1,024 bytes values_ = 31,133 bytes.
// No hash, modulo, probe loop, or data-dependent branch is needed by lookup().
class alignas(64) QuerySlotMap {
public:
    static constexpr uint16_t kMaxEntries = 255;
    static constexpr uint8_t kEmptySlot = 255;

    QuerySlotMap() noexcept : active_count_(0) {
        values_.fill(0.0f);
    }

    // Allocate and initialize the direct-address table only when a scratchpad
    // first sees a dimension (normally once per OpenMP parallel region).
    bool build(std::size_t dimension, const IndiceDataPair* query,
               uint32_t nnz) {
        clear();

        if (dimension_to_slot_.size() != dimension) {
            dimension_to_slot_.assign(
                dimension, static_cast<uint8_t>(kEmptySlot));
        }

        for (uint32_t i = 0; i < nnz; ++i) {
            const uint16_t dimension_id = query[i].indice;
            if (dimension_id >= dimension_to_slot_.size()) {
                clear();
                return false;
            }

            uint8_t& slot = dimension_to_slot_[dimension_id];
            if (slot == kEmptySlot) {
                if (active_count_ == kMaxEntries) {
                    clear();
                    return false;
                }
                slot = static_cast<uint8_t>(active_count_);
                active_dimensions_[active_count_] = dimension_id;
                ++active_count_;
            }

            // Duplicate dimensions retain the dense SPA's last-write-wins
            // behavior without consuming another value slot.
            values_[slot] = static_cast<float>(query[i].data);
        }

        return true;
    }

    // Clear only dimensions touched by the current query: O(query NNZ), not
    // O(vocabulary size). This also makes build() safe if a caller forgot to
    // explicitly clear the preceding query.
    void clear() noexcept {
        for (uint16_t i = 0; i < active_count_; ++i) {
            dimension_to_slot_[active_dimensions_[i]] = kEmptySlot;
        }
        active_count_ = 0;
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    inline float lookup(uint16_t dimension_id) const noexcept {
        return values_[dimension_to_slot_[dimension_id]];
    }

    std::size_t lookupBytes() const noexcept {
        return dimension_to_slot_.size() + sizeof(values_);
    }

    uint16_t size() const noexcept {
        return active_count_;
    }

private:
    std::vector<uint8_t> dimension_to_slot_;
    alignas(64) std::array<float, 256> values_;
    std::array<uint16_t, kMaxEntries> active_dimensions_;
    uint16_t active_count_;
};

} // namespace sparse_hnsw
