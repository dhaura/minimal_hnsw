#pragma once

#include "csr_matrix.h"

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sparse_hnsw {

// Exact sparse-query lookup accelerated by a blocked Bloom filter.
template <std::size_t Capacity = 128, std::size_t BloomBits = 2048>
class alignas(64) QueryBloomMap {
    static_assert(Capacity >= 16, "Capacity must be at least one AVX2 block");
    static_assert(BloomBits >= 64 && (BloomBits & (BloomBits - 1)) == 0,
                  "BloomBits must be a power of two");
    static_assert((BloomBits % 64) == 0,
                  "BloomBits must be a multiple of 64");

    static constexpr std::size_t kBloomWords = BloomBits / 64;
    static_assert((kBloomWords & (kBloomWords - 1)) == 0,
                  "Bloom word count must be a power of two");

public:
    QueryBloomMap() noexcept : size_(0) {
        bloom_.fill(0);
    }

    void clear() noexcept {
        bloom_.fill(0);
        size_ = 0;
    }

    // Duplicate keys preserve dense SPA's last-write-wins behavior. Returns
    // false only when the number of unique keys exceeds Capacity.
    bool build(const IndiceDataPair* query, uint32_t nnz) noexcept {
        clear();

        for (uint32_t i = 0; i < nnz; ++i) {
            const uint16_t key = query[i].indice;
            const float value = static_cast<float>(query[i].data);

            bool duplicate = false;
            for (uint16_t j = 0; j < size_; ++j) {
                if (keys_[j] == key) {
                    values_[j] = value;
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                continue;
            }

            if (size_ == Capacity) {
                clear();
                return false;
            }

            keys_[size_] = key;
            values_[size_] = value;
            addToBloom(key);
            ++size_;
        }
        return true;
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    inline float lookup(uint16_t key) const noexcept {
        std::size_t word;
        const uint64_t mask = bloomMask(key, word);
#if defined(__GNUC__) || defined(__clang__)
        if (__builtin_expect((bloom_[word] & mask) != mask, 1)) {
#else
        if ((bloom_[word] & mask) != mask) {
#endif
            return 0.0f;
        }

        std::size_t i = 0;
#if defined(__AVX2__)
        const __m256i needle = _mm256_set1_epi16(static_cast<int16_t>(key));
        for (; i + 16 <= size_; i += 16) {
            const __m256i haystack = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(keys_.data() + i));
            const __m256i equal = _mm256_cmpeq_epi16(haystack, needle);
            const unsigned matches =
                static_cast<unsigned>(_mm256_movemask_epi8(equal));
            if (matches != 0) {
                return values_[i + (static_cast<unsigned>(__builtin_ctz(matches)) >> 1)];
            }
        }
#endif
        for (; i < size_; ++i) {
            if (keys_[i] == key) {
                return values_[i];
            }
        }
        return 0.0f;
    }

    uint16_t size() const noexcept {
        return size_;
    }

    static constexpr std::size_t lookupBytes() noexcept {
        return sizeof(std::array<uint64_t, kBloomWords>) +
               sizeof(std::array<uint16_t, Capacity>) +
               sizeof(std::array<float, Capacity>);
    }

private:
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    static inline uint64_t bloomMask(uint16_t key,
                                     std::size_t& word) noexcept {
        uint32_t h1 = static_cast<uint32_t>(key) * 0x9E3779B1u;
        h1 ^= h1 >> 16;
        uint32_t h2 = static_cast<uint32_t>(key) * 0x85EBCA77u + 0xC2B2AE3Du;
        h2 ^= h2 >> 15;

        word = static_cast<std::size_t>(h1) & (kBloomWords - 1);
        const uint32_t bit1 = (h1 >> 5) & 63u;
        uint32_t bit2 = h2 & 63u;
        bit2 = (bit2 + static_cast<uint32_t>(bit2 == bit1)) & 63u;
        return (uint64_t{1} << bit1) | (uint64_t{1} << bit2);
    }

    void addToBloom(uint16_t key) noexcept {
        std::size_t word;
        const uint64_t mask = bloomMask(key, word);
        bloom_[word] |= mask;
    }

    alignas(64) std::array<uint64_t, kBloomWords> bloom_;
    alignas(64) std::array<uint16_t, Capacity> keys_;
    alignas(64) std::array<float, Capacity> values_;
    uint16_t size_;
};

using QueryBloomFilterMap = QueryBloomMap<128, 2048>;

} // namespace sparse_hnsw
