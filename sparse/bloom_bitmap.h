#ifndef SPARSE_BLOOM_BITMAP_H
#define SPARSE_BLOOM_BITMAP_H

#include "csr_matrix.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sparse_hnsw {

class BloomBitmap {
public:
    struct alignas(64) Row {
        std::array<uint64_t, 8> words{};
    };
    static_assert(sizeof(Row) == 64, "Bloom rows must occupy one cache line");

    struct Token {
        std::array<uint16_t, 5> positions{};
    };

    struct Query {
        std::vector<Token> terms;
        uint32_t nnz = 0;
        // Smallest match count that passes: min m with 100*m > cutoff (set by setCutoff).
        uint32_t need = 0;
        // For h == 1 only: level[j] has bit p set iff more than j query terms hash to p,
        // so sum_j popcount(row & level[j]) counts terms exactly as the term loop does.
        std::vector<std::array<uint64_t, 8>> level;

        void build(const IndiceDataPair* entries, uint32_t count, uint8_t hashes) {
            BloomBitmap::validateHashes(hashes);
            terms.clear();
            terms.reserve(count);
            level.clear();
            for (uint32_t i = 0; i < count; ++i) {
                if (entries[i].data == 0) { continue; }
                Token token;
                for (uint8_t h = 0; h < hashes; ++h) {
                    token.positions[h] = BloomBitmap::position(entries[i].indice, h);
                }
                terms.push_back(token);
                if (hashes == 1) {
                    const uint16_t bit = token.positions[0];
                    const uint64_t mask = uint64_t{1} << (bit & 63);
                    size_t j = 0;
                    while (j < level.size() && (level[j][bit >> 6] & mask)) { ++j; }
                    if (j == level.size()) { level.push_back({}); }
                    level[j][bit >> 6] |= mask;
                }
            }
            nnz = static_cast<uint32_t>(terms.size());
        }

        // cutoff = threshold_pct * nnz; same comparison the per-term loop used.
        void setCutoff(double cutoff) {
            need = 0;
            while (100.0 * need <= cutoff) { ++need; }
        }
    };

    static void validateHashes(uint8_t hashes) {
        if (hashes < 1 || hashes > 5) {
            throw std::invalid_argument("Bloom hash count must be in [1, 5]");
        }
    }

    void build(const CSRMatrix* source, uint8_t hashes) {
        validateHashes(hashes);
        if (!source || source->nrow < 0 || source->ncol <= 0 || source->ncol > 65536 ||
            static_cast<uint64_t>(source->nrow) > std::numeric_limits<size_t>::max() / sizeof(Row)) {
            throw std::invalid_argument("invalid Bloom source dimensions");
        }
        hashes_ = hashes;
        rows_.clear();
        rows_.resize(static_cast<size_t>(source->nrow));
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < source->nrow; ++i) {
            Row& target = rows_[static_cast<size_t>(i)];
            for (int64_t p = source->indptr[i]; p < source->indptr[i + 1]; ++p) {
                const auto& entry = source->indices_data[p];
                if (entry.data == 0) { continue; }
                for (uint8_t h = 0; h < hashes; ++h) {
                    const uint16_t bit = position(entry.indice, h);
                    target.words[bit >> 6] |= uint64_t{1} << (bit & 63);
                }
            }
        }
    }

    void clear() { rows_.clear(); hashes_ = 0; }
    size_t bytes() const { return rows_.size() * sizeof(Row); }
    const Row* row(size_t document) const { return &rows_[document]; }
    uint8_t hashes() const { return hashes_; }

    // Pass iff at least query.need query terms appear present. Call setCutoff first.
    bool passes(size_t document, const Query& query) const {
        const Row& bits = rows_[document];
        if (hashes_ == 1) {
            uint32_t matches = 0;
            for (const auto& mask : query.level) {
                for (int w = 0; w < 8; ++w) {
                    matches += static_cast<uint32_t>(__builtin_popcountll(bits.words[w] & mask[w]));
                }
            }
            return matches >= query.need;
        }
        uint32_t matches = 0;
        for (const Token& term : query.terms) {
            bool present = true;
            for (uint8_t h = 0; h < hashes_; ++h) {
                const uint16_t bit = term.positions[h];
                present &= (bits.words[bit >> 6] & (uint64_t{1} << (bit & 63))) != 0;
            }
            matches += present;
            if (matches >= query.need) { return true; }
        }
        return false;
    }

private:
    static uint16_t position(uint32_t dimension, uint8_t hash_number) {
        static constexpr uint64_t salts[5] = {
            0x9e3779b97f4a7c15ULL, 0xbf58476d1ce4e5b9ULL,
            0x94d049bb133111ebULL, 0xd6e8feb86659fd93ULL,
            0xa0761d6478bd642fULL
        };
        uint64_t x = static_cast<uint64_t>(dimension) + salts[hash_number];
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return static_cast<uint16_t>((x ^ (x >> 31)) & 511u);
    }

    std::vector<Row> rows_;
    uint8_t hashes_ = 0;
};

} // namespace sparse_hnsw
#endif
