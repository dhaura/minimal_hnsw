#ifndef SPARSE_HNSW_DIST_LOG_H
#define SPARSE_HNSW_DIST_LOG_H

// Per-distance-call byte-efficiency log, written as CSV. Gated behind
// SPARSE_HNSW_DIST_LOG so the production kernel is bit-identical when off.
// Records, for every document scored during traversal, how much of the row was
// streamed for nothing: entries whose dimension is absent from the query
// contribute zero to the dot product but cost the same bytes off DRAM.
#ifdef SPARSE_HNSW_DIST_LOG

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace sparse_hnsw {
namespace dist_log {

    // A row is ~40 bytes, so a thread hands over ~26k rows per fwrite. fwrite
    // holds the stdio lock for the whole call, so rows never interleave.
    constexpr size_t kFlushAt = 1u << 20;

    struct Buffer;

    // One FILE* shared by every thread; each thread batches into its own
    // Buffer. open()/close() bracket the search, so the hot path reads `fp`
    // without a lock -- `mu` guards only the registry.
    struct Sink {
        std::FILE* fp = nullptr;          // per-row CSV; may be null
        uint32_t stride = 1;              // per-row sampling only
        std::string hist_path;            // histogram CSV; empty = none
        uint32_t nbins = 100;
        bool active = false;              // either sink configured
        std::mutex mu;
        std::vector<Buffer*> buffers;
        // Merged from threads that died before close().
        std::vector<uint64_t> merged;
        uint64_t calls = 0, entries = 0, unused = 0, zero_overlap = 0;
    };

    // Leaked on purpose: thread_local Buffers may outlive static destruction.
    inline Sink& sink() { static Sink* s = new Sink(); return *s; }

    struct Buffer {
        std::vector<char> buf;
        uint32_t query_id = 0;
        bool row_on = false;   // this query is in the per-row sample
        bool hist_on = false;  // logging configured at all
        // Bins are per-thread so the hot path never touches shared state;
        // they are summed into the Sink exactly once, at teardown.
        std::vector<uint64_t> hist;
        uint64_t calls = 0, entries = 0, unused = 0, zero_overlap = 0;

        Buffer() {
            buf.reserve(kFlushAt + 64);
            std::lock_guard<std::mutex> g(sink().mu);
            hist.assign(sink().nbins, 0);
            sink().buffers.push_back(this);
        }

        // Caller holds sink().mu.
        void mergeInto(Sink& s) {
            if (s.merged.size() < hist.size()) { s.merged.resize(hist.size(), 0); }
            for (size_t i = 0; i < hist.size(); ++i) { s.merged[i] += hist[i]; }
            hist.assign(hist.size(), 0);
            s.calls += calls; s.entries += entries;
            s.unused += unused; s.zero_overlap += zero_overlap;
            calls = entries = unused = zero_overlap = 0;
        }

        ~Buffer() {
            std::lock_guard<std::mutex> g(sink().mu);
            writeOut();
            mergeInto(sink());
            auto& v = sink().buffers;
            for (size_t i = 0; i < v.size(); ++i) {
                if (v[i] == this) { v[i] = v.back(); v.pop_back(); break; }
            }
        }

        // No registry lock: only fwrite's own lock, taken once per ~26k rows.
        void writeOut() {
            if (!buf.empty() && sink().fp) {
                std::fwrite(buf.data(), 1, buf.size(), sink().fp);
            }
            buf.clear();
        }
    };

    inline Buffer& tls() { static thread_local Buffer b; return b; }

    inline void appendU32(std::vector<char>& b, uint32_t v) {
        char tmp[10];
        int n = 0;
        do { tmp[n++] = static_cast<char>('0' + v % 10); v /= 10; } while (v);
        while (n > 0) b.push_back(tmp[--n]);
    }

    // Log only every Nth query. The full log is ~ndist rows per query; a stride
    // keeps the CSV sane while leaving the per-row distribution unbiased.
    // row_path may be null: histogram-only, which needs no sampling because a
    // bin increment costs nothing. hist_path may be null: per-row only.
    inline bool open(const char* row_path, uint32_t stride,
                     const char* hist_path, uint32_t nbins) {
        Sink& s = sink();
        s.nbins = nbins ? nbins : 100;
        s.stride = stride ? stride : 1;
        s.hist_path = hist_path ? hist_path : "";
        s.merged.assign(s.nbins, 0);
        if (row_path) {
            s.fp = std::fopen(row_path, "w");
            if (!s.fp) { return false; }
            std::fputs("query_id,doc_id,total_nnz,unused_nnz,unused_pct\n", s.fp);
        }
        s.active = (s.fp != nullptr) || !s.hist_path.empty();
        return true;
    }

    // Ascending by bin_lo, every bin emitted including empty ones, so the
    // x-axis is complete and bars never shift. Counts ALL distance calls --
    // `stride` thins the per-row file only, never these bins.
    inline bool writeHistogram() {
        Sink& s = sink();
        if (s.hist_path.empty()) { return true; }
        std::FILE* f = std::fopen(s.hist_path.c_str(), "w");
        if (!f) { return false; }
        std::fputs("bin_lo,bin_hi,count,fraction\n", f);
        const double w = 100.0 / static_cast<double>(s.nbins);
        for (uint32_t i = 0; i < s.nbins; ++i) {
            const uint64_t c = i < s.merged.size() ? s.merged[i] : 0;
            std::fprintf(f, "%.4f,%.4f,%llu,%.10f\n", i * w, (i + 1) * w,
                         static_cast<unsigned long long>(c),
                         s.calls ? static_cast<double>(c) / static_cast<double>(s.calls) : 0.0);
        }
        std::fclose(f);
        return true;
    }

    // Exact over every call, so the headline number needs no re-reading of the
    // per-row CSV. zero_overlap is the spike that lands in the last bin.
    inline uint64_t totalCalls()   { return sink().calls; }
    inline uint64_t totalEntries() { return sink().entries; }
    inline uint64_t totalUnused()  { return sink().unused; }
    inline uint64_t zeroOverlap()  { return sink().zero_overlap; }

    // Must run after the search threads are done and before exit.
    inline void close() {
        Sink& s = sink();
        if (!s.active) { return; }
        {
            std::lock_guard<std::mutex> g(s.mu);
            for (Buffer* b : s.buffers) { b->writeOut(); b->mergeInto(s); }
            if (s.fp) { std::fclose(s.fp); s.fp = nullptr; }
        }
        writeHistogram();
        s.active = false;
    }

    inline void setQuery(uint32_t q) {
        Buffer& t = tls();
        t.query_id = q;
        t.hist_on = sink().active;
        t.row_on = sink().fp && (q % sink().stride) == 0;
    }

    inline void flushThread() { tls().writeOut(); }

    // query_id,doc_id,total_nnz,unused_nnz,unused_pct
    inline void emit(uint32_t doc_id, uint32_t total_nnz, uint32_t unused_nnz) {
        Buffer& t = tls();
        if (!t.hist_on) { return; }

        // Binning runs on EVERY call: integer-exact, no float, one increment
        // into thread-private memory. This is what the histogram is built from.
        if (total_nnz) {
            uint64_t bin = (static_cast<uint64_t>(unused_nnz) * t.hist.size())
                           / total_nnz;
            if (bin >= t.hist.size()) { bin = t.hist.size() - 1; }  // 100% -> last
            ++t.hist[bin];
            t.calls++;
            t.entries += total_nnz;
            t.unused += unused_nnz;
            if (unused_nnz == total_nnz) { t.zero_overlap++; }
        }

        if (!t.row_on) { return; }
        std::vector<char>& b = t.buf;
        appendU32(b, t.query_id);   b.push_back(',');
        appendU32(b, doc_id);       b.push_back(',');
        appendU32(b, total_nnz);    b.push_back(',');
        appendU32(b, unused_nnz);   b.push_back(',');
        // Two decimals via integer math; no float formatting on the hot path.
        const uint32_t bp = total_nnz
            ? static_cast<uint32_t>((static_cast<uint64_t>(unused_nnz) * 10000u
                                     + total_nnz / 2) / total_nnz)
            : 0u;
        appendU32(b, bp / 100);
        b.push_back('.');
        b.push_back(static_cast<char>('0' + (bp / 10) % 10));
        b.push_back(static_cast<char>('0' + bp % 10));
        b.push_back('\n');
        if (b.size() >= kFlushAt) { t.writeOut(); }
    }

}  // namespace dist_log
}  // namespace sparse_hnsw

#endif  // SPARSE_HNSW_DIST_LOG
#endif  // SPARSE_HNSW_DIST_LOG_H
