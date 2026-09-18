// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
//
// Open-addressing hash map with linear probing for uint64_t keys. Intended for
// fingerprint-keyed lookups in the causal dynamical triangulation (CDT) hot path,
// where std::unordered_map's chained hashing costs cache misses.
//
// Keys are already well mixed (by Fingerprint::mix64), so the hash is the identity.

#ifndef TESSERA_FLAT_HASH_MAP_H
#define TESSERA_FLAT_HASH_MAP_H

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {}
namespace tessera::mesh {
using namespace ::tessera::graph;
using namespace ::tessera::spacetime;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

/// Identity hash for uint64_t keys that are already well-distributed
/// (e.g. fingerprints produced by mix64).
struct IdentityHash {
    std::size_t operator()(std::uint64_t key) const noexcept { return key; }
};

/// Open-addressing hash map: uint64_t → V with linear probing.
///
/// Roughly 3x faster than std::unordered_map on lookup-heavy workloads: keys and
/// values sit in contiguous arrays.
///
/// Key 0 is reserved as the empty sentinel; a caller for whom 0 is a valid key has to
/// offset keys by 1. Fingerprints formed by XOR of mix64 values are never 0 for a
/// non-empty simplex.
template<typename V>
class FlatHashMap {
  public:
    static constexpr std::uint64_t EMPTY = 0;

    FlatHashMap() { resize(64); }

    void reserve(std::size_t n) {
        std::size_t need = (n * 10) / 7 + 1; // ~70% load factor
        if (need > cap_) resize(nextPow2(need));
    }

    V* find(std::uint64_t key) noexcept {
        std::size_t idx = key & mask_;
        while (true) {
            if (keys_[idx] == key) return &vals_[idx];
            if (keys_[idx] == EMPTY) return nullptr;
            idx = (idx + 1) & mask_;
        }
    }

    const V* find(std::uint64_t key) const noexcept {
        std::size_t idx = key & mask_;
        while (true) {
            if (keys_[idx] == key) return &vals_[idx];
            if (keys_[idx] == EMPTY) return nullptr;
            idx = (idx + 1) & mask_;
        }
    }

    /// Insert or update. Returns a reference to the value.
    V& operator[](std::uint64_t key) {
        if (size_ * 10 >= cap_ * 7) grow();
        std::size_t idx = key & mask_;
        while (true) {
            if (keys_[idx] == key) return vals_[idx];
            if (keys_[idx] == EMPTY) {
                keys_[idx] = key;
                vals_[idx] = V{};
                ++size_;
                return vals_[idx];
            }
            idx = (idx + 1) & mask_;
        }
    }

    /// Insert a key-value pair; no-op if the key already exists.
    /// Returns a reference to the existing or new value.
    V& insert(std::uint64_t key, const V &val) {
        if (size_ * 10 >= cap_ * 7) grow();
        std::size_t idx = key & mask_;
        while (true) {
            if (keys_[idx] == key) return vals_[idx];
            if (keys_[idx] == EMPTY) {
                keys_[idx] = key;
                vals_[idx] = val;
                ++size_;
                return vals_[idx];
            }
            idx = (idx + 1) & mask_;
        }
    }

    bool erase(std::uint64_t key) noexcept {
        std::size_t idx = key & mask_;
        while (true) {
            if (keys_[idx] == EMPTY) return false;
            if (keys_[idx] == key) {
                // Backward-shift deletion to maintain probe chains
                keys_[idx] = EMPTY;
                --size_;
                std::size_t next = (idx + 1) & mask_;
                while (keys_[next] != EMPTY) {
                    std::size_t ideal = keys_[next] & mask_;
                    // Check if 'next' is displaced past 'idx' (wrapping)
                    bool displaced = (idx <= next)
                        ? (ideal <= idx || ideal > next)
                        : (ideal <= idx && ideal > next);
                    if (displaced) {
                        keys_[idx] = keys_[next];
                        vals_[idx] = std::move(vals_[next]);
                        keys_[next] = EMPTY;
                        idx = next;
                    }
                    next = (next + 1) & mask_;
                }
                return true;
            }
            idx = (idx + 1) & mask_;
        }
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

  private:
    std::vector<std::uint64_t> keys_;
    std::vector<V> vals_;
    std::size_t cap_ = 0;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;

    static std::size_t nextPow2(std::size_t n) {
        std::size_t p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    void resize(std::size_t newCap) {
        auto oldKeys = std::move(keys_);
        auto oldVals = std::move(vals_);
        auto oldCap = cap_;
        cap_ = newCap;
        mask_ = cap_ - 1;
        keys_.assign(cap_, EMPTY);
        vals_.resize(cap_);
        size_ = 0;
        for (std::size_t i = 0; i < oldCap; ++i) {
            if (oldKeys[i] != EMPTY) {
                insert(oldKeys[i], oldVals[i]);
            }
        }
    }

    void grow() { resize(cap_ * 2); }
};

} // namespace tessera::mesh

#endif // TESSERA_FLAT_HASH_MAP_H
