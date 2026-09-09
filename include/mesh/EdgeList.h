// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.

#ifndef TESSERA_EDGELIST_H
#define TESSERA_EDGELIST_H

#include <cstdint>
#include <deque>
#include <vector>
#include <unordered_map>

#include "mesh/Edge.h"
#include "Logger.h"

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

/// Flat-pool edge container.
///
/// Edges live in a `std::deque` (stable element addresses) with a free-list
/// for slot reuse.  `fpToSlot_` maps edge fingerprint → pool slot for O(1)
/// deduplication.
class EdgeList {
  public:
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] const Edges &toVector() const noexcept;

    EdgePtr add(const VertexPtr &source, const VertexPtr &target);
    EdgePtr add(const VertexPtr &source, const VertexPtr &target, std::complex<double> length) noexcept;
    /// Insert if absent, otherwise return the existing edge.
    /// Returns {ptr, true} on fresh insert, {ptr, false} on dedupe-hit.
    /// Used by transactional Pachner moves to record which edges they
    /// freshly created (so rollback knows which to remove).
    std::pair<EdgePtr, bool> tryAdd(const VertexPtr &source, const VertexPtr &target,
                                    std::complex<double> length);
    EdgePtr get(const std::uint64_t &fingerprint);
    void remove(const EdgePtr &edge) noexcept;

    /// The lookup key for an edge, derived from the ids its endpoints hold now.
    ///
    /// This is the key ``add``/``tryAdd`` insert under, so it is the only value
    /// that is guaranteed to find an edge again. An ``Edge``'s own
    /// ``fingerprint`` member is maintained incrementally and can fall out of
    /// step with its endpoints; keying off it silently misses.
    [[nodiscard]] static std::uint64_t keyOf(const Edge &edge) noexcept;

    /// Re-key an edge's fingerprint in the lookup map without moving the object.
    void rekeyEdge(std::uint64_t oldFp, std::uint64_t newFp);

    /// Detach an edge from the lookup (but keep it in the pool).
    /// Returns the pool slot so the caller can update the fingerprint and call
    /// reattachEdge().  Returns UINT32_MAX if not found.
    std::uint32_t detachEdge(std::uint64_t fp) {
      auto it = fpToSlot_.find(fp);
      if (it == fpToSlot_.end()) return UINT32_MAX;
      auto slot = it->second;
      fpToSlot_.erase(it);
      return slot;
    }

    /// Re-attach a previously detached edge under its current key.
    ///
    /// Returns false when the key is already taken by a different slot, which
    /// means two live edges now claim the same vertex pair. The caller has to
    /// decide what that means; silently dropping one leaves it in the live
    /// vector with no way to find it again, and every later lookup for that
    /// pair then creates a duplicate.
    bool reattachEdge(std::uint32_t slot) {
      const auto fp = keyOf(pool_[slot]);
      const auto [it, inserted] = fpToSlot_.emplace(fp, slot);
      return inserted || it->second == slot;
    }

    void reserve(std::size_t nSimplices);

  private:
    std::deque<Edge> pool_;
    std::vector<std::uint32_t> freeSlots_;
    std::unordered_map<std::uint64_t, std::uint32_t> fpToSlot_;
    std::vector<EdgePtr> liveVec_;

    EdgePtr getOrInsert(const VertexPtr &source, const VertexPtr &target, std::complex<double> length);
    std::uint32_t allocSlot(const VertexPtr &source, const VertexPtr &target, std::complex<double> length);
};
} // namespace tessera::mesh

#endif //TESSERA_EDGELIST_H
