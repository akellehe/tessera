// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

//
// Created by andrew on 10/23/25.
//

#ifndef TESSERA_VERTEXLIST_H
#define TESSERA_VERTEXLIST_H

#include <cstdint>
#include <deque>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include "mesh/Vertex.h"

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

/// Flat-pool vertex container.
///
/// Objects live in a ``std::deque<std::unique_ptr<Vertex>>``, so element addresses
/// are stable across resizes and the pool can hold polymorphic subclasses (e.g.
/// ``quantum::QuantumVertex``) without slicing. A parallel ``liveVec_`` gives O(1)
/// random access without copying, and ``idToIndex_`` maps vertex id to pool slot.
///
/// Slots are never recycled. ``remove`` drops the vertex from the live vector and the
/// id index but leaves the ``unique_ptr<Vertex>`` at its pool slot, mirroring
/// ``Spacetime::simplexStorage_``. That removes the use-after-free hazard of slot
/// reuse: a raw ``Vertex*`` cached elsewhere stays dereferenceable for the life of the
/// VertexList. The cost is memory growth of one Vertex shell per ever-allocated
/// vertex; for QuantumVertex it also retains the Eigen state until the VertexList is
/// destroyed.
class VertexList {
  public:
    Vertex* operator[](const std::uint64_t vertexId) {
      auto it = idToIndex_.find(vertexId);
      if (it == idToIndex_.end()) return nullptr;
      return pool_[it->second].get();
    }

    Vertex* get(std::uint64_t id) {
      auto it = idToIndex_.find(id);
      if (it == idToIndex_.end()) return nullptr;
      return pool_[it->second].get();
    }

    bool contains(const std::uint64_t id) const noexcept {
      return idToIndex_.contains(id);
    }

    Vertex* add(const std::uint64_t id, const std::vector<double> &coords) noexcept {
      return addAs<Vertex>(id, id, coords);
    }

    Vertex* add(const std::uint64_t id) noexcept {
      return add(id, std::vector<double>{});
    }

    /// Construct a vertex (or any subclass ``T``) directly into the pool, taking
    /// ownership through ``std::unique_ptr<Vertex>``.
    ///
    /// ``id`` is the lookup key stored in ``idToIndex_``. It is normally also the
    /// first constructor argument, so callers pass it both as the key and inside
    /// ``args...``.
    ///
    /// Returns a non-owning ``T*`` whose lifetime is tied to the VertexList. For a new
    /// id the result is never null; for an existing id the existing slot's pointer is
    /// returned, and the downcast then fails if the recorded type differs.
    template <typename T, typename... Args>
    T* addAs(std::uint64_t id, Args&&... args) noexcept {
      static_assert(std::is_base_of_v<Vertex, T>,
                    "VertexList::addAs<T> requires T : public Vertex");
      auto found = idToIndex_.find(id);
      if (found != idToIndex_.end()) {
        CLOG(INFO_LEVEL, "Vertex ", std::to_string(id), " already exists!");
        return dynamic_cast<T*>(pool_[found->second].get());
      }
      auto owned = std::make_unique<T>(std::forward<Args>(args)...);
      T* raw = owned.get();
      const auto slot = static_cast<std::uint32_t>(pool_.size());
      pool_.push_back(std::move(owned));
      idToIndex_.emplace(id, slot);
      raw->liveIdx_ = static_cast<std::uint32_t>(liveVec_.size());
      liveVec_.push_back(raw);
      return raw;
    }

    void replace(Vertex* toRemove, Vertex* toAdd) {
#if TESSERA_ASSERTIONS
      if (toAdd == nullptr) throw std::invalid_argument("Cannot remove a nullptr vertex");
      if (toRemove == nullptr) throw std::invalid_argument("Cannot remove a nullptr vertex");
#endif
      remove(toRemove);
      add(toAdd->getId(), toAdd->getCoordinates());
    }

    void remove(Vertex* vertex) noexcept {
      auto id = vertex->getId();
      auto poolIt = idToIndex_.find(id);
      if (poolIt == idToIndex_.end()) return;
      // Slots are not recycled (see the class doc); drop only the live index and
      // the id mapping. The unique_ptr<Vertex> stays in pool_, so a cached Vertex*
      // into this slot stays valid for the lifetime of the VertexList.
      idToIndex_.erase(poolIt);

      // Swap-and-pop from liveVec_ using the index stored on the Vertex
      auto idx = vertex->liveIdx_;
      if (idx < liveVec_.size()) {
        auto lastIdx = static_cast<std::uint32_t>(liveVec_.size() - 1);
        if (idx != lastIdx) {
          liveVec_[idx] = liveVec_[lastIdx];
          liveVec_[idx]->liveIdx_ = idx;
        }
        liveVec_.pop_back();
      }
      vertex->liveIdx_ = UINT32_MAX;
    }

    /// Swap the map keys of two vertices without destroying either object.
    /// Used by Spacetime::swapVertexLabels for Brunekreef relabeling.
    /// The vertices' internal IDs must already have been swapped before calling.
    void swapKeys(std::uint64_t oldId1, std::uint64_t oldId2) {
      auto it1 = idToIndex_.find(oldId1);
      auto it2 = idToIndex_.find(oldId2);
      if (it1 == idToIndex_.end() || it2 == idToIndex_.end()) return;
      auto slot1 = it1->second;
      auto slot2 = it2->second;
      idToIndex_.erase(it1);
      idToIndex_.erase(it2);
      idToIndex_.emplace(oldId2, slot1);
      idToIndex_.emplace(oldId1, slot2);
      // liveIdx_ on the Vertex objects doesn't change — only the map keys do
    }

    std::size_t size() const noexcept {
      return idToIndex_.size();
    }

    /// O(1) — returns the live-vertex vector directly (no copy).
    const std::vector<Vertex*>& liveVector() const noexcept {
      return liveVec_;
    }

    /// Returns a copy of the live-vertex vector.
    std::vector<Vertex*> toVector() const noexcept {
      return liveVec_;
    }

    void reserve(std::size_t nSimplices) noexcept {
      idToIndex_.reserve(nSimplices);
    }

  private:
    /// Stable-address polymorphic storage. The deque never moves existing elements
    /// on growth, and ``unique_ptr`` holds ``Vertex`` subclasses without slicing.
    /// Slots are never recycled; ``remove`` updates only the live index and id map.
    std::deque<std::unique_ptr<Vertex>> pool_;
    std::unordered_map<std::uint64_t, std::uint32_t> idToIndex_; ///< vertex ID → pool slot
    std::vector<Vertex*> liveVec_;                             ///< Flat array of live vertices
};
} // namespace tessera::mesh

#endif //TESSERA_VERTEXLIST_H
