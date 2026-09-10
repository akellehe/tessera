// Copyright (c) 2026 Twin Vector Labs LLC. All rights reserved.
#include "mesh/EdgeList.h"
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

std::uint32_t EdgeList::allocSlot(const VertexPtr &source, const VertexPtr &target, std::complex<double> length) {
  // The factory layer speaks the edge's one degree of freedom: the complex LENGTH
  // (real = spacelike, imaginary = timelike). l^2 is derived by squaring at the point
  // of use and is never stored, so the two cannot fall out of sync (#639).
  const std::complex<double> squared = length;
  std::uint32_t slot;
  if (!freeSlots_.empty()) {
    slot = freeSlots_.back();
    freeSlots_.pop_back();
    pool_[slot] = Edge(source, target, squared);
  } else {
    slot = static_cast<std::uint32_t>(pool_.size());
    pool_.emplace_back(source, target, squared);
  }
  return slot;
}

EdgePtr EdgeList::add(const VertexPtr &source, const VertexPtr &target) {
  std::uint64_t fp = Fingerprint::mix64(source->getId()) ^ Fingerprint::mix64(target->getId());
  auto it = fpToSlot_.find(fp);
  if (it != fpToSlot_.end()) return &pool_[it->second];
  return getOrInsert(source, target, 0.0);
}

EdgePtr EdgeList::add(const VertexPtr &source, const VertexPtr &target, std::complex<double> length) noexcept {
  std::uint64_t fp = Fingerprint::mix64(source->getId()) ^ Fingerprint::mix64(target->getId());
  auto it = fpToSlot_.find(fp);
  if (it != fpToSlot_.end()) return &pool_[it->second];
  return getOrInsert(source, target, length);
}

std::pair<EdgePtr, bool> EdgeList::tryAdd(const VertexPtr &source, const VertexPtr &target,
                                          std::complex<double> length) {
  std::uint64_t fp = Fingerprint::mix64(source->getId()) ^ Fingerprint::mix64(target->getId());
  auto it = fpToSlot_.find(fp);
  if (it != fpToSlot_.end()) return {&pool_[it->second], false};
  return {getOrInsert(source, target, length), true};
}

EdgePtr EdgeList::getOrInsert(const VertexPtr &source, const VertexPtr &target, std::complex<double> length) {
  if (source->getId() == target->getId()) {
    throw std::runtime_error("You cannot create an edge from a vertex to itself.");
  }
  std::uint64_t fp = Fingerprint::mix64(source->getId()) ^ Fingerprint::mix64(target->getId());
  auto it = fpToSlot_.find(fp);
  if (it != fpToSlot_.end()) return &pool_[it->second];

  auto slot = allocSlot(source, target, length);
  fpToSlot_.emplace(fp, slot);
  EdgePtr raw = &pool_[slot];
  raw->liveIdx_ = static_cast<std::uint32_t>(liveVec_.size());
  liveVec_.push_back(raw);
  return raw;
}

std::uint64_t EdgeList::keyOf(const Edge &edge) noexcept {
  return Fingerprint::mix64(edge.getSource()->getId()) ^
         Fingerprint::mix64(edge.getTarget()->getId());
}

void EdgeList::remove(const EdgePtr &edge) noexcept {
  auto fp = keyOf(*edge);
  auto it = fpToSlot_.find(fp);
  if (it == fpToSlot_.end()) return;
  freeSlots_.push_back(it->second);
  fpToSlot_.erase(it);

  // Swap-and-pop from liveVec_ using the index stored on the Edge
  auto idx = edge->liveIdx_;
  if (idx < liveVec_.size()) {
    auto lastIdx = static_cast<std::uint32_t>(liveVec_.size() - 1);
    if (idx != lastIdx) {
      liveVec_[idx] = liveVec_[lastIdx];
      liveVec_[idx]->liveIdx_ = idx;
    }
    liveVec_.pop_back();
  }
  edge->liveIdx_ = UINT32_MAX;
}

void EdgeList::rekeyEdge(std::uint64_t oldFp, std::uint64_t newFp) {
  if (oldFp == newFp) return;
  auto it = fpToSlot_.find(oldFp);
  if (it == fpToSlot_.end()) return;
  auto slot = it->second;
  fpToSlot_.erase(it);
  fpToSlot_.emplace(newFp, slot);
  // liveIdx_ on the Edge object doesn't change — only the fingerprint key does
}

const Edges &EdgeList::toVector() const noexcept {
  return liveVec_;
}

std::size_t EdgeList::size() const {
  // The live vector is what toVector() hands out, so it is what size() has to
  // count. The lookup map can hold fewer entries than there are live edges if a
  // key is ever contended, and reporting the map's size hides exactly that.
  return liveVec_.size();
}

EdgePtr EdgeList::get(const std::uint64_t &fingerprint) {
  auto it = fpToSlot_.find(fingerprint);
  if (it == fpToSlot_.end()) throw std::out_of_range("Edge not found");
  return &pool_[it->second];
}

void EdgeList::reserve(std::size_t nSimplices) {
  fpToSlot_.reserve(nSimplices);
}

} // namespace tessera::mesh
