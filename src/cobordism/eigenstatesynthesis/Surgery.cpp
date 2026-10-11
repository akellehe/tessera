// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

// EigenstateSynthesis (include/cobordism/EigenstateSynthesis.h): surgery, the
// topology-changing interior remove move. One of the translation units that
// define the class's members by topic.

#include "cobordism/EigenstateSynthesis.h"
#include "quantum/ChoiJamiolkowski.h"

#include <numeric>
#include <limits>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cobordism/ChainComplex.h"
#include "mesh/Edge.h"
#include "mesh/EdgeList.h"
#include "mesh/Fingerprint.h"
#include "mesh/Simplex.h"
#include "mesh/Vertex.h"
#include "mesh/VertexList.h"
#include "spacetime/Metric.h"
#include "spacetime/Signature.h"
#include "spacetime/Spacetime.h"
#include "spacetime/pachner/AddMove.h"

#ifdef TESSERA_CUDA
#include "cuda/eigenstate_cuda.h"
#endif

namespace tessera::cobordism {

using cd = std::complex<double>;

// === Surgery: the topology-changing interior remove move ===

std::vector<std::vector<std::uint64_t>>
EigenstateSynthesis::interiorTopCells() const {
  std::vector<std::vector<std::uint64_t>> cells;
  if (!st_) return cells;
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts == 0) return cells;
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  for (const auto s : st_->getSimplices()) {
    if (s == nullptr || s->size() != topVerts) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(topVerts);
    bool allInterior = true;
    for (const auto v : s->getVertices()) {
      if (v == nullptr) { allInterior = false; break; }
      ids.push_back(v->getId());
      if (bverts.find(v->getId()) != bverts.end()) allInterior = false;
    }
    if (!allInterior || ids.size() != topVerts) continue;
    std::sort(ids.begin(), ids.end());
    cells.push_back(std::move(ids));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

bool EigenstateSynthesis::removeInteriorCell(
    const std::vector<std::uint64_t> &cell) {
  if (!st_) return false;
  const int d = st_->getMetric()->getSignature()->getDimensions();
  const std::size_t topVerts = (d >= 0) ? static_cast<std::size_t>(d) + 1 : 0;
  if (topVerts < 2 || cell.size() != topVerts) return false;
  std::vector<std::uint64_t> want(cell.begin(), cell.end());
  std::sort(want.begin(), want.end());

  // The cell must be interior: no boundary vertex (so no ∂W face is removed).
  const std::unordered_set<std::uint64_t> bverts(
      boundaryVertexIdsSorted_.begin(), boundaryVertexIdsSorted_.end());
  for (const std::uint64_t id : want)
    if (bverts.find(id) != bverts.end()) return false;

  // Locate the matching top simplex, and collect the other top cells' vertex
  // sets (to tell which of `want`'s edges remain covered after removal). Refuse
  // to remove the last top cell of the top dimension (it would drop the complex
  // dimension and promote orphan facets to top cells).
  ::tessera::mesh::Simplex *target = nullptr;
  std::vector<std::vector<std::uint64_t>> otherTop;
  for (const auto s : st_->getSimplices()) {
    if (s == nullptr || s->size() != topVerts) continue;
    std::vector<std::uint64_t> ids;
    ids.reserve(topVerts);
    for (const auto v : s->getVertices())
      if (v != nullptr) ids.push_back(v->getId());
    std::sort(ids.begin(), ids.end());
    if (target == nullptr && ids == want)
      target = s;
    else
      otherTop.push_back(std::move(ids));
  }
  if (target == nullptr || otherTop.empty()) return false;

  // An edge {u,v} of the cell is orphaned iff no other top cell contains both
  // endpoints. Map endpoint pairs -> Edge* for the orphaned ones.
  std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *>
      edgeByPair;
  for (const auto e : edges_) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    edgeByPair[{std::min(a, b), std::max(a, b)}] = e;
  }
  const auto covered = [&](std::uint64_t u, std::uint64_t v) {
    for (const auto &c : otherTop) {
      const bool hu = std::find(c.begin(), c.end(), u) != c.end();
      const bool hv = std::find(c.begin(), c.end(), v) != c.end();
      if (hu && hv) return true;
    }
    return false;
  };

  Removal rem;
  rem.cell = want;
  std::vector<::tessera::mesh::Edge *> toRemove;
  for (std::size_t i = 0; i + 1 < want.size(); ++i)
    for (std::size_t j = i + 1; j < want.size(); ++j) {
      const std::uint64_t u = want[i];
      const std::uint64_t v = want[j];
      if (covered(u, v)) continue;  // edge survives in another top cell
      // edgeByPair is keyed {min, max}; canonicalize here too rather than
      // leaning on `want` having been sorted fifty lines up. A miss would
      // silently leave an orphaned edge in place and out of removedEdges,
      // so the rollback could not undo the move.
      const auto it = edgeByPair.find({std::min(u, v), std::max(u, v)});
      if (it == edgeByPair.end()) continue;  // already absent
      rem.removedEdges.emplace_back(u, v, it->second->getLength(),
                                    it->second->getPhase());
      toRemove.push_back(it->second);
    }

  // Snapshot ∂W (id-pair -> (complex w, theta)) for the bit-exact check. The
  // whole complex l2, not (Re, phase): the dW invariant covers Im corruption.
  std::map<std::pair<std::uint64_t, std::uint64_t>,
           std::pair<std::complex<double>, std::complex<double>>>
      boundaryBefore;
  for (const auto i : boundaryEdgeIdx_) {
    const std::uint64_t a = edges_[i]->getSource()->getId();
    const std::uint64_t b = edges_[i]->getTarget()->getId();
    boundaryBefore[{std::min(a, b), std::max(a, b)}] = {
        (edges_[i]->getLength() * edges_[i]->getLength()), edges_[i]->getPhase()};
  }

  // Mutate: drop the top cell, then its orphaned edges.
  st_->removeSimplex(target);
  for (auto *e : toRemove)
    if (e != nullptr) st_->removeEdge(e);

  laplacian_ = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  capture();
  classifyBoundary();

  // ∂W must be preserved bit-exactly: every previously-boundary edge still
  // present with the same weight/phase. Newly exposed boundary edges — the
  // opened hole — are allowed, so this is a subset check, not equality.
  bool valid = true;
  std::map<std::pair<std::uint64_t, std::uint64_t>,
           std::pair<std::complex<double>, std::complex<double>>>
      liveWeights;
  for (const auto e : edges_) {
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    const std::pair<std::uint64_t, std::uint64_t> key{std::min(a, b),
                                                      std::max(a, b)};
    liveWeights[key] = {(e->getLength() * e->getLength()), e->getPhase()};
  }
  for (const auto &[key, wp] : boundaryBefore) {
    const auto it = liveWeights.find(key);
    if (it == liveWeights.end() || it->second.first != wp.first ||
        it->second.second != wp.second) {
      valid = false;
      break;
    }
  }

  if (!valid) {
    applyRestore(rem);
    laplacian_ = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
    capture();
    classifyBoundary();
    return false;
  }
  removals_.push_back(std::move(rem));
  return true;
}

bool EigenstateSynthesis::applyRestore(const Removal &rem) {
  if (!st_) return false;
  std::unordered_map<std::uint64_t, ::tessera::mesh::Vertex *> idToVert;
  for (const auto v : st_->getVertexList()->toVector())
    if (v != nullptr) idToVert.emplace(v->getId(), v);
  ::tessera::mesh::VertexPtrs verts;
  verts.reserve(rem.cell.size());
  for (const std::uint64_t id : rem.cell) {
    const auto it = idToVert.find(id);
    if (it == idToVert.end()) return false;  // a cell vertex vanished
    verts.push_back(it->second);
  }
  // Re-create the top cell; createSimplexTracked rebuilds exactly the missing
  // edges (the orphaned ones removed above), leaving surviving edges untouched.
  st_->createSimplexTracked(verts);
  // Restore the removed edges' weights/phases bit-exactly.
  std::map<std::pair<std::uint64_t, std::uint64_t>, ::tessera::mesh::Edge *>
      edgeByPair;
  for (const auto e : st_->getEdgeList()->toVector()) {
    if (e == nullptr || e->getSource() == nullptr || e->getTarget() == nullptr)
      continue;
    const std::uint64_t a = e->getSource()->getId();
    const std::uint64_t b = e->getTarget()->getId();
    edgeByPair[{std::min(a, b), std::max(a, b)}] = e;
  }
  for (const auto &[u, v, w, theta] : rem.removedEdges) {
    const auto it = edgeByPair.find({std::min(u, v), std::max(u, v)});
    if (it != edgeByPair.end()) {
      it->second->setLength(w);  // the recorded complex LENGTH, bit-exact
      it->second->setPhase(theta);
    }
  }
  return true;
}

bool EigenstateSynthesis::restoreLastRemoval() {
  if (!st_ || removals_.empty()) return false;
  const Removal rem = std::move(removals_.back());
  removals_.pop_back();
  applyRestore(rem);
  laplacian_ = HodgeLaplacian(st_, HodgeLaplacian::defaultWeightConvention(), metricSource_);
  capture();
  classifyBoundary();
  return true;
}

}  // namespace tessera::cobordism
