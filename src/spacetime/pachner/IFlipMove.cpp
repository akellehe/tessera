// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/pachner/IFlipMove.h"

#include <cmath>
#include <random>

#include "mesh/Edge.h"
#include "mesh/Simplex.h"
#include "mesh/TemporalOrientation.h"
#include "mesh/Vertex.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::graph {}
namespace tessera::mesh {}
namespace tessera::observables {}
namespace tessera::quantum {}
namespace tessera::simulations {}
namespace tessera::spacetime {
using namespace ::tessera::mesh;
using namespace ::tessera::graph;
using namespace ::tessera::observables;
using namespace ::tessera::simulations;
using namespace ::tessera::quantum;

IFlipMove::IFlipMove(Spacetime *st, std::mt19937 *rng, PachnerMode mode,
                     bool boundaryFixed)
    : PachnerMove(mode, boundaryFixed),
      st_(st), ownedRng_(nullptr), rng_(rng) {}

IFlipMove::IFlipMove(Spacetime *st, std::uint64_t seed, PachnerMode mode,
                     bool boundaryFixed)
    : PachnerMove(mode, boundaryFixed),
      st_(st),
      ownedRng_(std::make_unique<std::mt19937>(seed)),
      rng_(ownedRng_.get()) {}

bool IFlipMove::propose() {
  if (proposed_) return false;
  using namespace pachner_detail;

  // The generator this move was HANDED, not the complex's own. The
  // no-argument overload reads `Spacetime::rng`, which is initialized from
  // `std::random_device`, so a target drawn through it comes from entropy and
  // no seed can reproduce it (#1013). Every caller already supplies a
  // generator for exactly this purpose.
  SimplexPtr sigma = st_->getRandomTopSimplex(*rng_);
  if (!sigma) return false;

  // Pre-geometric complexes can carry a metric whose dimension differs
  // from the manifold's, so read the move dimension off the actual top
  // cell in that mode; CDT keeps using the (foliated) metric dimension.
  const int d = (mode_ == PachnerMode::PreGeometric)
                    ? static_cast<int>(sigma->size()) - 1
                    : spacetimeDim(*st_);
  const int dPlus1 = d + 1;
  if (d < 2) return false;

  const auto &edges = sigma->getEdges();
  if (edges.empty()) return false;
  std::uniform_int_distribution<std::size_t> edgeDist(0, edges.size() - 1);
  EdgePtr edge = edges[edgeDist(*rng_)];

  VertexPtr v1 = edge->getSource();
  VertexPtr v2 = edge->getTarget();

  // Find all top simplices containing both endpoints. The edge already indexes
  // the simplices registered on it (Spacetime::registerSimplex mirrors every
  // simplex into each of its edges), and that index is what bounds the work:
  // a vertex's incidence list grows with the four-volume -- measured, mean 80
  // simplices per vertex at N4 = 24k and 148 at N4 = 50k -- while an edge's
  // does not (#970).
  std::vector<SimplexPtr> sharing;
  for (const auto &s : edge->simplices()) {
    if (static_cast<int>(s->size()) == dPlus1) sharing.push_back(s);
  }
  if (static_cast<int>(sharing.size()) != d) return false;

  // Collect all vertices: should be d+2 total.
  auto allVerts = pachner_detail::unionVerticesAcross(sharing);
  if (static_cast<int>(allVerts.size()) != d + 2) return false;

  // Separate shared (the 2 edge endpoints) and unique (the d others).
  VertexPtrs shared, unique;
  for (const auto &v : allVerts) {
    if (v->getId() == v1->getId() || v->getId() == v2->getId())
      shared.push_back(v);
    else
      unique.push_back(v);
  }
  if (shared.size() != 2 || static_cast<int>(unique.size()) != d) return false;

  // Boundary-fixed: a d→2 flip collapses the shared edge (v1,v2).  If
  // that edge lies on ∂W the move would change the boundary, so reject.
  if (boundaryFixed_ &&
      !pachner_detail::isInteriorEdge(v1, v2, dPlus1)) return false;

  // Pre-geometric manifold check: the welded (d-1)-facet is the simplex
  // on the link (unique) vertices.  If any current top cell already
  // contains all of them that facet is already present, so the weld
  // would over-share it (>2 cofaces) and tear the pseudomanifold.  This
  // subsumes the CDT "new cells already exist" check below (a pre-
  // existing cell on those verts would be counted here).
  if (mode_ == PachnerMode::PreGeometric &&
      pachner_detail::topCofaceCount(unique, dPlus1) != 0) return false;

  // Manifold check: would either proposed new simplex already exist?
  // We look for a top simplex incident to unique[0] that contains all
  // unique vertices plus one shared vertex but is NOT one of the d
  // we're about to remove.  Matches CDT::iflip's check exactly.
  for (int i = 0; i < 2; ++i) {
    for (const auto &s : unique[0]->getSimplices()) {
      if (static_cast<int>(s->size()) != dPlus1) continue;
      bool isSharing = false;
      for (const auto &sh : sharing) {
        if (s == sh) { isSharing = true; break; }
      }
      if (isSharing) continue;
      if (!s->hasVertex(shared[i])) continue;
      bool hasAll = true;
      for (const auto &u : unique) {
        if (!s->hasVertex(u)) { hasAll = false; break; }
      }
      if (hasAll) return false;  // would create duplicate
    }
  }

  // Old orientation counts.
  int oldN41 = 0, oldN32 = 0;
  for (const auto &s : sharing) {
    if (isN41Type(s, d)) ++oldN41;
    else if (isN32Type(s, d)) ++oldN32;
  }

  // Build 2 new simplex tuples: each has all d unique + 1 of 2 shared.
  std::vector<VertexPtrs> proposedNew;
  for (int i = 0; i < 2; ++i) {
    VertexPtrs nv(unique.begin(), unique.end());
    nv.push_back(shared[i]);
    if (static_cast<int>(nv.size()) != dPlus1) return false;
    // Increasing-id orientation so the mutated complex's homology is
    // well-defined (see pachner_detail::sortByVertexId).  CDT cell order
    // is orientation-bearing, so only re-sort on the pre-geometric path.
    if (mode_ == PachnerMode::PreGeometric) sortByVertexId(nv);
    proposedNew.push_back(std::move(nv));
  }

  // The CDT orientation/time-slice guard is dropped in pre-geometric
  // mode; the manifold check above stands in for it.
  if (mode_ == PachnerMode::CDT) {
    for (const auto &nv : proposedNew) {
      if (!isValidCDTOrientation(nv, d)) return false;
    }
  }

  int newN41 = 0, newN32 = 0;
  for (const auto &nv : proposedNew) {
    if (isN41TypeVerts(nv, d)) ++newN41;
    else if (isN32TypeVerts(nv, d)) ++newN32;
  }

  oldSimplices_ = std::move(sharing);
  oldSimplexVerts_.reserve(oldSimplices_.size());
  for (const auto &s : oldSimplices_) {
    const auto &verts = s->getVertices();
    oldSimplexVerts_.emplace_back(verts.begin(), verts.end());
  }
  newSimplexVerts_ = std::move(proposedNew);
  dN41_ = newN41 - oldN41;
  dN32_ = newN32 - oldN32;

  // Combinatorial prefactor (matches CDT::iflip): log(N4 / (N4 - d + 2)).
  // The Metropolis prefactor is a CDT-sampling quantity; the
  // pre-geometric path is not Metropolis-driven, so it reports 0.
  if (mode_ == PachnerMode::PreGeometric) {
    logPrefactor_ = 0.0;
  } else {
    double N4 = static_cast<double>(st_->getSimplexCount());
    logPrefactor_ = std::log(N4) - std::log(N4 - d + 2);
  }

  touchedIds_.reserve(d + 2);
  for (const auto &v : shared) touchedIds_.push_back(v->getId());
  for (const auto &v : unique) touchedIds_.push_back(v->getId());

  proposed_ = true;
  return true;
}

bool IFlipMove::apply() {
  if (!proposed_ || applied_) return false;

  for (const auto &s : oldSimplices_) st_->removeSimplex(s);

  createdSimplexVerts_.reserve(newSimplexVerts_.size());
  for (const auto &nv : newSimplexVerts_) {
    auto r = st_->createSimplexTracked(nv);
    if (r.created) createdSimplexVerts_.push_back(nv);
    for (const auto &e : r.newEdges) createdEdges_.push_back(e);
  }

  applied_ = true;
  return true;
}

void IFlipMove::rollback() {
  if (!applied_) return;

  // Resolve created simplices by verts at rollback time (see ShiftMove).
  for (const auto &verts : createdSimplexVerts_) {
    if (auto s = st_->findSimplexByVerts(verts)) st_->removeSimplex(s);
  }
  createdSimplexVerts_.clear();

  pachner_detail::removeAndClearEdges(createdEdges_, st_);

  for (const auto &verts : oldSimplexVerts_) {
    st_->createSimplexTracked(verts);
  }

  applied_ = false;
}

std::vector<std::uint64_t> IFlipMove::touchedVertexIds() const {
  return touchedIds_;
}

}  // namespace tessera
