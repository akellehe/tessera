// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/pachner/ShiftMove.h"

#include <algorithm>

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

ShiftMove::ShiftMove(Spacetime *st, std::mt19937 *rng, PachnerMode mode,
                     bool boundaryFixed)
    : PachnerMove(mode, boundaryFixed),
      st_(st), ownedRng_(nullptr), rng_(rng) {}

ShiftMove::ShiftMove(Spacetime *st, std::uint64_t seed, PachnerMode mode,
                     bool boundaryFixed)
    : PachnerMove(mode, boundaryFixed),
      st_(st),
      ownedRng_(std::make_unique<std::mt19937>(seed)),
      rng_(ownedRng_.get()) {}

bool ShiftMove::propose() {
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
  const int hingeSize = d - 1;
  if (hingeSize < 1) return false;

  const auto &sigmaVertsRef = sigma->getVertices();
  if (static_cast<int>(sigmaVertsRef.size()) < hingeSize) return false;
  VertexPtrs sigmaVerts(sigmaVertsRef.begin(), sigmaVertsRef.end());
  std::shuffle(sigmaVerts.begin(), sigmaVerts.end(), *rng_);
  VertexPtrs faceVerts(sigmaVerts.begin(), sigmaVerts.begin() + hingeSize);

  // All d-simplices containing all (d-1) face vertices.
  std::vector<SimplexPtr> sharing;
  for (const auto &s : faceVerts[0]->getSimplices()) {
    if (static_cast<int>(s->size()) != dPlus1) continue;
    bool containsAll = true;
    for (int i = 1; i < hingeSize; ++i) {
      if (!s->hasVertex(faceVerts[i])) { containsAll = false; break; }
    }
    if (containsAll) sharing.push_back(s);
  }
  if (static_cast<int>(sharing.size()) != hingeSize) return false;

  // Collect all unique vertices across the sharing simplices (must be d+2).
  auto allVerts = pachner_detail::unionVerticesAcross(sharing);
  if (static_cast<int>(allVerts.size()) != d + 2) return false;

  // Separate shared (in all of `sharing`) and unique (in only one).
  VertexPtrs sharedVerts, uniqueVerts;
  for (const auto &v : allVerts) {
    bool inAll = true;
    for (const auto &s : sharing) {
      if (!s->hasVertex(v)) { inAll = false; break; }
    }
    if (inAll) sharedVerts.push_back(v);
    else       uniqueVerts.push_back(v);
  }
  if (static_cast<int>(sharedVerts.size()) != hingeSize ||
      static_cast<int>(uniqueVerts.size()) != hingeSize) return false;

  // Boundary-fixed: only fire when the whole affected region is interior
  // (no involved cell carries a face on ∂W), so the shift cannot move
  // the boundary.  Conservative but always correct.
  if (boundaryFixed_) {
    for (const auto &s : sharing) {
      const auto &sv = s->getVertices();
      for (std::size_t skip = 0; skip < sv.size(); ++skip) {
        VertexPtrs fv;
        fv.reserve(sv.size() - 1);
        for (std::size_t i = 0; i < sv.size(); ++i) {
          if (i != skip) fv.push_back(sv[i]);
        }
        if (isBoundaryFacet(fv, dPlus1)) return false;
      }
    }
  }

  // Old orientation counts.
  int oldN41 = 0, oldN32 = 0;
  for (const auto &s : sharing) {
    if (isN41Type(s, d)) ++oldN41;
    else if (isN32Type(s, d)) ++oldN32;
  }

  // Build new simplex vertex tuples: each takes (hingeSize-1) shared
  // + all unique = (d-2) + (d-1) = 2d-3 = (d+1)-2... wait, dPlus1.
  std::vector<VertexPtrs> proposedNew;
  for (int skip = 0; skip < hingeSize; ++skip) {
    VertexPtrs nv;
    nv.reserve(dPlus1);
    for (int i = 0; i < hingeSize; ++i) {
      if (i != skip) nv.push_back(sharedVerts[i]);
    }
    for (const auto &u : uniqueVerts) nv.push_back(u);
    if (static_cast<int>(nv.size()) != dPlus1) return false;
    // Increasing-id orientation on the pre-geometric path (see
    // pachner_detail::sortByVertexId); CDT cell order is orientation-
    // bearing and left untouched.
    if (mode_ == PachnerMode::PreGeometric) sortByVertexId(nv);
    proposedNew.push_back(std::move(nv));
  }

  // Reject if any new simplex would have a non-CDT orientation.  Dropped
  // in pre-geometric mode (no foliation to respect).
  if (mode_ == PachnerMode::CDT) {
    for (const auto &nv : proposedNew) {
      if (!isValidCDTOrientation(nv, d)) return false;
    }
  }

  // New orientation counts.
  int newN41 = 0, newN32 = 0;
  for (const auto &nv : proposedNew) {
    if (isN41TypeVerts(nv, d)) ++newN41;
    else if (isN32TypeVerts(nv, d)) ++newN32;
  }

  // Capture state for apply / rollback.
  oldSimplices_ = std::move(sharing);
  oldSimplexVerts_.reserve(oldSimplices_.size());
  for (const auto &s : oldSimplices_) {
    const auto &verts = s->getVertices();
    oldSimplexVerts_.emplace_back(verts.begin(), verts.end());
  }
  newSimplexVerts_ = std::move(proposedNew);
  dN41_ = newN41 - oldN41;
  dN32_ = newN32 - oldN32;

  touchedIds_.reserve(d + 2);
  for (const auto &v : sharedVerts) touchedIds_.push_back(v->getId());
  for (const auto &v : uniqueVerts) touchedIds_.push_back(v->getId());

  proposed_ = true;
  return true;
}

bool ShiftMove::apply() {
  if (!proposed_ || applied_) return false;

  // Remove the 3 old simplices.
  for (const auto &s : oldSimplices_) st_->removeSimplex(s);

  // Create the 3 new ones, recording the verts of those we actually
  // created (skipping any that already existed and were deduped).  We
  // store verts rather than SimplexPtr so rollback survives another
  // move removing the underlying Simplex in between.
  createdSimplexVerts_.reserve(newSimplexVerts_.size());
  for (const auto &nv : newSimplexVerts_) {
    auto r = st_->createSimplexTracked(nv);
    if (r.created) createdSimplexVerts_.push_back(nv);
    for (const auto &e : r.newEdges) createdEdges_.push_back(e);
  }

  applied_ = true;
  return true;
}

void ShiftMove::rollback() {
  if (!applied_) return;

  // 1. Remove the simplices we created.  Resolve each by its vertex
  // tuple at rollback time — the SimplexPtr captured at apply time
  // can be stale (another move may have removed-and-recreated the
  // same-verts simplex with a fresh allocation; the old pointer would
  // be dangling and trigger a use-after-free in removeSimplex's
  // swap-and-pop on stale vecIdx_).
  for (const auto &verts : createdSimplexVerts_) {
    if (auto s = st_->findSimplexByVerts(verts)) {
      st_->removeSimplex(s);
    }
  }
  createdSimplexVerts_.clear();

  // 2. Remove the edges we freshly inserted.
  // (Pre-existing edges that the new simplices reused are left alone.)
  pachner_detail::removeAndClearEdges(createdEdges_, st_);

  // 3. Recreate the old simplices from their captured vertex tuples.
  // ``createSimplexTracked`` re-inserts edges that the removed simplices
  // had been using and that no other simplex carries.  Pre-existing
  // edges (still in EdgeList from other simplices) are deduped.
  for (const auto &verts : oldSimplexVerts_) {
    st_->createSimplexTracked(verts);
  }

  applied_ = false;
}

std::vector<std::uint64_t> ShiftMove::touchedVertexIds() const {
  return touchedIds_;
}

}  // namespace tessera
