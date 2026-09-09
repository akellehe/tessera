// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/pachner/AddMove.h"

#include <cmath>

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

AddMove::AddMove(Spacetime *st, std::mt19937 *rng, bool relabelEnabled,
                 PachnerMode mode, bool boundaryFixed)
    : PachnerMove(mode, boundaryFixed),
      st_(st), ownedRng_(nullptr), rng_(rng),
      relabelEnabled_(relabelEnabled) {}

AddMove::AddMove(Spacetime *st, std::uint64_t seed, bool relabelEnabled,
                 PachnerMode mode, bool boundaryFixed)
    : PachnerMove(mode, boundaryFixed),
      st_(st),
      ownedRng_(std::make_unique<std::mt19937>(seed)),
      rng_(ownedRng_.get()),
      relabelEnabled_(relabelEnabled) {}

bool AddMove::propose() {
  if (proposed_) return false;
  if (mode_ == PachnerMode::PreGeometric) return proposePreGeometric();
  using namespace pachner_detail;
  const int d = spacetimeDim(*st_);
  const int dPlus1 = d + 1;

  // Pick an N41 top simplex.  Mirrors CDT::getRandomN41Simplex
  // (rejection-sample with linear-scan fallback).
  SimplexPtr sigma = nullptr;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto s = st_->getRandomTopSimplex(*rng_);  // this move's seeded rng (#262)
    if (s && static_cast<int>(s->size()) == dPlus1 && isN41Type(s, d)) {
      sigma = s;
      break;
    }
  }
  if (!sigma) {
    // Fallback linear scan.
    std::vector<SimplexPtr> matches;
    for (const auto &s : st_->getSimplices()) {
      if (static_cast<int>(s->size()) == dPlus1 && isN41Type(s, d))
        matches.push_back(s);
    }
    if (matches.empty()) return false;
    std::uniform_int_distribution<std::size_t> dist(0, matches.size() - 1);
    sigma = matches[dist(*rng_)];
  }

  // Find the spatial facet (vertices all at same time).
  SimplexPtr spatialFacet = nullptr;
  for (const auto &f : sigma->getFacets()) {
    if (f->isSpatial()) { spatialFacet = f; break; }
  }
  if (!spatialFacet) return false;

  // Adjacent simplex sharing this facet, of opposite orientation.
  SimplexPtr sigmaAdj = nullptr;
  for (const auto &cf : spatialFacet->getCofaces()) {
    if (static_cast<int>(cf->size()) == dPlus1 && cf != sigma) {
      sigmaAdj = cf;
      break;
    }
  }
  if (!sigmaAdj) return false;
  if (!isN41Type(sigmaAdj, d)) return false; // TODO: This can never be true for != 4-complexes!

  // Identify the non-spatial ("top" and "bottom") vertices.
  VertexPtr vertA = nullptr, vertB = nullptr;
  for (const auto &v : sigma->getVertices()) {
    if (!spatialFacet->hasVertex(v)) { vertA = v; break; }
  }
  for (const auto &v : sigmaAdj->getVertices()) {
    if (!spatialFacet->hasVertex(v)) { vertB = v; break; }
  }
  if (!vertA || !vertB) return false;

  // Combinatorial deltas: (2,2d) add adds 2d simplices and removes 2.
  dN41_ = 2 * d - 2;

  // Combinatorial prefactor: log(N41 / (N0 + 1))
  // (matches CDT::add).
  double N41 = static_cast<double>(st_->getN41());
  double N0 = static_cast<double>(st_->getVertexCount());
  logPrefactor_ = std::log(N41) - std::log(N0 + 1.0);

  // Capture state for apply.
  sigma_ = sigma;
  sigmaAdj_ = sigmaAdj;
  spatialFacet_ = spatialFacet;
  vertA_ = vertA;
  vertB_ = vertB;
  // Capture vertex tuples for old simplices (for rollback).
  {
    const auto &sv = sigma_->getVertices();
    sigmaVerts_.assign(sv.begin(), sv.end());
    const auto &av = sigmaAdj_->getVertices();
    sigmaAdjVerts_.assign(av.begin(), av.end());
  }
  // Capture spatial vertices (used to build new simplices).
  {
    const auto &sv = spatialFacet_->getVertices();
    spatialVerts_.assign(sv.begin(), sv.end());
  }
  spatialTime_ = spatialFacet_->getTi();

  // Touched vertices: the d+2 around the bipyramid (d spatial + vertA
  // + vertB).  The new vertex doesn't exist yet so isn't included.
  touchedIds_.reserve(d + 2);
  for (const auto &v : spatialVerts_) touchedIds_.push_back(v->getId());
  touchedIds_.push_back(vertA_->getId());
  touchedIds_.push_back(vertB_->getId());

  proposed_ = true;
  return true;
}

std::vector<std::vector<std::uint64_t>> AddMove::sitesOn(const Spacetime &spacetime) {
  std::vector<std::vector<std::uint64_t>> sites;
  for (const auto &topSimplex : spacetime.getTopSimplices()) {
    // The same eligibility `proposePreGeometricOn` applies, asked here so a
    // walk of this list never offers a site that is then refused.
    if (!topSimplex || static_cast<int>(topSimplex->size()) < 3) continue;
    sites.push_back(topSimplex->topTuple());
  }
  return sites;
}

bool AddMove::proposeAt(const std::vector<std::uint64_t> &site) {
  if (mode() != PachnerMode::PreGeometric) return false;
  return proposePreGeometricOn(topSimplexWithIds(*st_, site));
}

bool AddMove::proposePreGeometric() {
  return proposePreGeometricOn(st_->getRandomTopSimplex(*rng_));  // seeded rng (#262)
}

bool AddMove::proposePreGeometricOn(SimplexPtr sigma) {
  if (!sigma) return false;
  const int dPlus1 = static_cast<int>(sigma->size());
  if (dPlus1 < 3) return false;  // need at least a triangle to subdivide

  // A 1→(d+1) stellar subdivision lives entirely inside one cell, so it
  // is always interior — nothing to check for boundary-fixed mode.
  sigma_ = sigma;
  {
    const auto &sv = sigma_->getVertices();
    sigmaVerts_.assign(sv.begin(), sv.end());
  }
  // Causal bookkeeping is meaningless without a foliation.
  dN41_ = 0;
  logPrefactor_ = 0.0;

  touchedIds_.reserve(sigmaVerts_.size());
  for (const auto &v : sigmaVerts_) touchedIds_.push_back(v->getId());

  proposed_ = true;
  return true;
}

bool AddMove::applyPreGeometric() {
  // 1. Fresh coordinate-free interior vertex.
  newVert_ = st_->createVertex();

  // 2. Remove the cell being subdivided.
  st_->removeSimplex(sigma_);

  // 3. Cone the new vertex over each facet of the old cell: drop one
  //    original vertex, append the new vertex.  d+1 new cells in total.
  const std::size_t n = sigmaVerts_.size();
  for (std::size_t skip = 0; skip < n; ++skip) {
    VertexPtrs verts;
    verts.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      if (i != skip) verts.push_back(sigmaVerts_[i]);
    }
    verts.push_back(newVert_);
    // Increasing-id orientation (the new vertex has the largest id, so
    // this is already sorted, but keep it explicit and robust).
    pachner_detail::sortByVertexId(verts);
    auto r = st_->createSimplexTracked(verts);
    if (r.created) createdSimplexVerts_.push_back(verts);
    for (const auto &e : r.newEdges) createdEdges_.push_back(e);
  }

  applied_ = true;
  return true;
}

void AddMove::rollbackPreGeometric() {
  // Remove the d+1 created cells (resolve by verts; see ShiftMove).
  for (const auto &verts : createdSimplexVerts_) {
    if (auto s = st_->findSimplexByVerts(verts)) st_->removeSimplex(s);
  }
  createdSimplexVerts_.clear();

  // Remove the freshly-inserted edges (those from the new vertex).
  pachner_detail::removeAndClearEdges(createdEdges_, st_);

  // Remove the now-isolated new vertex.
  if (newVert_ != nullptr) {
    (void)st_->removeIfIsolated(newVert_);
    newVert_ = nullptr;
  }

  // Recreate the single original cell.
  st_->createSimplexTracked(sigmaVerts_);

  applied_ = false;
}

bool AddMove::apply() {
  if (!proposed_ || applied_) return false;
  if (mode_ == PachnerMode::PreGeometric) return applyPreGeometric();
  const int d = pachner_detail::spacetimeDim(*st_);

  // 1. Create the new vertex at the shared spatial time slice.
  newVert_ = st_->createVertex(std::vector<double>{spatialTime_});

  // 2. Remove the 2 old simplices.
  st_->removeSimplex(sigma_);
  st_->removeSimplex(sigmaAdj_);

  // 3. Build 2d new simplices.  For each spatial sub-face (drop one
  // spatial vertex), create 2 new simplices: one with vertA and one
  // with vertB, both anchored at newVert.
  for (int skip = 0; skip < d; ++skip) {
    VertexPtrs verts1, verts2;
    verts1.reserve(d + 1);
    verts2.reserve(d + 1);
    for (int i = 0; i < d; ++i) {
      if (i != skip) {
        verts1.push_back(spatialVerts_[i]);
        verts2.push_back(spatialVerts_[i]);
      }
    }
    verts1.push_back(newVert_);
    verts1.push_back(vertA_);
    verts2.push_back(newVert_);
    verts2.push_back(vertB_);

    auto r1 = st_->createSimplexTracked(verts1);
    if (r1.created) createdSimplexVerts_.push_back(verts1);
    for (const auto &e : r1.newEdges) createdEdges_.push_back(e);

    auto r2 = st_->createSimplexTracked(verts2);
    if (r2.created) createdSimplexVerts_.push_back(verts2);
    for (const auto &e : r2.newEdges) createdEdges_.push_back(e);
  }

  // 4. Optional vertex relabeling.  Save the swap partner so rollback
  // can un-swap.
  if (relabelEnabled_) {
    VertexPtr partner = st_->getRandomVertex(*rng_);  // seeded rng (#262)
    if (partner && partner->getId() != newVert_->getId()) {
      st_->swapVertexLabels(newVert_, partner);
      swapPartner_ = partner;
    }
  }

  applied_ = true;
  return true;
}

void AddMove::rollback() {
  if (!applied_) return;
  if (mode_ == PachnerMode::PreGeometric) { rollbackPreGeometric(); return; }

  // 1. Reverse the label swap (if any).  ``swapVertexLabels`` is its
  // own inverse — calling it again restores both vertices to their
  // pre-swap IDs and rekeys all dependent fingerprints (edges,
  // simplices) back to their pre-swap state.
  if (swapPartner_ != nullptr) {
    st_->swapVertexLabels(newVert_, swapPartner_);
    swapPartner_ = nullptr;
  }

  // 2. Remove the 2d created simplices.  Their fingerprints are now
  // back to the pre-swap state (involving newVert's auto-assigned ID).
  // Resolve by verts at rollback time — captured SimplexPtrs would be
  // stale if any other move removed-and-recreated these in between
  // (see ShiftMove for the full rationale).
  for (const auto &verts : createdSimplexVerts_) {
    if (auto s = st_->findSimplexByVerts(verts)) st_->removeSimplex(s);
  }
  createdSimplexVerts_.clear();

  // 3. Remove the freshly-inserted edges (mostly: edges from newVert
  // to spatial vertices and to vertA/vertB).
  pachner_detail::removeAndClearEdges(createdEdges_, st_);

  // 4. Remove the new vertex.
  if (newVert_ != nullptr) {
    st_->removeIfIsolated(newVert_);
    newVert_ = nullptr;
  }

  // 5. Recreate the 2 old simplices from their captured vertex tuples.
  st_->createSimplexTracked(sigmaVerts_);
  st_->createSimplexTracked(sigmaAdjVerts_);

  applied_ = false;
}

std::vector<std::uint64_t> AddMove::touchedVertexIds() const {
  return touchedIds_;
}

}  // namespace tessera
