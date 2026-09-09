// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/pachner/FlipMove.h"

#include <cmath>
#include <random>

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

FlipMove::FlipMove(Spacetime *st, std::mt19937 *rng, PachnerMode mode,
                   bool boundaryFixed)
    : PachnerMove(mode, boundaryFixed),
      st_(st), ownedRng_(nullptr), rng_(rng) {}

FlipMove::FlipMove(Spacetime *st, std::uint64_t seed, PachnerMode mode,
                   bool boundaryFixed)
    : PachnerMove(mode, boundaryFixed),
      st_(st),
      ownedRng_(std::make_unique<std::mt19937>(seed)),
      rng_(ownedRng_.get()) {}

bool FlipMove::propose() {
  if (proposed_) return false;
  if (mode_ == PachnerMode::PreGeometric) return proposePreGeometric();
  using namespace pachner_detail;
  const int d = spacetimeDim(*st_);
  const int dPlus1 = d + 1;

  // The generator this move was HANDED, not the complex's own. The
  // no-argument overload reads `Spacetime::rng`, which is initialized from
  // `std::random_device`, so a target drawn through it comes from entropy and
  // no seed can reproduce it (#1013). Every caller already supplies a
  // generator for exactly this purpose.
  SimplexPtr sigma = st_->getRandomTopSimplex(*rng_);
  if (!sigma) return false;

  const auto &facets = sigma->getFacets();
  if (facets.empty()) return false;

  std::uniform_int_distribution<std::size_t> facetDist(0, facets.size() - 1);
  SimplexPtr facet = facets[facetDist(*rng_)];

  // Need exactly 2 d-simplex cofaces.
  std::vector<SimplexPtr> topCofaces;
  for (const auto &cf : facet->getCofaces()) {
    if (static_cast<int>(cf->size()) == dPlus1) topCofaces.push_back(cf);
  }
  if (topCofaces.size() != 2) return false;

  SimplexPtr s1 = topCofaces[0];
  SimplexPtr s2 = topCofaces[1];

  // Collect d+2 unique vertices: d shared + 2 unique.
  auto allVerts = pachner_detail::unionVerticesAcross(
      Simplices{s1, s2});
  if (static_cast<int>(allVerts.size()) != d + 2) return false;

  VertexPtrs shared, unique;
  for (const auto &v : allVerts) {
    if (s1->hasVertex(v) && s2->hasVertex(v)) shared.push_back(v);
    else unique.push_back(v);
  }
  if (static_cast<int>(shared.size()) != d ||
      static_cast<int>(unique.size()) != 2) return false;

  // Old orientation counts.
  int oldN41 = 0, oldN32 = 0;
  for (const auto &s : {s1, s2}) {
    if (isN41Type(s, d)) ++oldN41;
    else if (isN32Type(s, d)) ++oldN32;
  }

  // Build d new simplex tuples: each has both unique + (d-1) of d shared.
  std::vector<VertexPtrs> proposedNew;
  for (int skip = 0; skip < d; ++skip) {
    VertexPtrs nv;
    nv.reserve(dPlus1);
    for (int i = 0; i < d; ++i) {
      if (i != skip) nv.push_back(shared[i]);
    }
    nv.push_back(unique[0]);
    nv.push_back(unique[1]);
    if (static_cast<int>(nv.size()) != dPlus1) return false;
    proposedNew.push_back(std::move(nv));
  }

  // Reject if any new simplex would have a non-CDT orientation.
  for (const auto &nv : proposedNew) {
    if (!isValidCDTOrientation(nv, d)) return false;
  }

  int newN41 = 0, newN32 = 0;
  for (const auto &nv : proposedNew) {
    if (isN41TypeVerts(nv, d)) ++newN41;
    else if (isN32TypeVerts(nv, d)) ++newN32;
  }

  // Capture state for apply / rollback.
  oldSimplices_ = {s1, s2};
  oldSimplexVerts_.reserve(2);
  for (const auto &s : oldSimplices_) {
    const auto &verts = s->getVertices();
    oldSimplexVerts_.emplace_back(verts.begin(), verts.end());
  }
  newSimplexVerts_ = std::move(proposedNew);
  dN41_ = newN41 - oldN41;
  dN32_ = newN32 - oldN32;

  // Combinatorial prefactor (matches CDT::flip): log(N4 / (N4 + d - 2)).
  double N4 = static_cast<double>(st_->getSimplexCount());
  logPrefactor_ = std::log(N4) - std::log(N4 + d - 2);

  touchedIds_.reserve(d + 2);
  for (const auto &v : shared) touchedIds_.push_back(v->getId());
  for (const auto &v : unique) touchedIds_.push_back(v->getId());

  proposed_ = true;
  return true;
}

std::vector<std::vector<std::uint64_t>> FlipMove::sitesOn(const Spacetime &spacetime) {
  std::vector<std::vector<std::uint64_t>> sites;
  for (const auto &topSimplex : spacetime.getTopSimplices()) {
    if (!topSimplex || static_cast<int>(topSimplex->size()) < 3) continue;
    const auto cell = topSimplex->topTuple();
    for (const auto &vertex : topSimplex->getVertices()) {
      if (!vertex) continue;
      std::vector<std::uint64_t> site(cell);
      site.push_back(vertex->getId());
      sites.push_back(std::move(site));
    }
  }
  return sites;
}

bool FlipMove::proposeAt(const std::vector<std::uint64_t> &site) {
  if (mode() != PachnerMode::PreGeometric) return false;
  if (site.size() < 4) return false;  // a cell of >=3 vertices plus the dropped one
  SimplexPtr sigma = topSimplexWithIds(
      *st_, std::vector<std::uint64_t>(site.begin(), site.end() - 1));
  if (!sigma) return false;
  // The drop is an INDEX into the cell's stored vertex order, which is what
  // the body walks; the site names the vertex, so resolve it here.
  const auto &vertices = sigma->getVertices();
  for (std::size_t i = 0; i < vertices.size(); ++i)
    if (vertices[i] && vertices[i]->getId() == site.back())
      return proposePreGeometricOn(sigma, i);
  return false;
}

bool FlipMove::proposePreGeometric() {
  // The generator this move was HANDED, not the complex's own (#1013): the
  // no-argument overload reads `Spacetime::rng`, initialized from
  // `std::random_device`, so a target drawn through it comes from entropy.
  SimplexPtr sigma = st_->getRandomTopSimplex(*rng_);
  if (!sigma) return false;
  const auto &vertsForDrop = sigma->getVertices();
  if (vertsForDrop.empty()) return false;
  std::uniform_int_distribution<std::size_t> dropDist(0, vertsForDrop.size() - 1);
  return proposePreGeometricOn(sigma, dropDist(*rng_));
}

bool FlipMove::proposePreGeometricOn(SimplexPtr sigma, std::size_t drop) {
  using namespace pachner_detail;

  if (!sigma) return false;
  const int dPlus1 = static_cast<int>(sigma->size());
  const int d = dPlus1 - 1;
  if (d < 2) return false;

  // Pick a random facet of sigma combinatorially (drop one vertex).  We
  // deliberately avoid Simplex::getFacets here: it materialises facet
  // simplices that the mesh never garbage-collects, which would litter
  // the complex with orphans after the flip removes sigma.
  const auto &svRef = sigma->getVertices();
  VertexPtrs sigmaV(svRef.begin(), svRef.end());
  if (drop >= sigmaV.size()) return false;
  VertexPtrs facetVerts;
  facetVerts.reserve(static_cast<std::size_t>(d));
  for (std::size_t i = 0; i < sigmaV.size(); ++i) {
    if (i != drop) facetVerts.push_back(sigmaV[i]);
  }

  // The two top cells sharing this facet, found by a local scan so we
  // don't depend on a pre-materialised coface cache (buildExplicit
  // fixtures don't have one).
  auto topCofaces = topCofacesOf(facetVerts, dPlus1);
  if (topCofaces.size() != 2) return false;  // boundary facet (∂W) or non-manifold

  SimplexPtr s1 = topCofaces[0];
  SimplexPtr s2 = topCofaces[1];

  auto allVerts = unionVerticesAcross(Simplices{s1, s2});
  if (static_cast<int>(allVerts.size()) != d + 2) return false;

  VertexPtrs shared, unique;
  for (const auto &v : allVerts) {
    if (s1->hasVertex(v) && s2->hasVertex(v)) shared.push_back(v);
    else unique.push_back(v);
  }
  if (static_cast<int>(shared.size()) != d ||
      static_cast<int>(unique.size()) != 2) return false;

  // Manifold check: the 2→(d+1) flip introduces the apex edge between
  // the two unique vertices.  If that edge already exists the flip would
  // create a degenerate (non-embedded) cell, so reject.
  if (verticesAdjacent(unique[0], unique[1])) return false;

  // Boundary-fixed: the operative facet is interior by construction (it
  // has exactly two top cofaces), so this flip never touches ∂W.  No
  // further restriction is needed — see ticket #112.

  std::vector<VertexPtrs> proposedNew;
  proposedNew.reserve(d);
  for (int skip = 0; skip < d; ++skip) {
    VertexPtrs nv;
    nv.reserve(dPlus1);
    for (int i = 0; i < d; ++i) {
      if (i != skip) nv.push_back(shared[i]);
    }
    nv.push_back(unique[0]);
    nv.push_back(unique[1]);
    if (static_cast<int>(nv.size()) != dPlus1) return false;
    sortByVertexId(nv);  // increasing-id orientation (see helper)
    proposedNew.push_back(std::move(nv));
  }

  oldSimplices_ = {s1, s2};
  oldSimplexVerts_.reserve(2);
  for (const auto &s : oldSimplices_) {
    const auto &verts = s->getVertices();
    oldSimplexVerts_.emplace_back(verts.begin(), verts.end());
  }
  newSimplexVerts_ = std::move(proposedNew);
  // Causal (N41/N32) bookkeeping is meaningless without a foliation; the
  // pre-geometric path reports zero and the move is scored by simplex
  // count only.
  dN41_ = 0;
  dN32_ = 0;
  logPrefactor_ = 0.0;

  touchedIds_.reserve(d + 2);
  for (const auto &v : shared) touchedIds_.push_back(v->getId());
  for (const auto &v : unique) touchedIds_.push_back(v->getId());

  proposed_ = true;
  return true;
}

bool FlipMove::apply() {
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

void FlipMove::rollback() {
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

std::vector<std::uint64_t> FlipMove::touchedVertexIds() const {
  return touchedIds_;
}

}  // namespace tessera
