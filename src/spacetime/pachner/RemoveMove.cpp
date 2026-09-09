// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#include "spacetime/pachner/RemoveMove.h"

#include <algorithm>
#include <cmath>
#include <map>

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

RemoveMove::RemoveMove(Spacetime *st, std::mt19937 *rng, PachnerMode mode,
                       bool boundaryFixed)
    : PachnerMove(mode, boundaryFixed),
      st_(st), ownedRng_(nullptr), rng_(rng) {}

RemoveMove::RemoveMove(Spacetime *st, std::uint64_t seed, PachnerMode mode,
                       bool boundaryFixed)
    : PachnerMove(mode, boundaryFixed),
      st_(st),
      ownedRng_(std::make_unique<std::mt19937>(seed)),
      rng_(ownedRng_.get()) {}

bool RemoveMove::propose() {
  if (proposed_) return false;
  if (mode_ == PachnerMode::PreGeometric) return proposePreGeometric();
  using namespace pachner_detail;
  const int d = spacetimeDim(*st_);
  const int dPlus1 = d + 1;
  const int requiredOrder = 2 * d;

  // Mirrors CDT::remove's blind-guessing strategy.
  // Use the spacetime's RNG-free getRandomVertex for parity with the
  // existing code; we don't get to influence vertex selection here.
  // (CDT::remove does the same.)
  // The generator this move was HANDED, not the complex's own. The
  // no-argument overload reads `Spacetime::rng`, which is initialized from
  // `std::random_device`, so a target drawn through it comes from entropy and
  // no seed can reproduce it (#1013). Every caller already supplies a
  // generator for exactly this purpose.
  VertexPtr v = st_->getRandomVertex(*rng_);
  if (!v) return false;

  std::vector<SimplexPtr> incident;
  for (const auto &s : v->getSimplices()) {
    if (static_cast<int>(s->size()) == dPlus1) incident.push_back(s);
  }
  if (static_cast<int>(incident.size()) != requiredOrder) return false;

  // Verify structural prerequisites: all incident must be N41-type.
  for (const auto &s : incident) {
    if (!isN41Type(s, d)) return false;
  }

  // Collect d+2 other vertices (excluding v) and their per-simplex
  // counts.  Spatial vertices appear in 2(d-1) of the 2d simplices;
  // the 2 non-spatial each appear in d.
  std::map<std::uint64_t, VertexPtr> otherVerts;
  std::map<std::uint64_t, int> vertCounts;
  for (const auto &s : incident) {
    for (const auto &vert : s->getVertices()) {
      if (vert->getId() == v->getId()) continue;
      otherVerts[vert->getId()] = vert;
      vertCounts[vert->getId()]++;
    }
  }
  if (static_cast<int>(otherVerts.size()) != d + 2) return false;

  VertexPtr vertA = nullptr, vertB = nullptr;
  VertexPtrs spatialVerts;
  for (const auto &[vid, count] : vertCounts) {
    if (count == d) {
      if (!vertA) vertA = otherVerts[vid];
      else if (!vertB) vertB = otherVerts[vid];
      else return false;
    } else if (count == 2 * (d - 1)) {
      spatialVerts.push_back(otherVerts[vid]);
    } else {
      return false;
    }
  }
  if (!vertA || !vertB || static_cast<int>(spatialVerts.size()) != d)
    return false;

  // Combinatorial deltas.
  dN41_ = -(2 * d - 2);

  // Combinatorial prefactor (matches CDT::remove): log(N0 / N41after).
  double N41 = static_cast<double>(st_->getN41());
  double N0 = static_cast<double>(st_->getVertexCount());
  double N41after = N41 + dN41_;
  if (N41after <= 0) return false;
  logPrefactor_ = std::log(N0) - std::log(N41after);

  // Capture for apply.
  v_ = v;
  incident_ = std::move(incident);
  vertA_ = vertA;
  vertB_ = vertB;
  spatialVerts_ = std::move(spatialVerts);

  // Capture vertex tuples for rollback.
  incidentVerts_.reserve(incident_.size());
  for (const auto &s : incident_) {
    const auto &verts = s->getVertices();
    incidentVerts_.emplace_back(verts.begin(), verts.end());
  }

  // Touched: v + d spatial + vertA + vertB.
  touchedIds_.reserve(d + 3);
  touchedIds_.push_back(v_->getId());
  for (const auto &sv : spatialVerts_) touchedIds_.push_back(sv->getId());
  touchedIds_.push_back(vertA_->getId());
  touchedIds_.push_back(vertB_->getId());

  proposed_ = true;
  return true;
}

bool RemoveMove::proposePreGeometric() {
  // The generator this move was HANDED, not the complex's own. The
  // no-argument overload reads `Spacetime::rng`, which is initialized from
  // `std::random_device`, so a target drawn through it comes from entropy and
  // no seed can reproduce it (#1013). Every caller already supplies a
  // generator for exactly this purpose.
  VertexPtr v = st_->getRandomVertex(*rng_);
  if (!v) return false;

  // Read the top-cell vertex count off v's incident cells.
  int dPlus1 = 0;
  for (const auto &s : v->getSimplices())
    dPlus1 = std::max(dPlus1, static_cast<int>(s->size()));
  if (dPlus1 < 3) return false;
  const int d = dPlus1 - 1;

  // (d+1)→1 requires v to be incident to exactly d+1 top cells.
  std::vector<SimplexPtr> incident;
  for (const auto &s : v->getSimplices())
    if (static_cast<int>(s->size()) == dPlus1) incident.push_back(s);
  if (static_cast<int>(incident.size()) != dPlus1) return false;

  // The d+1 link vertices, with how many cells each appears in.
  std::map<std::uint64_t, VertexPtr> otherVerts;
  std::map<std::uint64_t, int> counts;
  for (const auto &s : incident)
    for (const auto &vert : s->getVertices()) {
      if (vert->getId() == v->getId()) continue;
      otherVerts[vert->getId()] = vert;
      counts[vert->getId()]++;
    }
  if (static_cast<int>(otherVerts.size()) != dPlus1) return false;
  // Each link vertex must be omitted from exactly one cell (present in
  // exactly d): the signature of a stellar star whose link is ∂(newTop).
  for (const auto &[vid, c] : counts) {
    (void)vid;
    if (c != d) return false;
  }

  // The single replacement cell is the simplex on the link vertices.
  VertexPtrs newTop;
  newTop.reserve(static_cast<std::size_t>(dPlus1));
  for (const auto &[vid, vp] : otherVerts) {
    (void)vid;
    newTop.push_back(vp);
  }
  pachner_detail::sortByVertexId(newTop);  // increasing-id orientation
  if (st_->findSimplexByVerts(newTop)) return false;  // would duplicate a cell

  // The structural check forces v's link to be the (d-1)-sphere
  // ∂(newTop), so v is interior; the weld never touches ∂W and there is
  // nothing extra to verify for boundary-fixed mode.

  v_ = v;
  incident_ = std::move(incident);
  spatialVerts_ = std::move(newTop);  // the replacement cell's vertices
  dN41_ = 0;
  logPrefactor_ = 0.0;

  incidentVerts_.reserve(incident_.size());
  for (const auto &s : incident_) {
    const auto &verts = s->getVertices();
    incidentVerts_.emplace_back(verts.begin(), verts.end());
  }

  touchedIds_.reserve(static_cast<std::size_t>(dPlus1) + 1);
  touchedIds_.push_back(v_->getId());
  for (const auto &vp : spatialVerts_) touchedIds_.push_back(vp->getId());

  proposed_ = true;
  return true;
}

void RemoveMove::removeIncidentSubSimplices() {
  // After the incident top cells are gone, every remaining simplex on v_ is a
  // sub-top facet/hinge orphaned by the removal. Snapshot first: removeSimplex
  // mutates v_'s simplex list.
  std::vector<SimplexPtr> sub(v_->getSimplices().begin(),
                              v_->getSimplices().end());
  for (const auto &s : sub) {
    if (!s->isStale()) st_->removeSimplex(s);
  }
}

bool RemoveMove::applyPreGeometric() {
  // Capture the vertex and its incident edges for rollback.  Pre-geometric
  // vertices are coordinate-free, so getCoordinates() would throw; an empty
  // coordinate vector recreates an identical coordinate-independent vertex
  // (rollback's createVertex(id, {}) is the same call buildExplicit makes).
  vertexId_ = v_->getId();
  vertexCoords_.clear();
  for (const auto &e : v_->getInEdges())
    deletedEdges_.push_back({e->getSource(), e->getTarget(),
                             e->getLength(), e->getPhase()});
  for (const auto &e : v_->getOutEdges())
    deletedEdges_.push_back({e->getSource(), e->getTarget(),
                             e->getLength(), e->getPhase()});

  // Remove the d+1 incident cells.
  for (const auto &s : incident_) st_->removeSimplex(s);

  // Drop the sub-simplices (facets/hinges) those cells materialised on v: with
  // v's top cells gone they are orphans, and leaving them registered lets a
  // later materialisation reuse them by fingerprint carrying a now-stale pointer
  // to v once removeIfIsolated frees v and rollback recreates it as a fresh
  // object — which would corrupt the dual-volume / deficit coface walk over the
  // restored star. rollback re-materialises clean ones referencing the new v.
  removeIncidentSubSimplices();

  // Remove edges incident to v from both endpoints + the global list.
  Edges inCopy(v_->getInEdges().begin(), v_->getInEdges().end());
  for (const auto &e : inCopy) {
    e->getSource()->removeOutEdge(e);
    v_->removeInEdge(e);
    st_->absorbRemovedEdgeRevisions(e);
    st_->getEdgeList()->remove(e);
  }
  Edges outCopy(v_->getOutEdges().begin(), v_->getOutEdges().end());
  for (const auto &e : outCopy) {
    e->getTarget()->removeInEdge(e);
    v_->removeOutEdge(e);
    st_->absorbRemovedEdgeRevisions(e);
    st_->getEdgeList()->remove(e);
  }
  (void)st_->removeIfIsolated(v_);

  // Weld in the single replacement cell on the link vertices.
  auto r = st_->createSimplexTracked(spatialVerts_);
  if (r.created) createdSimplexVerts_.push_back(spatialVerts_);
  for (const auto &e : r.newEdges) createdEdges_.push_back(e);

  applied_ = true;
  return true;
}

bool RemoveMove::apply() {
  if (!proposed_ || applied_) return false;
  if (mode_ == PachnerMode::PreGeometric) return applyPreGeometric();

  // 1. Capture edge data BEFORE deletion (for rollback).
  // Mirrors CDT::remove's edge-cleanup loop.
  vertexId_ = v_->getId();
  vertexCoords_ = v_->getCoordinates();

  // Snapshot incident edges (in + out): the full complex squared length
  // AND the U(1) phase, so rollback is bit-exact (#581).  (We can't
  // capture EdgePtr — those slots get freed by EdgeList::remove.)
  for (const auto &e : v_->getInEdges()) {
    deletedEdges_.push_back({e->getSource(), e->getTarget(),
                             e->getLength(), e->getPhase()});
  }
  for (const auto &e : v_->getOutEdges()) {
    deletedEdges_.push_back({e->getSource(), e->getTarget(),
                             e->getLength(), e->getPhase()});
  }

  // 2. Remove the 2d incident simplices.
  for (const auto &s : incident_) st_->removeSimplex(s);

  // 2b. Drop the now-orphaned sub-simplices the removed cells materialised on v
  // (see applyPreGeometric for the stale-pointer rationale): rollback recreates
  // v as a fresh object, so any lingering facet/hinge that still points at the
  // old v would corrupt the restored star's dual/coface walk.
  removeIncidentSubSimplices();

  // 3. Remove edges incident to v from both endpoints + the global list.
  // Mirrors CDT::remove's cleanup.
  Edges inCopy(v_->getInEdges().begin(), v_->getInEdges().end());
  for (const auto &e : inCopy) {
    e->getSource()->removeOutEdge(e);
    v_->removeInEdge(e);
    st_->absorbRemovedEdgeRevisions(e);
    st_->getEdgeList()->remove(e);
  }
  Edges outCopy(v_->getOutEdges().begin(), v_->getOutEdges().end());
  for (const auto &e : outCopy) {
    e->getTarget()->removeInEdge(e);
    v_->removeOutEdge(e);
    st_->absorbRemovedEdgeRevisions(e);
    st_->getEdgeList()->remove(e);
  }
  (void)st_->removeIfIsolated(v_);

  // 4. Create 2 replacement simplices.
  VertexPtrs verts1(spatialVerts_.begin(), spatialVerts_.end());
  verts1.push_back(vertA_);
  VertexPtrs verts2(spatialVerts_.begin(), spatialVerts_.end());
  verts2.push_back(vertB_);

  auto r1 = st_->createSimplexTracked(verts1);
  if (r1.created) createdSimplexVerts_.push_back(verts1);
  for (const auto &e : r1.newEdges) createdEdges_.push_back(e);

  auto r2 = st_->createSimplexTracked(verts2);
  if (r2.created) createdSimplexVerts_.push_back(verts2);
  for (const auto &e : r2.newEdges) createdEdges_.push_back(e);

  applied_ = true;
  return true;
}

void RemoveMove::rollback() {
  if (!applied_) return;

  // 1. Remove the 2 replacement simplices.  Resolve by verts at
  // rollback time (see ShiftMove for the staleness-bug rationale).
  for (const auto &verts : createdSimplexVerts_) {
    if (auto s = st_->findSimplexByVerts(verts)) st_->removeSimplex(s);
  }
  createdSimplexVerts_.clear();

  // 2. Remove freshly-inserted edges (from the replacement simplices).
  pachner_detail::removeAndClearEdges(createdEdges_, st_);

  // 3. Recreate the deleted vertex with its original ID and coordinates.
  v_ = st_->createVertex(vertexId_, vertexCoords_);

  // 4. Reinsert the deleted edges.  All endpoints still exist
  // (only v_ was removed, and we just recreated it).  Need to be
  // careful: the EdgeRecord's source/target pointers refer to
  // pre-deletion VertexPtrs.  v_'s pointer is fresh (from step 3),
  // so swap any reference to the *old* v_ pointer with the new one.
  // Actually — the way createVertex works, it allocates a new Vertex
  // and returns its pointer.  The OLD v_ pointer (from before
  // step 3) is invalid, but we already overwrote v_ with the new
  // pointer in step 3.  EdgeRecord captured pointers BEFORE we
  // overwrote v_, so those still point to the old (deleted) Vertex.
  //
  // To handle this cleanly: identify edges that reference the deleted
  // vertex by ID match (vertexId_) rather than by pointer comparison.
  // For each EdgeRecord:
  //   - If src->getId() == vertexId_ → use new v_ as source, target unchanged
  //   - Else if tgt->getId() == vertexId_ → use new v_ as target
  //   - Else: shouldn't happen (every captured edge was incident to v).
  for (const auto &er : deletedEdges_) {
    VertexPtr src = (er.source->getId() == vertexId_) ? v_ : er.source;
    VertexPtr tgt = (er.target->getId() == vertexId_) ? v_ : er.target;
    // After re-creating the vertex with the original ID, the OLD
    // pointer captured in EdgeRecord may have been freed.  We rely on
    // the ID match above; ID-stable accessors are sufficient.
    // tryAdd's factory unit is the real signed l^2; the exact complex
    // value and the U(1) phase are written onto the fresh edge right
    // after creation so the restore is bit-exact (#581).  An edge that
    // already exists was never deleted, so its values are left alone.
    auto r = st_->getEdgeList()->tryAdd(src, tgt, er.length);
    if (r.second) {
      r.first->setLength(er.length);  // verbatim: branch-exact, no round-trip
      r.first->setPhase(er.phase);
    }
    src->addOutEdge(r.first);
    tgt->addInEdge(r.first);
  }
  deletedEdges_.clear();

  // 5. Recreate the 2d removed simplices.  Their vertex tuples
  // contain the OLD v_ pointer (captured pre-deletion).  Swap to the
  // new pointer based on ID.
  for (const auto &origVerts : incidentVerts_) {
    VertexPtrs verts;
    verts.reserve(origVerts.size());
    for (const auto &vp : origVerts) {
      verts.push_back(vp->getId() == vertexId_ ? v_ : vp);
    }
    st_->createSimplexTracked(verts);
  }

  applied_ = false;
}

std::vector<std::uint64_t> RemoveMove::touchedVertexIds() const {
  return touchedIds_;
}

}  // namespace tessera
