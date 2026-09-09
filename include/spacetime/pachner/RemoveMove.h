// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_PACHNER_REMOVEMOVE_H
#define TESSERA_PACHNER_REMOVEMOVE_H

#include <complex>
#include <memory>
#include <random>
#include <vector>

#include "mesh/ForwardDeclarations.h"
#include "spacetime/PachnerMove.h"
#include "spacetime/Spacetime.h"

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

/// (2d, 2) Pachner remove (vertex deletion) with apply / rollback.
///
/// Picks a random vertex with order 2d (all incident top simplices are
/// N41-type), removes the 2d simplices and the vertex, and creates 2
/// replacement simplices.  ``dN0 = -1``; ``dN41 = -(2d-2) = -6`` in 4D;
/// ``dN32 = 0``.  Inverse: :class:`AddMove`.
///
/// Rollback is the most involved of the move types: it has to recreate
/// the deleted vertex (with its original ID and coordinates), reinsert
/// the d edges incident to it (with their original squared lengths),
/// and recreate the 2d removed simplices.  All of that data is captured
/// by ``apply()`` before the geometry is mutated.
class RemoveMove : public PachnerMove {
public:
  RemoveMove(Spacetime *st, std::mt19937 *rng,
             PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);
  RemoveMove(Spacetime *st, std::uint64_t seed,
             PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);

  bool propose() override;
  /// Propose at a NAMED vertex: \p site is the single vertex id to remove
  /// (see `sitesOn`). Pre-geometric mode only.
  bool proposeAt(const std::vector<std::uint64_t> &site) override;
  /// Every vertex this move could remove, each as a one-element id list. The
  /// (d+1)->1 move collapses one vertex, so the site set is the vertices; the
  /// incidence conditions are left to the proposal, which already checks them.
  static std::vector<std::vector<std::uint64_t>> sitesOn(const Spacetime &spacetime);
  int dN0() const override { return -1; }
  int dN41() const override { return dN41_; }
  int dN32() const override { return 0; }
  double metropolisLogPrefactor() const override { return logPrefactor_; }
  bool apply() override;
  void rollback() override;
  bool isApplied() const override { return applied_; }
  std::vector<std::uint64_t> touchedVertexIds() const override;
  /// The canonical name of this move type, defined ONCE here so callers
  /// that dispatch on it (MultiCobordism's move draw, CDT's acceptance-rate
  /// accounting) reference this rather than re-spelling the literal.
  static constexpr const char *kMoveType = "remove";
  std::string moveType() const override { return kMoveType; }

private:
  // Pre-geometric (d+1)→1 stellar weld: the inverse of the 1→(d+1)
  // add.  Picks a vertex whose star is exactly the cone over the
  // boundary of one d-simplex (degree d+1, link = a (d-1)-sphere),
  // deletes it and its d+1 cells, and welds in the single cell on the
  // link vertices.  The structural check guarantees the vertex is
  // interior, so the move never touches ∂W.  rollback() is shared with
  // the CDT path (it is generic over the removed-cell count).
  bool proposePreGeometric();
  /// The shared body of both proposals: everything after the vertex is chosen.
  bool proposePreGeometricOn(VertexPtr v);
  bool applyPreGeometric();

  // Unregister every sub-top-dimensional simplex (facet/hinge) still incident to
  // ``v_`` after its top cells have been removed. Those are orphans now, and
  // leaving them registered would let a later facet materialisation reuse them
  // by fingerprint while they hold a stale pointer to ``v_`` (which rollback
  // recreates as a fresh object), corrupting the restored star's dual/coface
  // walk. Call after removing the incident top cells, before removing ``v_``.
  void removeIncidentSubSimplices();

  Spacetime *st_;
  std::unique_ptr<std::mt19937> ownedRng_;
  std::mt19937 *rng_;

  // Filled by propose()
  bool proposed_ = false;
  VertexPtr v_;                                  // vertex to be removed
  std::vector<SimplexPtr> incident_;             // 2d incident top simplices
  std::vector<VertexPtrs> incidentVerts_;        // their vertex tuples
  VertexPtr vertA_, vertB_;                      // the two non-spatial vertices
  VertexPtrs spatialVerts_;                      // the d spatial vertices
  std::vector<std::uint64_t> touchedIds_;
  int dN41_ = 0;
  double logPrefactor_ = 0.0;

  // Captured by apply() — for rollback
  bool applied_ = false;
  // Edge data captured before deletion: (sourcePtr, targetPtr, the full
  // COMPLEX length — recorded as the LENGTH, not l^2: the sqrt/square
  // round-trip is not bit-exact and loses the branch (#639) — and the U(1)
  // phase) so rollback restores the
  // edge bit-exactly — a Re-only record silently projected analytically
  // continued geometry onto the real axis and dropped the connection
  // phase on every rejected move (#581).  EdgePtr alone is not enough
  // because EdgeList::remove invalidates the slot.
  struct EdgeRecord {
    VertexPtr source;
    VertexPtr target;
    std::complex<double> length;
    std::complex<double> phase;
  };
  std::vector<EdgeRecord> deletedEdges_;
  // Vertex tuples of the 2 replacement simplices we actually created in
  // apply() (see ShiftMove for the staleness-bug rationale).
  std::vector<VertexPtrs> createdSimplexVerts_;
  // Edges freshly inserted by createSimplexTracked when we built the
  // 2 replacement simplices (likely empty since they reuse existing
  // edges, but track for safety).
  Edges createdEdges_;
  // Captured vertex coordinates for rollback (unique vertex with
  // captured ID).
  std::uint64_t vertexId_ = 0;
  std::vector<double> vertexCoords_;
};

}  // namespace tessera

#endif  // TESSERA_PACHNER_REMOVEMOVE_H
