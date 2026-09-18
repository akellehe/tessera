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

/// (2d, 2) Pachner move — vertex deletion — with apply / rollback; the (8,2)
/// move in 4D.
///
/// Picks a random vertex with order 2d (all incident top simplices are
/// N41-type), removes the 2d simplices and the vertex, and creates 2
/// replacement simplices.  ``dN0 = -1``; ``dN41 = -(2d-2) = -6`` in 4D;
/// ``dN32 = 0``.  Inverse: :class:`AddMove`.
///
/// Rollback recreates the deleted vertex (original id and coordinates),
/// reinserts the edges incident to it (original complex lengths and U(1)
/// phases) and recreates the 2d removed simplices.  ``apply()`` captures all
/// of that before it mutates the geometry.
///
/// Reference: Ambjorn, Jurkiewicz & Loll, arXiv:hep-th/0105267.
class RemoveMove : public PachnerMove {
public:
  RemoveMove(Spacetime *st, std::mt19937 *rng,
             PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);
  RemoveMove(Spacetime *st, std::uint64_t seed,
             PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);

  bool propose() override;
  /// Propose at a named vertex: \p site is the single vertex id to remove
  /// (see `sitesOn`). Pre-geometric mode only.
  bool proposeAt(const std::vector<std::uint64_t> &site) override;
  /// Every vertex this move could remove, each as a one-element id list. The
  /// (d+1) -> 1 move collapses one vertex, so the site set is the vertices;
  /// the incidence conditions are left to the proposal.
  static std::vector<std::vector<std::uint64_t>> sitesOn(const Spacetime &spacetime);
  int dN0() const override { return -1; }
  int dN41() const override { return dN41_; }
  int dN32() const override { return 0; }
  double metropolisLogPrefactor() const override { return logPrefactor_; }
  bool apply() override;
  void rollback() override;
  bool isApplied() const override { return applied_; }
  std::vector<std::uint64_t> touchedVertexIds() const override;
  /// Canonical name of this move type, for the callers that dispatch on it
  /// (MultiCobordism's move draw, CDT's acceptance-rate accounting).
  static constexpr const char *kMoveType = "remove";
  std::string moveType() const override { return kMoveType; }

private:
  // Pre-geometric (d+1)→1 stellar weld: the inverse of the 1→(d+1) add.
  // Picks a vertex whose star is the cone over the boundary of one
  // d-simplex (degree d+1, link a (d-1)-sphere), deletes it and its d+1
  // cells, and welds in the single cell on the link vertices.  The
  // structural check makes the vertex interior, so the move never touches
  // ∂W.  rollback() is shared with the CDT path: it is generic over the
  // removed-cell count.
  bool proposePreGeometric();
  /// The shared body of both proposals: everything after the vertex is chosen.
  bool proposePreGeometricOn(VertexPtr v);
  bool applyPreGeometric();

  // Unregister every sub-top-dimensional simplex (facet/hinge) still incident
  // to ``v_`` once its top cells are gone. Those are orphans, and leaving them
  // registered would let a later facet materialisation reuse them by
  // fingerprint while they hold a stale pointer to ``v_`` (which rollback
  // recreates as a fresh object), corrupting the restored star's dual/coface
  // walk. Call after removing the incident top cells, before removing ``v_``.
  void removeIncidentSubSimplices();

  /// Tear down the star of ``v_``: remove its incident top cells, drop the
  /// sub-simplices they materialised on it, unlink and delete its edges from
  /// both endpoints and the global list, then drop the vertex itself once it
  /// is isolated. Shared by ``apply`` and ``applyPreGeometric``, which differ
  /// only in what they weld in afterwards.
  void tearDownVertexStar();

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
  // Edge data captured before deletion: the two endpoints, the complex
  // length and the U(1) phase, so rollback restores the edge bit-exactly.
  // The length is stored as a length, not as l^2: the sqrt/square round-trip
  // is not bit-exact and loses the branch.  Storing the full complex value
  // (not only its real part) keeps analytically continued geometry off the
  // real axis and preserves the connection phase across a rejected move.
  // An EdgePtr is not enough — EdgeList::remove invalidates the slot.
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
  // Edges freshly inserted by createSimplexTracked for the 2 replacement
  // simplices.  Usually empty — they reuse existing edges — but tracked so
  // rollback can remove any that were new.
  Edges createdEdges_;
  // Vertex id and coordinates captured for rollback.
  std::uint64_t vertexId_ = 0;
  std::vector<double> vertexCoords_;
};

}  // namespace tessera

#endif  // TESSERA_PACHNER_REMOVEMOVE_H
