// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_PACHNER_ADDMOVE_H
#define TESSERA_PACHNER_ADDMOVE_H

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

/// (2, 2d) Pachner move — vertex insertion — with apply / rollback; the
/// (2,8) move in 4D.
///
/// Picks a random N41 top simplex, finds its spatial facet and the adjacent
/// simplex of opposite orientation, then inserts a vertex on that shared
/// spatial slice, replacing the 2 simplices by 2d new ones.
/// ``dN0 = +1``; ``dN41 = +(2d - 2) = +6`` in 4D; ``dN32 = 0``.
/// Inverse: :class:`RemoveMove`.
///
/// After the move commits, the new vertex's id is optionally swapped with a
/// randomly chosen existing vertex, so a vertex's id carries no trace of when
/// it was inserted. Selected by the constructor argument ``relabelEnabled``
/// (default true); rollback un-swaps before removing the new vertex.
///
/// Reference: Ambjorn, Jurkiewicz & Loll, arXiv:hep-th/0105267.
class AddMove : public PachnerMove {
public:
  AddMove(Spacetime *st, std::mt19937 *rng, bool relabelEnabled = true,
          PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);
  AddMove(Spacetime *st, std::uint64_t seed, bool relabelEnabled = true,
          PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);

  bool propose() override;
  /// Propose at a named top cell: \p site is that cell's vertex ids, in any
  /// order (see `sitesOn`). Pre-geometric mode only; the CDT path targets a
  /// causal pair of cells rather than a single cell.
  bool proposeAt(const std::vector<std::uint64_t> &site) override;
  /// Every cell this move could subdivide, each as its vertex ids. The
  /// 1 -> (d+1) stellar move lives inside a single cell, so the site set is
  /// the top cells large enough to subdivide.
  static std::vector<std::vector<std::uint64_t>> sitesOn(const Spacetime &spacetime);
  int dN0() const override { return 1; }
  int dN41() const override { return dN41_; }
  int dN32() const override { return 0; }
  double metropolisLogPrefactor() const override { return logPrefactor_; }
  bool apply() override;
  void rollback() override;
  bool isApplied() const override { return applied_; }
  std::vector<std::uint64_t> touchedVertexIds() const override;
  /// Canonical name of this move type, for the callers that dispatch on it
  /// (MultiCobordism's move draw, CDT's acceptance-rate accounting).
  static constexpr const char *kMoveType = "add";
  std::string moveType() const override { return kMoveType; }

private:
  // Pre-geometric 1→(d+1) stellar move: insert a fresh interior vertex into a
  // single top cell and cone it over the cell's facets, replacing 1 cell with
  // d+1.  It never touches ∂W, so it is always safe in boundary-fixed mode.
  bool proposePreGeometric();
  /// Shared body of both proposals: everything after the cell is chosen.
  /// `propose` draws the cell; `proposeAt` is handed it.
  bool proposePreGeometricOn(SimplexPtr sigma);
  bool applyPreGeometric();
  void rollbackPreGeometric();

  Spacetime *st_;
  std::unique_ptr<std::mt19937> ownedRng_;
  std::mt19937 *rng_;
  bool relabelEnabled_;

  // Filled by propose()
  bool proposed_ = false;
  SimplexPtr sigma_, sigmaAdj_, spatialFacet_;
  VertexPtr vertA_, vertB_;
  VertexPtrs spatialVerts_;
  VertexPtrs sigmaVerts_, sigmaAdjVerts_;
  double spatialTime_ = 0.0;
  std::vector<std::uint64_t> touchedIds_;
  int dN41_ = 0;
  double logPrefactor_ = 0.0;

  // Filled by apply()
  bool applied_ = false;
  VertexPtr newVert_;                       // the inserted vertex
  VertexPtr swapPartner_ = nullptr;         // null if no relabel happened
  // Vertex tuples of simplices we actually created (see ShiftMove for the
  // staleness-bug rationale).
  std::vector<VertexPtrs> createdSimplexVerts_;
  Edges createdEdges_;
};

}  // namespace tessera

#endif  // TESSERA_PACHNER_ADDMOVE_H
