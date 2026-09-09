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

/// (2, 2d) Pachner add (vertex insertion) with apply / rollback.
///
/// Picks a random N41 top simplex, finds its spatial face and the
/// adjacent simplex of opposite orientation.  Inserts a new vertex at
/// the shared spatial time slice, replacing the 2 simplices with 2d
/// new ones.  ``dN0 = +1``; ``dN41 = +(2d - 2) = +6`` in 4D;
/// ``dN32 = 0``.
///
/// Vertex relabeling: after the move commits, the new vertex's ID is
/// optionally swapped with a randomly-chosen existing vertex (per
/// [BGL] Sec. 2.2.1).  Toggle via ``setRelabelEnabled(bool)`` at
/// construction time; default is enabled.  Rollback un-swaps before
/// removing the new vertex.
class AddMove : public PachnerMove {
public:
  AddMove(Spacetime *st, std::mt19937 *rng, bool relabelEnabled = true,
          PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);
  AddMove(Spacetime *st, std::uint64_t seed, bool relabelEnabled = true,
          PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);

  bool propose() override;
  /// Propose at a NAMED top cell: \p site is that cell's vertex ids, in any
  /// order (see `sitesOn`). Pre-geometric mode only -- the CDT path's target
  /// is a causal pair, not a single cell, and is not addressable this way.
  bool proposeAt(const std::vector<std::uint64_t> &site) override;
  /// Every cell this move could subdivide, each as its vertex ids. The 1->(d+1)
  /// stellar move lives inside ONE cell, so the site set is exactly the top
  /// cells big enough to subdivide.
  static std::vector<std::vector<std::uint64_t>> sitesOn(const Spacetime &spacetime);
  int dN0() const override { return 1; }
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
  static constexpr const char *kMoveType = "add";
  std::string moveType() const override { return kMoveType; }

private:
  // Pre-geometric 1→(d+1) stellar move: insert a fresh interior vertex
  // into a single top cell and cone it over the cell's facets, replacing
  // 1 cell with d+1.  Always interior (it never touches ∂W), so it is
  // unconditionally boundary-fixed-safe.
  bool proposePreGeometric();
  /// The shared body of both proposals: everything after the cell is chosen.
  /// `propose` draws the cell, `proposeAt` is handed it, and neither has its
  /// own copy of what follows.
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
