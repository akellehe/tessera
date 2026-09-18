// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_PACHNER_SHIFTMOVE_H
#define TESSERA_PACHNER_SHIFTMOVE_H

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

/// (3,3) Pachner move (shift) with apply / rollback.
///
/// Removes 3 d-simplices sharing a (d-2)-face (hinge) and creates 3 new
/// simplices sharing the complementary (d-2)-face.  ``dN0 = 0`` and
/// ``dN41 + dN32 = 0``: the top-simplex count is preserved.  Self-inverse —
/// the inverse is another shift on the same hinge with the roles swapped.
///
/// The Metropolis log prefactor is 0: selection is symmetric under that
/// swap.  ``CDT::shiftImpl`` is the non-transactional version of the same
/// move.
///
/// Reference: Ambjorn, Jurkiewicz & Loll, arXiv:hep-th/0105267.
class ShiftMove : public PachnerMove {
public:
  /// Construct with a caller-owned RNG (used by CDT internally so all
  /// moves share the same Markov chain).
  ShiftMove(Spacetime *st, std::mt19937 *rng,
            PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);

  /// Construct with an internally-owned RNG seeded from ``seed``
  /// (convenient for one-shot use from Python tests).
  ShiftMove(Spacetime *st, std::uint64_t seed,
            PachnerMode mode = PachnerMode::CDT, bool boundaryFixed = false);

  bool propose() override;
  int dN0() const override { return 0; }
  int dN41() const override { return dN41_; }
  int dN32() const override { return dN32_; }
  double metropolisLogPrefactor() const override { return 0.0; }
  bool apply() override;
  void rollback() override;
  bool isApplied() const override { return applied_; }
  std::vector<std::uint64_t> touchedVertexIds() const override;
  /// Canonical name of this move type, for the callers that dispatch on it
  /// (MultiCobordism's move draw, CDT's acceptance-rate accounting).
  static constexpr const char *kMoveType = "shift";
  std::string moveType() const override { return kMoveType; }

private:
  Spacetime *st_;
  std::unique_ptr<std::mt19937> ownedRng_;  // non-null iff this move owns its RNG
  std::mt19937 *rng_;                        // points to caller-owned or ownedRng_

  // Filled in by propose()
  bool proposed_ = false;
  std::vector<VertexPtrs> oldSimplexVerts_;   // 3 vertex tuples (d+1 each)
  std::vector<VertexPtrs> newSimplexVerts_;   // 3 vertex tuples
  std::vector<std::uint64_t> touchedIds_;     // d+2 vertex ids
  int dN41_ = 0;
  int dN32_ = 0;

  // Filled in by apply() — used by rollback()
  bool applied_ = false;
  std::vector<SimplexPtr> oldSimplices_;      // captured pre-apply
  // Vertex tuples of the simplices actually created (createSimplexTracked
  // returned created=true).  Stored as verts rather than SimplexPtr so
  // rollback survives another move deleting the underlying Simplex in
  // between: the pointer would dangle, the verts stay valid.  Resolved
  // through Spacetime::findSimplexByVerts.
  std::vector<VertexPtrs> createdSimplexVerts_;
  Edges createdEdges_;
};

}  // namespace tessera

#endif  // TESSERA_PACHNER_SHIFTMOVE_H
