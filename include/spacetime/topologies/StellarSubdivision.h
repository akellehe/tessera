// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_STELLARSUBDIVISION_H
#define TESSERA_STELLARSUBDIVISION_H

#include <memory>

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Stellar subdivision of a triangulation
///
/// Wraps a base topology and refines it by a single stellar subdivision
/// (a \f$ 1 \to (n+1) \f$ Pachner move): one top \f$ n \f$-simplex
/// \f$ \tau = \{v_0, \ldots, v_n\} \f$ is starred at a fresh interior vertex
/// \f$ c \f$ — \f$ \tau \f$ is removed and replaced by the \f$ n+1 \f$ simplices
/// \f$ \{c\} \cup (\tau \setminus \{v_i\}) \f$. To keep the result deterministic
/// the starred simplex is the lexicographically smallest top-simplex tuple.
///
/// Stellar moves are piecewise-linear homeomorphisms, so the subdivided
/// complex is the same manifold as the base with the same homology, but it is
/// a distinct labelled complex (one extra vertex, \f$ n \f$ extra top
/// simplices) and so not isomorphic to the base. That makes it the standard
/// way to produce a second, inequivalent triangulation of a fixture — e.g. a
/// retriangulated \f$ T^3 = \text{StellarSubdivision}(S^1 \times S^1 \times
/// S^1) \f$ — for triangulation-invariance checks. Mirrors
/// :class:`SimplicialProduct` as a composable wrapper over other topologies.
///
/// Exact and pre-geometric (coordinate-free); ``build()`` ignores
/// ``numSimplices``.
///
/// Reference: Pachner, "P.L. homeomorphic manifolds are equivalent by
/// elementary shellings", 1991.
class StellarSubdivision : public Topology {
  public:
    explicit StellarSubdivision(std::shared_ptr<Topology> base)
        : base_(std::move(base)) {}

    /// A stellar subdivision is a PL homeomorphism, so it preserves dimension.
    [[nodiscard]] int dimension() const override { return base_->dimension(); }

    /// Build the base topology and apply one stellar subdivision.
    /// ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;

  private:
    std::shared_ptr<Topology> base_;
};

} // namespace tessera::spacetime

#endif // TESSERA_STELLARSUBDIVISION_H
