// Copyright (c) 2026 Twin Vector Labs LLC.
// All rights reserved.

#ifndef TESSERA_SIMPLICIALPRODUCT_H
#define TESSERA_SIMPLICIALPRODUCT_H

#include <memory>

#include "Topology.h"

// === tessera subsystem ns fwd-decls ===
namespace tessera::spacetime {
class Spacetime;

/// # Simplicial product of two triangulations
///
/// The product \f$ K \times L \f$ of two triangulations, triangulated by the
/// standard staircase (Eilenberg–Zilber) construction. Vertices are pairs
/// \f$ (u, v) \in V(K)
/// \times V(L) \f$; a product cell \f$ \sigma^p \times \tau^q \f$ is cut into
/// \f$ \binom{p+q}{p} \f$ simplices of dimension \f$ p+q \f$, one per monotone
/// lattice path \f$ (0,0) \to (p,q) \f$, using each factor's vertex order so the
/// pieces glue consistently across shared faces.
///
/// Used to build product manifolds for the cobordism fixtures — e.g.
/// \f$ S^2 \times S^2 = \partial\Delta^3 \times \partial\Delta^3 \f$ (a closed
/// oriented 4-manifold with \f$ b_2 = 2 \f$ and signature 0), or \f$ T^2 =
/// S^1 \times S^1 \f$. Exact and pre-geometric (coordinate-free); ``build()``
/// ignores ``numSimplices``.
///
/// The factor vertex set is taken to be \f$ \{0, \ldots, |V|-1\} \f$ (as
/// produced by the coordinate-free fixture topologies); product vertex
/// \f$ (u, v) \f$ is assigned id \f$ u \cdot |V(L)| + v \f$.
class SimplicialProduct : public Topology {
  public:
    SimplicialProduct(std::shared_ptr<Topology> left, std::shared_ptr<Topology> right)
        : left_(std::move(left)), right_(std::move(right)) {}

    /// \f$ \dim(K \times L) = \dim K + \dim L \f$.
    [[nodiscard]] int dimension() const override {
      return left_->dimension() + right_->dimension();
    }

    /// Build \f$ K \times L \f$. ``numSimplices`` is ignored.
    void build(Spacetime *spacetime, int numSimplices) override;

  private:
    std::shared_ptr<Topology> left_;
    std::shared_ptr<Topology> right_;
};

} // namespace tessera::spacetime

#endif // TESSERA_SIMPLICIALPRODUCT_H
